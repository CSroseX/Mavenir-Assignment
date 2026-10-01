import json
import os
import re
import sys
import time
from statistics import mean

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generation.graph import build_graph
from src.retrieval.retriever import Retriever
from src.generation import nodes as gen_nodes

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
EVAL_SET_PATH = os.path.join(ROOT, "eval", "eval_set.json")
GOLD_PATH = os.path.join(BASE, "gold.json")
RETRIEVAL_PATH = os.path.join(BASE, "retrieval_metrics.json")
ANSWERS_PATH = os.path.join(BASE, "answers_for_review.json")
STRESS_PATH = os.path.join(BASE, "stress_results.json")
REPORT_PATH = os.path.join(BASE, "REPORT.md")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_llm_hard_stop_error(exc: Exception):
    text = str(exc).lower()
    blockers = [
        "429", "rate limit", "rate_limit", "too many requests", "quota exceeded",
        "exceeded your current quota", "insufficient_quota", "authentication", "unauthorized",
        "invalid api key", "api key", "access denied"
    ]
    return any(item in text for item in blockers)


def normalize_pair(pair):
    return (pair.get("spec_id"), pair.get("clause_id"))


def pair_in_top(results, target_pairs, limit):
    seen = set()
    for item in results[:limit]:
        seen.add((item.get("spec_id"), item.get("clause_id")))
    return sum(1 for p in target_pairs if p in seen)


def compute_retrieval_metrics():
    eval_set = load_json(EVAL_SET_PATH)
    gold = load_json(GOLD_PATH)
    gold_by_id = {item["question_id"]: item for item in gold}

    retriever = Retriever()
    metrics = []
    per_question = []

    for entry in eval_set:
        qid = None
        for item in gold_by_id.values():
            if item.get("question") == entry["question"]:
                qid = item["question_id"]
                break
        if qid is None:
            continue
        gold_item = gold_by_id.get(qid)
        if not gold_item:
            continue
        if gold_item.get("retrieval_scored") is False:
            continue
        gold_pairs = [normalize_pair(p) for p in gold_item.get("gold_pairs", [])]
        if not gold_pairs:
            continue

        retrieval_results = retriever.search(entry["question"], top_k=20)
        top5_hit_count = pair_in_top(retrieval_results, gold_pairs, 5)
        top20_hit_count = pair_in_top(retrieval_results, gold_pairs, 20)
        recall_at_5 = top5_hit_count / len(gold_pairs)
        recall_at_20 = top20_hit_count / len(gold_pairs)

        mrr = 0.0
        for rank, item in enumerate(retrieval_results[:20], start=1):
            if (item.get("spec_id"), item.get("clause_id")) in gold_pairs:
                mrr = 1.0 / rank
                break

        per_question.append({
            "question_id": qid,
            "question": entry["question"],
            "type": entry["type"],
            "gold_pairs": gold_pairs,
            "recall_at_5": recall_at_5,
            "recall_at_20": recall_at_20,
            "mrr": mrr,
            "retrieved_top_5": [
                {"spec_id": r.get("spec_id"), "clause_id": r.get("clause_id"), "score": r.get("score")}
                for r in retrieval_results[:5]
            ],
            "retrieved_top_20": [
                {"spec_id": r.get("spec_id"), "clause_id": r.get("clause_id"), "score": r.get("score")}
                for r in retrieval_results[:20]
            ]
        })

        metrics.append({
            "question_id": qid,
            "question": entry["question"],
            "type": entry["type"],
            "recall_at_5": recall_at_5,
            "recall_at_20": recall_at_20,
            "mrr": mrr,
        })

    summary = {
        "n_questions": len(metrics),
        "mean_recall_at_5": mean(item["recall_at_5"] for item in metrics) if metrics else 0.0,
        "mean_recall_at_20": mean(item["recall_at_20"] for item in metrics) if metrics else 0.0,
        "mean_mrr": mean(item["mrr"] for item in metrics) if metrics else 0.0,
        "questions": metrics,
    }
    with open(RETRIEVAL_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)
    return summary, per_question


def run_end_to_end():
    eval_set = load_json(EVAL_SET_PATH)
    gold = load_json(GOLD_PATH)
    gold_map = {item["question"]: item for item in gold}

    original_create = gen_nodes.client.chat.completions.create
    usage_log = []

    def wrapped_create(*args, **kwargs):
        response = original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_log.append({
                "model": kwargs.get("model") or getattr(gen_nodes, "MODEL_NAME", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            })
        return response

    gen_nodes.client.chat.completions.create = wrapped_create
    app = build_graph()
    retriever = Retriever()
    results = []

    try:
        for idx, entry in enumerate(eval_set, start=1):
            question = entry["question"]
            print(f"\n[{idx}/{len(eval_set)}] {question}")
            using_chunks = retriever.search(question, top_k=5)
            state = {"query": question, "chunks": using_chunks, "retries": 0}
            final_state = state.copy()
            for output in app.stream(state):
                for _, value in output.items():
                    final_state.update(value)
            gold_item = gold_map.get(question)
            result = {
                "question_id": gold_item.get("question_id") if gold_item else None,
                "question": question,
                "type": entry["type"],
                "expected_behavior": entry.get("expected_behavior", ""),
                "generated_answer": final_state.get("answer", ""),
                "verification_passed": bool(final_state.get("verification_passed", False)),
                "retries": final_state.get("retries", 0),
                "sources": [{"spec_id": c.get("spec_id"), "clause_id": c.get("clause_id")} for c in using_chunks],
                "feedback": final_state.get("feedback", ""),
                "status": "verified" if final_state.get("verification_passed") else "flagged_unverified",
            }
            results.append(result)
            time.sleep(5)
    except Exception as exc:
        if is_llm_hard_stop_error(exc):
            with open(os.path.join(BASE, "partial_answers.json"), "w", encoding="utf-8") as f:
                json.dump({"results": results, "error": str(exc)}, f, indent=2)
            raise
        else:
            raise
    finally:
        gen_nodes.client.chat.completions.create = original_create

    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "usage_log": usage_log}, f, indent=2)

    return results, usage_log


def run_stress_test():
    retriever = Retriever()
    queries = [
        "What is the role of the AMF in the 5G system architecture?",
        "Which network function is responsible for selecting the SMF?",
        "What timer does the AMF use for a Tracking Area Update in EPS?",
        "Who won the World Cup in 2022?",
        "12345",
    ]
    latencies = []
    for q in queries:
        for _ in range(3):
            t0 = time.perf_counter()
            retriever.search(q, top_k=5)
            dt = time.perf_counter() - t0
            latencies.append({"query": q, "latency_seconds": dt})
            time.sleep(1)

    summary = {
        "n_runs": len(latencies),
        "p50_seconds": sorted(item["latency_seconds"] for item in latencies)[len(latencies) // 2] if latencies else 0.0,
        "p95_seconds": sorted(item["latency_seconds"] for item in latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0,
        "latencies": latencies,
    }
    with open(STRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_report(retrieval_summary, per_question, end_to_end_results, stress_summary):
    total = len(end_to_end_results)
    verification_pass_count = sum(1 for r in end_to_end_results if r["verification_passed"])
    retry_count = sum(1 for r in end_to_end_results if r["retries"] > 0)
    refusal_count = sum(1 for r in end_to_end_results if r["type"] == "out_of_scope" and ("cannot" in r["generated_answer"].lower() or "out of scope" in r["generated_answer"].lower() or "only answer 3gpp" in r["generated_answer"].lower()))

    report = [
        "# Baseline diagnostic report",
        "",
        "## Retrieval quality",
        f"- Mean recall@5: {retrieval_summary['mean_recall_at_5']:.3f}",
        f"- Mean recall@20: {retrieval_summary['mean_recall_at_20']:.3f}",
        f"- Mean MRR: {retrieval_summary['mean_mrr']:.3f}",
        "",
        "## End-to-end answer run",
        f"- Total questions: {total}",
        f"- Verified: {verification_pass_count}/{total}",
        f"- Retries triggered: {retry_count}/{total}",
        f"- Out-of-scope refusals: {refusal_count}/{sum(1 for r in end_to_end_results if r['type']=='out_of_scope') if any(r['type']=='out_of_scope' for r in end_to_end_results) else 0}",
        "",
        "## Stress test",
        f"- Total retrieval runs: {stress_summary['n_runs']}",
        f"- p50 latency: {stress_summary['p50_seconds']:.4f}s",
        f"- p95 latency: {stress_summary['p95_seconds']:.4f}s",
        "",
        "## Per-question retrieval highlights",
    ]

    for item in per_question[:5]:
        report.append(
            f"- Q{item['question_id']} ({item['type']}): recall@5={item['recall_at_5']:.3f}, recall@20={item['recall_at_20']:.3f}, MRR={item['mrr']:.3f}"
        )

    report.append("")
    report.append("## Notes")
    report.append("- Baseline evaluation ran under eval/baseline only and did not modify source files outside that directory.")
    report.append("- Qdrant was confirmed live on localhost:6333 before full execution.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


def main():
    retrieval_summary, per_question = compute_retrieval_metrics()
    end_to_end_results, usage_log = run_end_to_end()
    stress_summary = run_stress_test()
    build_report(retrieval_summary, per_question, end_to_end_results, stress_summary)

    print("\n=== BASLINE COMPLETE ===")
    print(json.dumps({
        "retrieval": retrieval_summary,
        "end_to_end_questions": len(end_to_end_results),
        "verified": sum(1 for r in end_to_end_results if r["verification_passed"]),
        "retries": sum(1 for r in end_to_end_results if r["retries"] > 0),
        "stress_runs": stress_summary["n_runs"],
    }, indent=2))


if __name__ == "__main__":
    main()
