import json
import os
import sys
import time
from statistics import mean

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generation import nodes as gen_nodes
from src.generation.graph import build_graph
from src.retrieval.retriever import Retriever

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
EVAL_SET_PATH = os.path.join(ROOT, "eval", "eval_set.json")
GOLD_PATH = os.path.join(BASE, "gold.json")
RETRIEVAL_PATH = os.path.join(BASE, "retrieval_metrics.json")
ANSWERS_PATH = os.path.join(BASE, "answers_for_review.json")
STRESS_PATH = os.path.join(BASE, "stress_results.json")
REPORT_PATH = os.path.join(BASE, "REPORT.md")

TARGET_QUESTION_IDS = [1, 2, 7]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_pair(pair):
    return (pair.get("spec_id"), pair.get("clause_id"))


def pair_in_top(results, target_pairs, limit):
    seen = set()
    for item in results[:limit]:
        seen.add((item.get("spec_id"), item.get("clause_id")))
    return sum(1 for p in target_pairs if p in seen)


def compute_retrieval_metrics(target_ids=None):
    eval_set = load_json(EVAL_SET_PATH)
    gold = load_json(GOLD_PATH)
    gold_by_question = {entry["question"]: entry for entry in gold}
    gold_by_id = {entry["question_id"]: entry for entry in gold}

    selected = []
    for entry in eval_set:
        question = entry["question"]
        gold_entry = gold_by_question.get(question)
        if not gold_entry:
            continue
        qid = gold_entry["question_id"]
        if target_ids is not None and qid not in target_ids:
            continue
        if gold_entry.get("retrieval_scored") is False:
            continue
        gold_pairs = [normalize_pair(p) for p in gold_entry.get("gold_pairs", [])]
        if not gold_pairs:
            continue
        selected.append((entry, qid, gold_entry, gold_pairs))

    retriever = Retriever()
    metrics = []
    per_question = []
    for entry, qid, gold_entry, gold_pairs in selected:
        results = retriever.search(entry["question"], top_k=20)
        recall5 = pair_in_top(results, gold_pairs, 5) / len(gold_pairs)
        recall20 = pair_in_top(results, gold_pairs, 20) / len(gold_pairs)
        mrr = 0.0
        for rank, item in enumerate(results[:20], start=1):
            if (item.get("spec_id"), item.get("clause_id")) in gold_pairs:
                mrr = 1.0 / rank
                break
        per_question.append({
            "question_id": qid,
            "question": entry["question"],
            "type": entry["type"],
            "gold_pairs": gold_pairs,
            "recall_at_5": recall5,
            "recall_at_20": recall20,
            "mrr": mrr,
            "top_5": [{"spec_id": r.get("spec_id"), "clause_id": r.get("clause_id")} for r in results[:5]],
        })
        metrics.append({
            "question_id": qid,
            "question": entry["question"],
            "type": entry["type"],
            "recall_at_5": recall5,
            "recall_at_20": recall20,
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


def run_end_to_end(target_ids=None):
    eval_set = load_json(EVAL_SET_PATH)
    gold = load_json(GOLD_PATH)
    gold_by_question = {entry["question"]: entry for entry in gold}
    selected = [entry for entry in eval_set if gold_by_question.get(entry["question"]) and (target_ids is None or gold_by_question[entry["question"]]["question_id"] in target_ids)]

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
        for idx, entry in enumerate(selected, start=1):
            question = entry["question"]
            print(f"[{idx}/{len(selected)}] {question}")
            chunks = retriever.search(question, top_k=5)
            state = {"query": question, "chunks": chunks, "retries": 0}
            final_state = state.copy()
            for output in app.stream(state):
                for _, value in output.items():
                    final_state.update(value)
            result = {
                "question_id": gold_by_question[question]["question_id"],
                "question": question,
                "type": entry["type"],
                "generated_answer": final_state.get("answer", ""),
                "verification_passed": bool(final_state.get("verification_passed", False)),
                "retries": final_state.get("retries", 0),
                "sources": [{"spec_id": c.get("spec_id"), "clause_id": c.get("clause_id")} for c in chunks],
                "feedback": final_state.get("feedback", ""),
            }
            results.append(result)
            time.sleep(1)
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
        "What is the purpose of the Network Slice Selection Assistance Information (NSSAI)?",
    ]
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        retriever.search(q, top_k=5)
        dt = time.perf_counter() - t0
        latencies.append({"query": q, "latency_seconds": dt})
    summary = {
        "n_runs": len(latencies),
        "avg_seconds": sum(item["latency_seconds"] for item in latencies) / len(latencies) if latencies else 0.0,
        "latencies": latencies,
    }
    with open(STRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_report(retrieval_summary, per_question, end_to_end_results, stress_summary):
    total = len(end_to_end_results)
    verified = sum(1 for r in end_to_end_results if r["verification_passed"])
    report = [
        "# Minimal baseline resume report",
        "",
        "## Retrieval quality",
        f"- Questions evaluated: {retrieval_summary['n_questions']}",
        f"- Mean recall@5: {retrieval_summary['mean_recall_at_5']:.3f}",
        f"- Mean recall@20: {retrieval_summary['mean_recall_at_20']:.3f}",
        f"- Mean MRR: {retrieval_summary['mean_mrr']:.3f}",
        "",
        "## End-to-end sample run",
        f"- Total questions: {total}",
        f"- Verified: {verified}/{total}",
        "",
        "## Stress sample",
        f"- Total runs: {stress_summary['n_runs']}",
        f"- Average latency: {stress_summary['avg_seconds']:.3f}s",
        "",
        "## Per-question highlights",
    ]
    for item in per_question:
        report.append(f"- Q{item['question_id']} ({item['type']}): recall@5={item['recall_at_5']:.3f}, recall@20={item['recall_at_20']:.3f}, MRR={item['mrr']:.3f}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


def main():
    retrieval_summary, per_question = compute_retrieval_metrics(TARGET_QUESTION_IDS)
    end_to_end_results, usage_log = run_end_to_end(TARGET_QUESTION_IDS)
    stress_summary = run_stress_test()
    build_report(retrieval_summary, per_question, end_to_end_results, stress_summary)
    print(json.dumps({
        "retrieval_summary": retrieval_summary,
        "questions_run": len(end_to_end_results),
        "usage_log": usage_log,
        "stress_summary": stress_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
