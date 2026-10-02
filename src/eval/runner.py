"""Eval runner: executes the golden set through the generate/verify graph.

Rewrite of eval/run_eval.py (kept as-is; this module supersedes it going
forward). Three defects fixed:

1. Resume was positional (`start_index = len(results)`), and never
   validated that `results[i].question == eval_set[i].question`. If the
   golden set were ever reordered or edited, a resumed run would silently
   pair old results with the wrong questions. This runner resumes by
   matching each golden-set `id` against results already on disk, and
   refuses to resume (raises) if the golden set's content hash has
   changed since those results were written — reordering or editing
   questions invalidates a partial run instead of silently corrupting it.

2. A corrupted results file was handled with a bare `except: pass`,
   which silently discarded the parse error and restarted from zero,
   then overwrote the previous (possibly salvageable) file. This runner
   raises a clear error on malformed JSON instead of guessing.

3. Writes were direct (`open(path, "w")`), so a crash mid-write could
   leave a half-written, unparseable results file — which the above
   bare `except: pass` would then have silently wiped on the next run.
   This runner writes to a temp file and atomically replaces the target,
   so a crash mid-write never corrupts the previous good file.

Also adds --limit and --types for cheap iteration (a 3-question subset
instead of a full, rate-limited 18-question run), and --types/--limit
apply to the resume logic too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import get_settings
from src.generation.graph import build_graph
from src.retrieval.retriever import Retriever

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "golden" / "golden_set.v1.json"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "runner_results.json"

REFUSAL_KEYWORDS = [
    "cannot answer", "out of scope", "only answer 3gpp",
    "sorry", "cannot provide", "not provided in the",
    "do not have information", "not supported by",
]


class ResultsFileCorruptError(Exception):
    """Raised when an existing results file exists but is not valid JSON."""


class GoldenSetDriftError(Exception):
    """Raised when resuming against a results file whose recorded golden-set
    hash no longer matches the current golden set on disk."""


def is_refusal(answer: str) -> bool:
    lower_ans = answer.lower()
    return any(kw in lower_ans for kw in REFUSAL_KEYWORDS)


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def golden_set_hash(golden_set: List[Dict[str, Any]]) -> str:
    """Content hash of the golden set, used to detect drift between a
    results file's recorded hash and the current golden set on disk."""
    canonical = json.dumps(golden_set, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_existing_results(path: Path, expected_hash: str) -> Dict[int, Dict[str, Any]]:
    """Load prior results keyed by golden-set id. Returns {} if no file
    exists. Raises ResultsFileCorruptError on malformed JSON instead of
    silently discarding it. Raises GoldenSetDriftError if the file's
    recorded golden_set_hash doesn't match the current golden set."""
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ResultsFileCorruptError(
            f"{path} exists but is not valid JSON ({e}). "
            "Refusing to silently discard it — inspect or remove it manually before re-running."
        ) from e

    recorded_hash = payload.get("golden_set_hash")
    if recorded_hash is not None and recorded_hash != expected_hash:
        raise GoldenSetDriftError(
            f"{path} was written against a different golden set (hash {recorded_hash[:12]}...) "
            f"than the one currently on disk (hash {expected_hash[:12]}...). "
            "Resuming would pair stale results with possibly-reordered or edited questions. "
            "Delete the results file to force a clean run."
        )

    results_by_id = {}
    for r in payload.get("results", []):
        results_by_id[r["id"]] = r
    return results_by_id


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write payload as JSON to path atomically: write to a temp file in
    the same directory, then os.replace (atomic on POSIX and Windows) so a
    crash mid-write never leaves path itself truncated or unparseable."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def run_one_question(app, retriever: Retriever, item: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    question = item["question"]
    chunks = retriever.search(question, top_k=top_k)

    inputs = {"query": question, "chunks": chunks}
    final_state = inputs.copy()
    for output in app.stream(inputs):
        for _, value in output.items():
            final_state.update(value)

    if final_state.get("verification_passed"):
        status = "verified_after_retry" if final_state.get("retries", 0) > 0 else "verified"
    else:
        status = "flagged_unverified"

    return {
        "id": item["id"],
        "question": question,
        "type": item["type"],
        "expected_behavior": item.get("expected_behavior", ""),
        "generated_answer": final_state.get("answer", ""),
        "sources": [{"spec_id": c.get("spec_id"), "clause_id": c.get("clause_id")} for c in chunks],
        "verification_passed": final_state.get("verification_passed", False),
        "retries": final_state.get("retries", 0),
        "termination_reason": final_state.get("termination_reason"),
        "status": status,
        "feedback": final_state.get("feedback", ""),
    }


def run(
    golden_set_path: Path = GOLDEN_SET_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    limit: Optional[int] = None,
    types: Optional[List[str]] = None,
    sleep_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    cfg = get_settings()
    golden_set = load_golden_set(golden_set_path)
    expected_hash = golden_set_hash(golden_set)

    questions = golden_set
    if types:
        questions = [q for q in questions if q["type"] in types]
    if limit is not None:
        questions = questions[:limit]

    existing_by_id = load_existing_results(results_path, expected_hash)
    results_by_id: Dict[int, Dict[str, Any]] = dict(existing_by_id)

    pending = [q for q in questions if q["id"] not in results_by_id]
    print(f"Golden set: {len(golden_set)} questions. Selected: {len(questions)}. "
          f"Already done: {len(questions) - len(pending)}. Pending: {len(pending)}.")

    if pending:
        app = build_graph()
        retriever = Retriever()
        sleep_time = sleep_seconds if sleep_seconds is not None else cfg.eval_sleep_seconds

        for idx, item in enumerate(pending, start=1):
            print(f"\n[{idx}/{len(pending)}] (id={item['id']}, {item['type']}) {item['question']}")
            result = run_one_question(app, retriever, item, cfg.default_top_k)
            results_by_id[item["id"]] = result

            atomic_write_json(results_path, {
                "golden_set_hash": expected_hash,
                "results": [results_by_id[k] for k in sorted(results_by_id)],
            })

            if idx < len(pending):
                time.sleep(sleep_time)

    selected_results = [results_by_id[q["id"]] for q in questions if q["id"] in results_by_id]
    print_summary(selected_results)
    return selected_results


def print_summary(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    passed = sum(1 for r in results if r["verification_passed"])
    retried = sum(1 for r in results if r["retries"] > 0)
    flagged = sum(1 for r in results if r["status"] == "flagged_unverified")

    out_of_scope = [r for r in results if r["type"] == "out_of_scope"]
    oos_refusals = sum(
        1 for r in out_of_scope
        if is_refusal(r["generated_answer"]) or len(r["sources"]) == 0 or "flagged" in r["status"]
    )

    adversarial = [r for r in results if r["type"] == "adversarial"]
    adv_caught = sum(1 for r in adversarial if r["retries"] > 0)

    print("\n" + "=" * 60)
    print("EVALUATION AGGREGATE METRICS")
    print("=" * 60)
    print(f"Total Questions: {total}")
    print(f"Verification Pass Rate: {passed / total * 100:.1f}% ({passed}/{total})")
    print(f"Retry Rate: {retried / total * 100:.1f}% ({retried}/{total})")
    print(f"Flagged Rate: {flagged / total * 100:.1f}% ({flagged}/{total})")
    if out_of_scope:
        print(f"Refusal Accuracy (Out-of-Scope): {oos_refusals / len(out_of_scope) * 100:.1f}% ({oos_refusals}/{len(out_of_scope)})")
    if adversarial:
        print(f"Conflation Catch Rate (Adversarial): {adv_caught / len(adversarial) * 100:.1f}% ({adv_caught}/{len(adversarial)})")


def main():
    parser = argparse.ArgumentParser(description="Run the golden set through the generate/verify graph.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected questions.")
    parser.add_argument("--types", type=str, default=None,
                         help="Comma-separated question types to include (normal,adversarial,out_of_scope).")
    parser.add_argument("--results-path", type=str, default=str(DEFAULT_RESULTS_PATH))
    args = parser.parse_args()

    types = args.types.split(",") if args.types else None
    run(results_path=Path(args.results_path), limit=args.limit, types=types)


if __name__ == "__main__":
    main()
