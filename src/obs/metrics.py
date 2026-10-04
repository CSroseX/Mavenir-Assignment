"""RunRecorder: writes self-contained eval run artifacts to disk.

Produces eval/runs/<run_id>/{run.json, per_question.jsonl, summary.json}.
No Langfuse dependency — this is the offline eval gate's persistence layer.
Cost figures are ESTIMATES derived from the price table in src/config.py.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_settings


class RunRecorder:
    """Records per-node latency, token counts, cost estimates, and node paths
    for an eval run. Call start_run / start_question / record_node /
    end_question / end_run in sequence; end_run writes the three artifact files.

    All cost figures are labelled ESTIMATE — they use a static price table
    from config, not the provider's actual billing.
    """

    def __init__(self, run_id: Optional[str] = None, output_dir: Optional[Path] = None):
        cfg = get_settings()
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.output_dir = (
            Path(output_dir) if output_dir
            else cfg.project_root / "eval" / "runs" / self.run_id
        )
        self._questions: List[Dict[str, Any]] = []
        self._current_question: Optional[Dict[str, Any]] = None
        self._run_meta: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start_run(self, model_name: str, **metadata: Any) -> None:
        self._start_time = time.monotonic()
        self._run_meta = {
            "run_id": self.run_id,
            "model_name": model_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }

    def start_question(
        self, question_id: int, question_text: str, question_type: str
    ) -> None:
        if self._current_question is not None:
            raise RuntimeError(
                f"start_question called while question {self._current_question['question_id']} "
                "is still open — call end_question first."
            )
        self._current_question = {
            "question_id": question_id,
            "question": question_text,
            "type": question_type,
            "nodes": [],
            "_start_mono": time.monotonic(),
        }

    def record_node(
        self,
        node_name: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        **extra: Any,
    ) -> None:
        if self._current_question is None:
            raise RuntimeError(
                "record_node called outside a question — call start_question first."
            )
        cfg = get_settings()
        cost_estimate = (
            tokens_in * cfg.cost_per_1k_input_tokens / 1000
            + tokens_out * cfg.cost_per_1k_output_tokens / 1000
        )
        node_record: Dict[str, Any] = {
            "node": node_name,
            "latency_ms": round(latency_ms, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_estimate_usd": round(cost_estimate, 8),
            **extra,
        }
        self._current_question["nodes"].append(node_record)

    def end_question(
        self,
        termination_reason: Optional[str] = None,
        **result_fields: Any,
    ) -> Dict[str, Any]:
        if self._current_question is None:
            raise RuntimeError(
                "end_question called with no open question — call start_question first."
            )
        q = self._current_question
        elapsed_s = round(time.monotonic() - q.pop("_start_mono"), 3)

        total_in = sum(n["tokens_in"] for n in q["nodes"])
        total_out = sum(n["tokens_out"] for n in q["nodes"])
        total_cost = sum(n["cost_estimate_usd"] for n in q["nodes"])

        record = {
            "question_id": q["question_id"],
            "question": q["question"],
            "type": q["type"],
            "elapsed_s": elapsed_s,
            "termination_reason": termination_reason,
            "node_path": [n["node"] for n in q["nodes"]],
            "nodes": q["nodes"],
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_cost_estimate_usd": round(total_cost, 8),
            **result_fields,
        }
        self._questions.append(record)
        self._current_question = None
        return record

    def end_run(self) -> Path:
        """Finalize the run and write all artifacts. Returns the output dir."""
        if self._start_time is None:
            raise RuntimeError("end_run called before start_run.")
        self._end_time = time.monotonic()
        self._write_artifacts()
        return self.output_dir

    def _write_artifacts(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        run_json = {
            **self._run_meta,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(self._end_time - self._start_time, 3),
            "question_count": len(self._questions),
        }
        _write_json(self.output_dir / "run.json", run_json)

        with open(self.output_dir / "per_question.jsonl", "w", encoding="utf-8") as f:
            for q in self._questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        summary = self._build_summary()
        _write_json(self.output_dir / "summary.json", summary)

    def _build_summary(self) -> Dict[str, Any]:
        total_in = sum(q["total_tokens_in"] for q in self._questions)
        total_out = sum(q["total_tokens_out"] for q in self._questions)
        total_cost = sum(q["total_cost_estimate_usd"] for q in self._questions)

        per_node_totals: Dict[str, Dict[str, Any]] = {}
        for q in self._questions:
            for n in q["nodes"]:
                name = n["node"]
                bucket = per_node_totals.setdefault(
                    name,
                    {"calls": 0, "tokens_in": 0, "tokens_out": 0,
                     "cost_estimate_usd": 0.0, "total_latency_ms": 0.0},
                )
                bucket["calls"] += 1
                bucket["tokens_in"] += n["tokens_in"]
                bucket["tokens_out"] += n["tokens_out"]
                bucket["cost_estimate_usd"] += n["cost_estimate_usd"]
                bucket["total_latency_ms"] += n["latency_ms"]

        for bucket in per_node_totals.values():
            bucket["cost_estimate_usd"] = round(bucket["cost_estimate_usd"], 8)
            bucket["total_latency_ms"] = round(bucket["total_latency_ms"], 2)

        termination_counts: Dict[str, int] = {}
        for q in self._questions:
            reason = q.get("termination_reason") or "unknown"
            termination_counts[reason] = termination_counts.get(reason, 0) + 1

        return {
            "run_id": self.run_id,
            "question_count": len(self._questions),
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_tokens": total_in + total_out,
            "total_cost_estimate_usd (ESTIMATE)": round(total_cost, 8),
            "per_node": per_node_totals,
            "termination_reasons": termination_counts,
            "_note": "All cost figures are ESTIMATES based on a static price table, not actual billing.",
        }


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
