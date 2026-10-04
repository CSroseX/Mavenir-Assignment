"""Tests for RunRecorder: artifact writing, cost estimation, state machine."""

import json

import pytest

from src.obs.metrics import RunRecorder


class TestRunRecorderLifecycle:
    """Happy-path lifecycle: start_run → questions → end_run → check files."""

    def _run_simple(self, tmp_path):
        rec = RunRecorder(run_id="test-run-001", output_dir=tmp_path / "test-run-001")
        rec.start_run(model_name="gpt-3.5-turbo", extra_field="value")

        rec.start_question(1, "What is the AMF?", "normal")
        rec.record_node("generate", latency_ms=120.5, tokens_in=500, tokens_out=200)
        rec.record_node("verify", latency_ms=80.3, tokens_in=600, tokens_out=100)
        rec.end_question(termination_reason="verified")

        rec.start_question(2, "What is the capital of France?", "out_of_scope")
        rec.record_node("generate", latency_ms=50.0, tokens_in=300, tokens_out=150)
        rec.record_node("verify", latency_ms=40.0, tokens_in=400, tokens_out=80)
        rec.record_node("abstain", latency_ms=1.0, tokens_in=0, tokens_out=0)
        rec.end_question(termination_reason="verification_failed")

        out = rec.end_run()
        return out

    def test_creates_three_artifact_files(self, tmp_path):
        out = self._run_simple(tmp_path)
        assert (out / "run.json").exists()
        assert (out / "per_question.jsonl").exists()
        assert (out / "summary.json").exists()

    def test_run_json_metadata(self, tmp_path):
        out = self._run_simple(tmp_path)
        run = json.loads((out / "run.json").read_text())
        assert run["run_id"] == "test-run-001"
        assert run["model_name"] == "gpt-3.5-turbo"
        assert run["extra_field"] == "value"
        assert run["question_count"] == 2
        assert "started_at" in run
        assert "ended_at" in run
        assert run["elapsed_s"] >= 0

    def test_per_question_jsonl_line_count(self, tmp_path):
        out = self._run_simple(tmp_path)
        lines = (out / "per_question.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_per_question_content(self, tmp_path):
        out = self._run_simple(tmp_path)
        lines = (out / "per_question.jsonl").read_text().strip().split("\n")
        q1 = json.loads(lines[0])
        assert q1["question_id"] == 1
        assert q1["question"] == "What is the AMF?"
        assert q1["type"] == "normal"
        assert q1["termination_reason"] == "verified"
        assert q1["node_path"] == ["generate", "verify"]
        assert len(q1["nodes"]) == 2
        assert q1["total_tokens_in"] == 1100  # 500 + 600
        assert q1["total_tokens_out"] == 300  # 200 + 100

    def test_per_question_node_path_with_abstain(self, tmp_path):
        out = self._run_simple(tmp_path)
        lines = (out / "per_question.jsonl").read_text().strip().split("\n")
        q2 = json.loads(lines[1])
        assert q2["node_path"] == ["generate", "verify", "abstain"]

    def test_summary_totals(self, tmp_path):
        out = self._run_simple(tmp_path)
        summary = json.loads((out / "summary.json").read_text())
        assert summary["run_id"] == "test-run-001"
        assert summary["question_count"] == 2
        # Q1: in=1100, out=300; Q2: in=700, out=230
        assert summary["total_tokens_in"] == 1800
        assert summary["total_tokens_out"] == 530
        assert summary["total_tokens"] == 2330

    def test_summary_per_node_breakdown(self, tmp_path):
        out = self._run_simple(tmp_path)
        summary = json.loads((out / "summary.json").read_text())
        per_node = summary["per_node"]
        assert per_node["generate"]["calls"] == 2
        assert per_node["verify"]["calls"] == 2
        assert per_node["abstain"]["calls"] == 1
        assert per_node["generate"]["tokens_in"] == 800  # 500 + 300
        assert per_node["generate"]["tokens_out"] == 350  # 200 + 150

    def test_summary_termination_reasons(self, tmp_path):
        out = self._run_simple(tmp_path)
        summary = json.loads((out / "summary.json").read_text())
        reasons = summary["termination_reasons"]
        assert reasons["verified"] == 1
        assert reasons["verification_failed"] == 1

    def test_end_run_returns_output_dir(self, tmp_path):
        out = self._run_simple(tmp_path)
        assert out == tmp_path / "test-run-001"


class TestCostEstimation:
    """Cost estimates use config's price table and are clearly labelled."""

    def test_cost_labelled_estimate_in_summary(self, tmp_path):
        rec = RunRecorder(run_id="cost-test", output_dir=tmp_path / "cost-test")
        rec.start_run(model_name="test")
        rec.start_question(1, "q", "normal")
        rec.record_node("generate", latency_ms=10, tokens_in=1000, tokens_out=500)
        rec.end_question(termination_reason="verified")
        rec.end_run()
        summary = json.loads((tmp_path / "cost-test" / "summary.json").read_text())
        assert "total_cost_estimate_usd (ESTIMATE)" in summary
        assert "ESTIMATE" in summary["_note"]

    def test_cost_calculation_accuracy(self, tmp_path):
        rec = RunRecorder(run_id="cost-calc", output_dir=tmp_path / "cost-calc")
        rec.start_run(model_name="test")
        rec.start_question(1, "q", "normal")
        # Default: 0.0015/1k input, 0.002/1k output
        # 1000 in * 0.0015/1000 = 0.0015
        # 500 out * 0.002/1000 = 0.001
        rec.record_node("generate", latency_ms=10, tokens_in=1000, tokens_out=500)
        rec.end_question(termination_reason="verified")
        rec.end_run()

        lines = (tmp_path / "cost-calc" / "per_question.jsonl").read_text().strip().split("\n")
        q = json.loads(lines[0])
        node = q["nodes"][0]
        assert node["cost_estimate_usd"] == pytest.approx(0.0025, abs=1e-6)

    def test_cost_in_node_records(self, tmp_path):
        rec = RunRecorder(run_id="node-cost", output_dir=tmp_path / "node-cost")
        rec.start_run(model_name="test")
        rec.start_question(1, "q", "normal")
        rec.record_node("generate", latency_ms=10, tokens_in=2000, tokens_out=1000)
        rec.end_question(termination_reason="verified")
        rec.end_run()

        lines = (tmp_path / "node-cost" / "per_question.jsonl").read_text().strip().split("\n")
        q = json.loads(lines[0])
        assert q["total_cost_estimate_usd"] == pytest.approx(0.005, abs=1e-6)


class TestStateMachineGuards:
    """RunRecorder rejects out-of-order calls."""

    def test_end_run_before_start_raises(self, tmp_path):
        rec = RunRecorder(run_id="guard-1", output_dir=tmp_path / "guard-1")
        with pytest.raises(RuntimeError, match="end_run called before start_run"):
            rec.end_run()

    def test_nested_start_question_raises(self, tmp_path):
        rec = RunRecorder(run_id="guard-2", output_dir=tmp_path / "guard-2")
        rec.start_run(model_name="test")
        rec.start_question(1, "q1", "normal")
        with pytest.raises(RuntimeError, match="start_question called while question 1"):
            rec.start_question(2, "q2", "normal")

    def test_record_node_without_question_raises(self, tmp_path):
        rec = RunRecorder(run_id="guard-3", output_dir=tmp_path / "guard-3")
        rec.start_run(model_name="test")
        with pytest.raises(RuntimeError, match="record_node called outside a question"):
            rec.record_node("generate", latency_ms=10)

    def test_end_question_without_start_raises(self, tmp_path):
        rec = RunRecorder(run_id="guard-4", output_dir=tmp_path / "guard-4")
        rec.start_run(model_name="test")
        with pytest.raises(RuntimeError, match="end_question called with no open question"):
            rec.end_question(termination_reason="verified")


class TestEdgeCases:
    """Zero-question runs, extra fields in record_node, default run_id."""

    def test_zero_question_run(self, tmp_path):
        rec = RunRecorder(run_id="empty", output_dir=tmp_path / "empty")
        rec.start_run(model_name="test")
        out = rec.end_run()
        summary = json.loads((out / "summary.json").read_text())
        assert summary["question_count"] == 0
        assert summary["total_tokens"] == 0
        lines = (out / "per_question.jsonl").read_text()
        assert lines == ""

    def test_extra_fields_in_record_node(self, tmp_path):
        rec = RunRecorder(run_id="extras", output_dir=tmp_path / "extras")
        rec.start_run(model_name="test")
        rec.start_question(1, "q", "normal")
        rec.record_node("verify", latency_ms=5, tokens_in=100, tokens_out=50,
                         is_supported=True, reasoning="all good")
        rec.end_question(termination_reason="verified")
        rec.end_run()
        lines = (tmp_path / "extras" / "per_question.jsonl").read_text().strip().split("\n")
        q = json.loads(lines[0])
        assert q["nodes"][0]["is_supported"] is True
        assert q["nodes"][0]["reasoning"] == "all good"

    def test_extra_fields_in_end_question(self, tmp_path):
        rec = RunRecorder(run_id="q-extras", output_dir=tmp_path / "q-extras")
        rec.start_run(model_name="test")
        rec.start_question(1, "q", "normal")
        rec.record_node("generate", latency_ms=5, tokens_in=100, tokens_out=50)
        rec.end_question(termination_reason="verified", answer="The AMF is...")
        rec.end_run()
        lines = (tmp_path / "q-extras" / "per_question.jsonl").read_text().strip().split("\n")
        q = json.loads(lines[0])
        assert q["answer"] == "The AMF is..."

    def test_default_run_id_format(self, tmp_path):
        rec = RunRecorder(output_dir=tmp_path / "default-id")
        assert len(rec.run_id) > 0
        # Should be a UTC timestamp like 20261003T120000Z
        assert "T" in rec.run_id

    def test_node_zero_tokens(self, tmp_path):
        rec = RunRecorder(run_id="zero-tok", output_dir=tmp_path / "zero-tok")
        rec.start_run(model_name="test")
        rec.start_question(1, "q", "normal")
        rec.record_node("abstain", latency_ms=0.5)  # no tokens
        rec.end_question(termination_reason="verification_failed")
        rec.end_run()
        lines = (tmp_path / "zero-tok" / "per_question.jsonl").read_text().strip().split("\n")
        q = json.loads(lines[0])
        assert q["nodes"][0]["tokens_in"] == 0
        assert q["nodes"][0]["tokens_out"] == 0
        assert q["nodes"][0]["cost_estimate_usd"] == 0.0
