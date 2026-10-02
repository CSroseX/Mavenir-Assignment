import json

import pytest

from src.eval.runner import (
    GoldenSetDriftError,
    ResultsFileCorruptError,
    atomic_write_json,
    golden_set_hash,
    load_existing_results,
)


def make_golden_set():
    return [
        {"id": 1, "question": "Q1?", "type": "normal"},
        {"id": 2, "question": "Q2?", "type": "adversarial"},
        {"id": 3, "question": "Q3?", "type": "out_of_scope"},
    ]


class TestGoldenSetHash:
    def test_same_content_same_hash(self):
        gs = make_golden_set()
        assert golden_set_hash(gs) == golden_set_hash(list(gs))

    def test_different_content_different_hash(self):
        gs1 = make_golden_set()
        gs2 = make_golden_set()
        gs2[0]["question"] = "Changed?"
        assert golden_set_hash(gs1) != golden_set_hash(gs2)

    def test_reordering_does_not_change_hash_content_but_id_keying_handles_it(self):
        # sort_keys in json.dumps sorts dict keys, not list order, so a
        # reordered list IS a different hash (list order is preserved).
        gs1 = make_golden_set()
        gs2 = list(reversed(make_golden_set()))
        assert golden_set_hash(gs1) != golden_set_hash(gs2)


class TestLoadExistingResultsNoFile:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        path = tmp_path / "results.json"
        result = load_existing_results(path, expected_hash="anything")
        assert result == {}


class TestLoadExistingResultsCorruptFile:
    def test_malformed_json_raises_instead_of_silently_discarding(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ResultsFileCorruptError):
            load_existing_results(path, expected_hash="anything")

    def test_corrupt_file_is_left_on_disk_after_raising(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ResultsFileCorruptError):
            load_existing_results(path, expected_hash="anything")
        # The old bug silently wiped the file on the next write; this
        # function must not touch the file at all when it can't parse it.
        assert path.read_text(encoding="utf-8") == "{not valid json"


class TestLoadExistingResultsDriftDetection:
    def test_matching_hash_loads_results_keyed_by_id(self, tmp_path):
        gs = make_golden_set()
        h = golden_set_hash(gs)
        path = tmp_path / "results.json"
        path.write_text(json.dumps({
            "golden_set_hash": h,
            "results": [{"id": 1, "question": "Q1?", "answer": "x"}],
        }), encoding="utf-8")

        loaded = load_existing_results(path, expected_hash=h)
        assert loaded == {1: {"id": 1, "question": "Q1?", "answer": "x"}}

    def test_mismatched_hash_raises_drift_error(self, tmp_path):
        gs = make_golden_set()
        h = golden_set_hash(gs)
        path = tmp_path / "results.json"
        path.write_text(json.dumps({
            "golden_set_hash": "some-stale-hash",
            "results": [{"id": 1, "question": "Q1?", "answer": "x"}],
        }), encoding="utf-8")

        with pytest.raises(GoldenSetDriftError):
            load_existing_results(path, expected_hash=h)

    def test_missing_hash_field_is_tolerated_for_backward_compat(self, tmp_path):
        """A results file with no golden_set_hash (e.g. hand-written, or
        from before this field existed) should load, not drift-error."""
        path = tmp_path / "results.json"
        path.write_text(json.dumps({
            "results": [{"id": 1, "question": "Q1?", "answer": "x"}],
        }), encoding="utf-8")
        loaded = load_existing_results(path, expected_hash="whatever")
        assert loaded == {1: {"id": 1, "question": "Q1?", "answer": "x"}}


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"a": 1})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    def test_no_leftover_temp_file(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"a": 1})
        assert not (tmp_path / "out.json.tmp").exists()

    def test_overwrites_existing_file_completely(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"a": 1, "b": 2})
        atomic_write_json(path, {"c": 3})
        assert json.loads(path.read_text(encoding="utf-8")) == {"c": 3}
