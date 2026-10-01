import json
import os


GOLDEN_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "eval", "golden", "golden_set.v1.json"
)

REQUIRED_FIELDS = {
    "id", "question", "type", "expected_behavior",
    "gold_clauses", "retrieval_scored", "corpus_answerable",
    "borderline", "notes",
}

VALID_TYPES = {"normal", "adversarial", "out_of_scope"}


def _load_golden_set():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestGoldenSetStructure:
    def test_file_exists(self):
        assert os.path.exists(GOLDEN_SET_PATH)

    def test_has_18_records(self):
        gs = _load_golden_set()
        assert len(gs) == 18

    def test_stable_ids(self):
        gs = _load_golden_set()
        ids = [r["id"] for r in gs]
        assert ids == list(range(1, 19))

    def test_required_fields_present(self):
        gs = _load_golden_set()
        for r in gs:
            missing = REQUIRED_FIELDS - set(r.keys())
            assert not missing, f"Q{r.get('id', '?')} missing fields: {missing}"

    def test_valid_types(self):
        gs = _load_golden_set()
        for r in gs:
            assert r["type"] in VALID_TYPES, f"Q{r['id']} has invalid type: {r['type']}"

    def test_type_counts(self):
        gs = _load_golden_set()
        counts = {}
        for r in gs:
            counts[r["type"]] = counts.get(r["type"], 0) + 1
        assert counts["normal"] == 6
        assert counts["adversarial"] == 6
        assert counts["out_of_scope"] == 6


class TestGoldenSetContent:
    def test_q15_is_borderline(self):
        gs = _load_golden_set()
        q15 = [r for r in gs if r["id"] == 15][0]
        assert q15["borderline"] is True
        assert "TCP" in q15["question"]

    def test_non_borderline_default(self):
        gs = _load_golden_set()
        non_borderline = [r for r in gs if r["id"] != 15]
        for r in non_borderline:
            assert r["borderline"] is False, f"Q{r['id']} unexpectedly borderline"

    def test_normal_questions_have_gold_clauses(self):
        gs = _load_golden_set()
        for r in gs:
            if r["type"] == "normal":
                assert len(r["gold_clauses"]) > 0, f"Q{r['id']} (normal) has no gold_clauses"

    def test_normal_questions_retrieval_scored(self):
        gs = _load_golden_set()
        for r in gs:
            if r["type"] == "normal":
                assert r["retrieval_scored"] is True, f"Q{r['id']} (normal) not retrieval_scored"

    def test_gold_clause_structure(self):
        gs = _load_golden_set()
        for r in gs:
            for gc in r["gold_clauses"]:
                assert "spec_id" in gc, f"Q{r['id']} gold_clause missing spec_id"
                assert "clause_id" in gc, f"Q{r['id']} gold_clause missing clause_id"

    def test_out_of_scope_no_gold_clauses(self):
        gs = _load_golden_set()
        for r in gs:
            if r["type"] == "out_of_scope":
                assert len(r["gold_clauses"]) == 0, f"Q{r['id']} (out_of_scope) has gold_clauses"

    def test_adversarial_not_retrieval_scored(self):
        gs = _load_golden_set()
        for r in gs:
            if r["type"] == "adversarial":
                assert r["retrieval_scored"] is False, f"Q{r['id']} (adversarial) is retrieval_scored"

    def test_questions_match_eval_set(self):
        eval_set_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "eval", "eval_set.json"
        )
        with open(eval_set_path, "r", encoding="utf-8") as f:
            eval_set = json.load(f)

        gs = _load_golden_set()
        eval_questions = [e["question"] for e in eval_set]
        golden_questions = [r["question"] for r in gs]
        assert eval_questions == golden_questions

    def test_baseline_frozen(self):
        """eval/baseline/ must not be modified — the golden set is separate."""
        baseline_gold_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "eval", "baseline", "gold.json"
        )
        assert os.path.exists(baseline_gold_path), "eval/baseline/gold.json must still exist"
