"""Pure parts of src/eval/triage_sweep.py (no model needed)."""

from src.eval.triage_sweep import recommend, render_svg, simulate, sweep


def rows(in_scores, out_scores, tier1_in=False):
    out = [{"id": i, "question": f"in {i}", "gate": "in", "tier1_hit": tier1_in, "score": s}
           for i, s in enumerate(in_scores)]
    out += [{"id": 100 + i, "question": f"out {i}", "gate": "out", "tier1_hit": False, "score": s}
            for i, s in enumerate(out_scores)]
    out.append({"id": 999, "question": "borderline", "gate": "excluded", "tier1_hit": True, "score": 0.9})
    return out


class TestRecommend:
    def test_separable_classes_get_a_safety_band(self):
        rec = recommend(rows([0.80, 0.85], [0.55, 0.60]), margin=0.02)
        assert rec["separable"] is True
        assert rec["triage_refuse_below"] == 0.62  # just above the highest out-of-scope
        assert rec["triage_accept_above"] == 0.78  # just below the lowest in-domain

    def test_overlap_goes_to_the_llm_band(self):
        rec = recommend(rows([0.65, 0.85], [0.55, 0.70]), margin=0.0)
        assert rec["separable"] is False
        assert rec["triage_refuse_below"] == 0.65
        assert rec["triage_accept_above"] == 0.70

    def test_borderline_is_excluded_from_the_gate(self):
        # The borderline row scores 0.9; if it counted as out-of-scope it
        # would drag accept_above up to 0.92.
        assert recommend(rows([0.80], [0.55]), margin=0.02)["triage_accept_above"] == 0.78


class TestSweepAndSimulate:
    def test_counts(self):
        r = rows([0.80, 0.85], [0.55, 0.60])
        by_t = {c["threshold"]: c for c in sweep(r)}
        assert by_t[0.7]["out_of_scope_refused"] == 2
        assert by_t[0.7]["in_domain_refused_tier2_only"] == 0
        assert by_t[0.825]["in_domain_refused_tier2_only"] == 1

    def test_simulate_marks_band_for_llm(self):
        r = simulate(rows([0.70], [0.50]), refuse_below=0.6, accept_above=0.75)
        decisions = {x["id"]: x["decision"] for x in r}
        assert decisions[0] == "llm (tier 3)"
        assert decisions[100] == "refuse (tier 2)"
        assert decisions[999] == "retrieve (tier 1)"

    def test_svg_renders(self):
        r = rows([0.80, 0.85], [0.55, 0.60])
        rec = recommend(r, 0.02)
        svg = render_svg(simulate(r, rec["triage_refuse_below"], rec["triage_accept_above"]), sweep(r), rec)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert svg.count("<circle") == len(r)
