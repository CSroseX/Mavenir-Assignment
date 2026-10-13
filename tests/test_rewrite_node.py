"""rewrite_query_node: deterministic, corpus-lexicon-driven rewrites."""

from src.generation.lexicon import DomainLexicon
from src.generation.nodes.rewrite import rewrite_query, rewrite_query_node, strip_framing


def fake_lexicon():
    return DomainLexicon(
        long_forms={
            "AMF": "Access and Mobility Management Function",
            "MME": "Mobility Management Entity",
            "UDM": "Unified Data Management",
        },
        acronyms={"AMF", "MME", "UDM", "UE", "TS"},
        acronym_freq={},
        title_bigrams=set(),
        generation={"AMF": "5g", "UDM": "5g", "MME": "4g"},
    )


class TestStripFraming:
    def test_strips_interrogative(self):
        assert strip_framing("What is the role of the AMF?") == "the role of the AMF"

    def test_strips_stacked_framing(self):
        assert strip_framing("Hi, can you please explain how the UE registers?") == "the UE registers"

    def test_keeps_spec_and_clause_refs(self):
        assert strip_framing("Tell me about clause 5.5.1 of TS 24.501 please") == "clause 5.5.1 of TS 24.501"

    def test_never_empties_the_query(self):
        assert strip_framing("What?") == "What"


class TestRewriteQuery:
    def test_expands_acronym_to_long_form(self):
        out = rewrite_query("What is the AMF?", fake_lexicon())
        assert out == ["the AMF Access and Mobility Management Function"]

    def test_long_form_gets_its_acronym(self):
        out = rewrite_query("What does the Access and Mobility Management Function do?", fake_lexicon())
        assert len(out) == 1
        assert out[0].endswith("AMF")

    def test_spec_prefix_is_not_expanded(self):
        lex = fake_lexicon()
        lex.long_forms["TS"] = "Technical Specification"
        out = rewrite_query("What is the AMF according to TS 23.501?", lex)
        assert "Technical Specification" not in out[0]
        assert "TS 23.501" in out[0]

    def test_cross_generation_query_is_split(self):
        out = rewrite_query("When an MME receives an Attach Request, does it communicate with the UDM?", fake_lexicon())
        assert len(out) == 2
        part_4g, part_5g = out
        assert "MME" in part_4g and "UDM" not in part_4g
        assert "Mobility Management Entity" in part_4g
        assert part_4g.endswith("TS 24.301")  # pinned to the EPS spec
        assert "UDM" in part_5g and "MME" not in part_5g
        assert "Unified Data Management" in part_5g

    def test_split_does_not_pin_when_user_named_a_spec(self):
        out = rewrite_query("Does the MME talk to the UDM per TS 23.501?", fake_lexicon())
        assert len(out) == 2
        assert "24.301" not in out[0]

    def test_single_generation_is_not_split(self):
        assert len(rewrite_query("Does the AMF talk to the UDM?", fake_lexicon())) == 1

    def test_real_lexicon_splits_adversarial_question(self):
        # Golden-set Q7, against the corpus-derived lexicon.
        out = rewrite_query("When an MME receives an Attach Request, does it communicate with the UDM?")
        assert len(out) == 2


class TestRewriteQueryNode:
    def test_first_rewrite_sets_queries_and_counter(self, monkeypatch):
        import src.generation.nodes.rewrite as rw
        monkeypatch.setattr(rw, "get_lexicon", fake_lexicon)
        result = rewrite_query_node({"query": "What is the AMF?"})
        assert result["retrieval_queries"] == ["the AMF Access and Mobility Management Function"]
        assert result["rewrites"] == 1
        assert result["rewrite_noop"] is False

    def test_cap_from_config_makes_node_a_noop(self):
        from src.config import get_settings
        get_settings.cache_clear()
        cap = get_settings().max_rewrites
        result = rewrite_query_node({"query": "What is the AMF?", "rewrites": cap})
        assert result == {"rewrite_noop": True}

    def test_cap_is_read_from_config(self, monkeypatch):
        from src.config import get_settings
        monkeypatch.setenv("MAX_REWRITES", "2")
        get_settings.cache_clear()
        try:
            import src.generation.nodes.rewrite as rw
            monkeypatch.setattr(rw, "get_lexicon", fake_lexicon)
            result = rewrite_query_node({"query": "What is the AMF?", "rewrites": 1})
            assert result["rewrites"] == 2
        finally:
            monkeypatch.delenv("MAX_REWRITES")
            get_settings.cache_clear()

    def test_identical_rewrite_is_a_noop(self, monkeypatch):
        import src.generation.nodes.rewrite as rw
        monkeypatch.setattr(rw, "get_lexicon", fake_lexicon)
        # Nothing to strip or expand: the rewrite reproduces the original.
        result = rewrite_query_node({"query": "paging in idle mode"})
        assert result["rewrite_noop"] is True
