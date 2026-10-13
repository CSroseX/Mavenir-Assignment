"""grade_sufficiency_node and after_grade, with a fake LLM (no network)."""

import json

import httpx
import openai

from src.config import get_settings
from src.generation import nodes as gen_nodes
from src.generation.nodes.grade import make_grade_sufficiency_node, parse_relevance
from src.generation.routers import after_grade


def chunks_with_scores(*scores, key="score"):
    return [
        {"spec_id": "23.501", "clause_id": f"6.2.{i}", key: s, "content": f"passage {i}"}
        for i, s in enumerate(scores, start=1)
    ]


class FakeLLM:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.exc:
            raise self.exc
        return self.response


def flags_json(*flags):
    return json.dumps({"relevant": list(flags)})


class TestGradeSignals:
    def test_no_chunks_is_insufficient_without_llm(self):
        llm = FakeLLM(flags_json())
        result = make_grade_sufficiency_node(llm)({"query": "q", "chunks": []})
        assert result["n_relevant_chunks"] == 0
        assert result["sufficiency"] == "insufficient"
        assert result["termination_reason"] == "insufficient_context"
        assert llm.calls == []

    def test_top1_below_floor_skips_llm(self):
        get_settings.cache_clear()
        floor = get_settings().sufficiency_min_top1_score
        llm = FakeLLM(flags_json(True, True))
        result = make_grade_sufficiency_node(llm)(
            {"query": "What is the capital of France?", "chunks": chunks_with_scores(floor - 1, floor - 5)}
        )
        assert llm.calls == []
        assert result["n_relevant_chunks"] == 0
        assert result["sufficiency_signal"] == "cross_encoder_floor"

    def test_prefers_cross_score_over_boosted_score(self):
        # score includes a lineage boost; the floor must key on the raw logit.
        chunk = {"spec_id": "23.501", "clause_id": "6.2.1", "score": 9.0, "cross_score": -9.0, "content": "x"}
        llm = FakeLLM(flags_json(True))
        result = make_grade_sufficiency_node(llm)({"query": "q", "chunks": [chunk]})
        assert llm.calls == []
        assert result["top1_cross_score"] == -9.0

    def test_two_relevant_is_sufficient_and_filters_chunks(self):
        llm = FakeLLM(flags_json(True, False, True, False, False))
        chunks = chunks_with_scores(8.0, 7.0, 6.0, 5.0, 4.0)
        result = make_grade_sufficiency_node(llm)({"query": "q", "chunks": chunks})
        assert result["n_relevant_chunks"] == 2
        assert result["sufficiency"] == "sufficient"
        assert result["termination_reason"] is None
        assert [c["clause_id"] for c in result["chunks"]] == ["6.2.1", "6.2.3"]
        assert llm.calls[0][1]["max_tokens"] == get_settings().sufficiency_llm_max_tokens

    def test_one_relevant_is_insufficient_and_keeps_chunks(self):
        llm = FakeLLM(flags_json(True, False, False))
        result = make_grade_sufficiency_node(llm)({"query": "q", "chunks": chunks_with_scores(8.0, 7.0, 6.0)})
        assert result["n_relevant_chunks"] == 1
        assert result["termination_reason"] == "insufficient_context"
        assert "chunks" not in result

    def test_only_top_n_are_judged(self):
        get_settings.cache_clear()
        top_n = get_settings().sufficiency_top_n
        llm = FakeLLM(flags_json(*([True] * top_n)))
        chunks = chunks_with_scores(*[8.0] * (top_n + 3))
        result = make_grade_sufficiency_node(llm)({"query": "q", "chunks": chunks})
        assert len(result["relevance_flags"]) == top_n
        prompt = llm.calls[0][0][1]["content"]
        assert f"[{top_n}]" in prompt and f"[{top_n + 1}]" not in prompt

    def test_judge_prompt_caps_chunk_text(self):
        get_settings.cache_clear()
        cap = get_settings().sufficiency_chunk_char_cap
        chunk = {"spec_id": "23.501", "clause_id": "1", "score": 5.0, "content": "x" * (cap * 3)}
        llm = FakeLLM(flags_json(True))
        make_grade_sufficiency_node(llm)({"query": "q", "chunks": [chunk]})
        assert "x" * (cap + 1) not in llm.calls[0][0][1]["content"]


class TestGradeFallback:
    def test_malformed_json_falls_back_to_cross_encoder_count(self):
        get_settings.cache_clear()
        thr = get_settings().sufficiency_fallback_min_score
        llm = FakeLLM("not json")
        result = make_grade_sufficiency_node(llm)(
            {"query": "q", "chunks": chunks_with_scores(thr + 2, thr + 1, thr - 1)}
        )
        assert result["sufficiency_signal"] == "cross_encoder_fallback"
        assert result["n_relevant_chunks"] == 2

    def test_wrong_length_falls_back(self):
        llm = FakeLLM(flags_json(True))
        result = make_grade_sufficiency_node(llm)({"query": "q", "chunks": chunks_with_scores(8.0, 7.0)})
        assert result["sufficiency_signal"] == "cross_encoder_fallback"

    def test_transport_error_on_shared_client_falls_back(self, monkeypatch):
        request = httpx.Request("POST", "http://x")

        def raise_conn(*args, **kwargs):
            raise openai.APIConnectionError(request=request)

        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", raise_conn)
        node = make_grade_sufficiency_node()  # default seam: shared client
        result = node({"query": "q", "chunks": chunks_with_scores(8.0, 7.0)})
        assert result["sufficiency_signal"] == "cross_encoder_fallback"
        assert result["n_relevant_chunks"] == 2


class TestParseRelevance:
    def test_accepts_exact_booleans(self):
        assert parse_relevance('{"relevant": [true, false]}', 2) == [True, False]

    def test_rejects_non_booleans(self):
        import pytest
        with pytest.raises(ValueError):
            parse_relevance('{"relevant": ["yes", "no"]}', 2)


class TestAfterGrade:
    def test_sufficient_routes_to_generate(self):
        get_settings.cache_clear()
        cfg = get_settings()
        assert after_grade({"n_relevant_chunks": cfg.sufficiency_min_relevant, "rewrites": 0}) == "generate"

    def test_insufficient_with_rewrites_left_routes_to_rewrite(self):
        assert after_grade({"n_relevant_chunks": 1, "rewrites": 0}) == "rewrite_query"

    def test_insufficient_exhausted_routes_to_abstain(self):
        get_settings.cache_clear()
        cfg = get_settings()
        assert after_grade({"n_relevant_chunks": 0, "rewrites": cfg.max_rewrites}) == "abstain"

    def test_poisoned_retrieval_sequence(self):
        """Poisoned retrieval: zero relevant chunks on both passes routes
        rewrite -> abstain, never to generate (no 3 pointless regenerations)."""
        assert after_grade({"n_relevant_chunks": 0}) == "rewrite_query"
        assert after_grade({"n_relevant_chunks": 0, "rewrites": 1}) == "abstain"

    def test_cross_product(self):
        get_settings.cache_clear()
        cfg = get_settings()
        for n in (None, 0, 1, cfg.sufficiency_min_relevant, 5):
            for rewrites in (None, 0, cfg.max_rewrites - 1, cfg.max_rewrites, cfg.max_rewrites + 1):
                state = {"n_relevant_chunks": n, "rewrites": rewrites}
                got = after_grade(state)
                if (n or 0) >= cfg.sufficiency_min_relevant:
                    assert got == "generate", state
                elif (rewrites or 0) < cfg.max_rewrites:
                    assert got == "rewrite_query", state
                else:
                    assert got == "abstain", state


class TestPromptChunkCap:
    def test_generate_and_verify_prompts_cap_chunk_text(self):
        from src.generation.nodes.core import _build_generate_messages, _build_verify_prompt

        get_settings.cache_clear()
        cap = get_settings().prompt_chunk_char_cap
        chunks = [{"spec_id": "23.501", "clause_id": "1", "content": "y" * (cap * 2)}]
        gen = _build_generate_messages("q", chunks, "")[1]["content"]
        ver = _build_verify_prompt("a", chunks)
        assert "y" * cap in gen and "y" * (cap + 1) not in gen
        assert "y" * cap in ver and "y" * (cap + 1) not in ver
