"""triage_node, refuse_node, after_triage and the centroid cache — with a
hashing bag-of-words fake embedder and fake LLMs (no models, no network)."""

import json
import re
import zlib

import httpx
import numpy as np
import openai
import pytest

from src.config import get_settings
from src.generation.centroids import (
    Centroids,
    build_centroids,
    get_centroids,
    load_centroids,
    save_centroids,
)
from src.generation.lexicon import get_lexicon
from src.generation.nodes.triage import (
    REFUSAL_ANSWER,
    lexical_signals,
    make_triage_node,
    parse_yes_no,
    refuse_node,
)
from src.generation.routers import after_triage

DIM = 64


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedder."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % DIM] += 1.0
    return out


def unit(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def fixed_centroids():
    return Centroids(
        vectors=np.vstack([unit(0), unit(1)]),
        spec_ids=np.array(["23.501", "24.301"]),
        clause_ids=np.array(["5.3", "5.5"]),
    )


def embed_at(cosine):
    """Embedder whose every query has exactly `cosine` similarity to unit(0)."""
    def embed(texts):
        v = np.zeros(DIM, dtype=np.float32)
        v[0] = cosine
        v[5] = np.sqrt(max(0.0, 1 - cosine ** 2))
        return np.vstack([v for _ in texts])
    return embed


class FakeLLM:
    def __init__(self, response="", exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.exc:
            raise self.exc
        return self.response


def node_with(score, llm):
    return make_triage_node(embed=embed_at(score), centroids_loader=lambda embed: fixed_centroids(), llm=llm)


GOLDEN = json.load(open(get_settings().project_root / "eval" / "golden" / "golden_set.v1.json", encoding="utf-8"))


class TestTier1Lexical:
    @pytest.mark.parametrize("item", [q for q in GOLDEN if q["type"] in ("normal", "adversarial")],
                             ids=lambda q: f"Q{q['id']}")
    def test_every_in_domain_golden_question_hits(self, item):
        assert any(lexical_signals(item["question"], get_lexicon()).values())

    @pytest.mark.parametrize("item", [q for q in GOLDEN if q["type"] == "out_of_scope" and not q["borderline"]],
                             ids=lambda q: f"Q{q['id']}")
    def test_no_hard_gate_out_of_scope_question_hits(self, item):
        assert not any(lexical_signals(item["question"], get_lexicon()).values())

    def test_spec_and_clause_refs(self):
        sig = lexical_signals("what does clause 5.5.1 of TS 24.501 say", get_lexicon())
        assert sig["spec_refs"] == ["24.501"]
        assert sig["clause_refs"] == ["5.5.1"]

    def test_bare_number_that_is_not_a_corpus_spec_is_ignored(self):
        assert lexical_signals("it cost 12.345 dollars", get_lexicon())["spec_refs"] == []

    def test_lexical_hit_skips_tiers_2_and_3(self):
        llm = FakeLLM("no")
        result = node_with(0.0, llm)({"query": "What is the role of the AMF?"})
        assert result["triage_decision"] == "in_domain"
        assert result["triage_tier"] == "lexical"
        assert llm.calls == []


class TestTier2Centroid:
    def test_capital_of_france_is_refused_without_llm(self):
        get_settings.cache_clear()
        cfg = get_settings()
        llm = FakeLLM("yes")
        result = node_with(cfg.triage_refuse_below - 0.1, llm)({"query": "What is the capital of France?"})
        assert result["triage_decision"] == "out_of_domain"
        assert result["triage_tier"] == "centroid"
        assert result["termination_reason"] == "out_of_domain"
        assert llm.calls == []

    def test_high_similarity_is_accepted_without_llm(self):
        cfg = get_settings()
        llm = FakeLLM("no")
        result = node_with(min(1.0, cfg.triage_accept_above + 0.05), llm)({"query": "how does a phone attach"})
        assert result["triage_decision"] == "in_domain"
        assert result["triage_tier"] == "centroid"
        assert result["termination_reason"] is None
        assert llm.calls == []


class TestTier3LLM:
    def band_score(self):
        cfg = get_settings()
        return (cfg.triage_refuse_below + cfg.triage_accept_above) / 2

    def test_ambiguous_band_asks_llm_with_tiny_budget(self):
        get_settings.cache_clear()
        llm = FakeLLM("No.")
        result = node_with(self.band_score(), llm)({"query": "how do networks hand over calls"})
        assert result["triage_decision"] == "out_of_domain"
        assert result["triage_tier"] == "llm"
        assert len(llm.calls) == 1
        assert llm.calls[0][1]["max_tokens"] == get_settings().triage_llm_max_tokens == 10

    def test_llm_yes_accepts(self):
        result = node_with(self.band_score(), FakeLLM("yes"))({"query": "how do networks hand over calls"})
        assert result["triage_decision"] == "in_domain"

    def test_unparseable_llm_answer_fails_open(self):
        result = node_with(self.band_score(), FakeLLM(""))({"query": "how do networks hand over calls"})
        assert result["triage_decision"] == "in_domain"
        assert result["triage_tier"] == "fail_open"

    def test_transport_error_on_shared_client_fails_open(self, monkeypatch):
        from src.generation import nodes as gen_nodes

        request = httpx.Request("POST", "http://x")

        def raise_conn(*args, **kwargs):
            raise openai.APIConnectionError(request=request)

        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", raise_conn)
        node = make_triage_node(embed=embed_at(self.band_score()), centroids_loader=lambda e: fixed_centroids())
        result = node({"query": "how do networks hand over calls"})
        assert result["triage_decision"] == "in_domain"
        assert result["triage_tier"] == "fail_open"

    def test_no_embedder_goes_straight_to_llm(self):
        llm = FakeLLM("no")
        result = make_triage_node(embed=None, llm=llm)({"query": "What is the capital of France?"})
        assert result["triage_decision"] == "out_of_domain"
        assert result["triage_tier"] == "llm"
        assert result["triage_score"] is None

    def test_centroid_loader_failure_degrades_to_llm(self):
        def broken(embed):
            raise OSError("no cache")

        llm = FakeLLM("no")
        node = make_triage_node(embed=fake_embed, centroids_loader=broken, llm=llm)
        assert node({"query": "What is the capital of France?"})["triage_tier"] == "llm"


class TestParseYesNo:
    def test_variants(self):
        assert parse_yes_no("Yes") is True
        assert parse_yes_no(" no.") is False
        assert parse_yes_no("maybe") is None
        assert parse_yes_no("") is None


class TestRefuseAndRouter:
    def test_refuse_node_is_a_terminal_refusal(self):
        result = refuse_node({"query": "What is the capital of France?"})
        assert result["answer"] == REFUSAL_ANSWER
        assert "out of scope" in result["answer"]
        assert result["chunks"] == []
        assert result["termination_reason"] == "out_of_domain"
        assert result["abstention"]["termination_reason"] == "out_of_domain"

    def test_after_triage(self):
        assert after_triage({"triage_decision": "out_of_domain"}) == "refuse"
        assert after_triage({"triage_decision": "in_domain"}) == "retrieve"
        assert after_triage({}) == "retrieve"


def synthetic_chunks():
    return [
        {"spec_id": "23.501", "version": "17.11.0", "clause_id": "0", "clause_title": "Frontmatter",
         "chunk_type": "text", "content": "copyright"},
        {"spec_id": "23.501", "version": "17.11.0", "clause_id": "2", "clause_title": "References",
         "chunk_type": "text", "content": "IETF RFC 793 TCP"},
        {"spec_id": "23.501", "version": "17.11.0", "clause_id": "5.3.1", "clause_title": "Registration",
         "chunk_type": "text", "content": "registration management amf ue"},
        {"spec_id": "23.501", "version": "17.11.0", "clause_id": "5.3.2", "clause_title": "Registration",
         "chunk_type": "text", "content": "registration area tracking"},
        {"spec_id": "24.301", "version": "17.9.0", "clause_id": "5.5.1", "clause_title": "Attach",
         "chunk_type": "table", "content": [["IE", "Presence"], ["EPS attach type", "M"]]},
    ]


class TestCentroids:
    def test_groups_by_clause_prefix_and_excludes_frontmatter_and_references(self):
        c = build_centroids(synthetic_chunks(), fake_embed)
        assert sorted(zip(c.spec_ids.tolist(), c.clause_ids.tolist())) == [("23.501", "5.3"), ("24.301", "5.5")]
        assert np.allclose(np.linalg.norm(c.vectors, axis=1), 1.0)

    def test_in_domain_query_scores_higher_than_out_of_domain(self):
        c = build_centroids(synthetic_chunks(), fake_embed)
        in_score, spec_id, clause = c.max_similarity(fake_embed(["amf registration management"])[0])
        out_score, _, _ = c.max_similarity(fake_embed(["capital of france"])[0])
        assert (spec_id, clause) == ("23.501", "5.3")
        assert in_score > out_score

    def test_save_load_roundtrip(self, tmp_path):
        c = build_centroids(synthetic_chunks(), fake_embed)
        c.corpus_fingerprint = "abc"
        save_centroids(c, tmp_path / "c.npz")
        loaded = load_centroids(tmp_path / "c.npz")
        assert np.allclose(loaded.vectors, c.vectors)
        assert loaded.clause_ids.tolist() == c.clause_ids.tolist()
        assert loaded.corpus_fingerprint == "abc"

    def test_stale_cache_without_embedder_is_unavailable(self, tmp_path):
        c = build_centroids(synthetic_chunks(), fake_embed)
        c.embedding_model = "some-other-model"
        save_centroids(c, tmp_path / "c.npz")
        assert get_centroids(embed=None, path=tmp_path / "c.npz") is None

    def test_missing_cache_is_built_and_reused(self, tmp_path, monkeypatch):
        import src.generation.lexicon as lexicon_mod

        monkeypatch.setattr(lexicon_mod, "load_corpus_chunks", lambda: synthetic_chunks())
        path = tmp_path / "c.npz"
        built = get_centroids(embed=fake_embed, path=path)
        assert path.exists() and len(built.vectors) == 2

        def must_not_embed(texts):
            raise AssertionError("valid cache should be reused, not rebuilt")

        reused = get_centroids(embed=must_not_embed, path=path)
        assert np.allclose(reused.vectors, built.vectors)
