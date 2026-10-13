"""Central configuration for the 3GPP RAG system, powered by pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing 'src/'."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "src").is_dir() and (current / "data").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Infrastructure ──────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "3gpp_specs"

    ollama_base_url: str = "http://localhost:11434/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_base_url: str = "https://api.openai.com/v1"

    # ── Models ──────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    local_llm_model: str = "qwen3.5:4b"
    remote_llm_model_openrouter: str = "openai/gpt-oss-20b"
    remote_llm_model_openai: str = "gpt-3.5-turbo"

    # ── Vector DB ───────────────────────────────────────────────────
    vector_size: int = 384

    # ── Retrieval ───────────────────────────────────────────────────
    default_top_k: int = 5
    # Candidates sent to the cross-encoder = rerank_multiplier * top_k.
    # (Was a flat fetch_k=20 behind max(fetch_k, top_k), so top_k=5 and
    # top_k=20 both reranked the same 20 candidates.)
    rerank_multiplier: int = 8
    # BM25 (in-process, src/retrieval/hybrid.py) fused with dense via RRF.
    hybrid_bm25: bool = True
    rrf_k: int = 60
    # Added to the cross-encoder logit × lineage_match (src/retrieval/lineage.py).
    # Off until swept against the reranker: it also fully matches decoys,
    # e.g. 24.301 "GUTI" for a 5G-GUTI question.
    lineage_boost: float = 0.0
    verify_top_k: int = 3

    # ── Generation ──────────────────────────────────────────────────
    generate_temperature: float = 0.3
    verify_temperature: float = 0.0
    verify_max_tokens: int = 2000
    max_retries: int = 3

    # ── Chunking ────────────────────────────────────────────────────
    chunk_soft_limit: int = 400
    chunk_hard_cap: int = 800

    # ── Indexing ────────────────────────────────────────────────────
    indexer_batch_size: int = 100

    # ── Eval ────────────────────────────────────────────────────────
    eval_sleep_seconds: int = 5
    # Ranked-list length the retrieval report scores (recall@20 needs 20).
    eval_top_k: int = 20

    # ── Cost estimation (USD per 1k tokens, clearly an ESTIMATE) ───
    cost_per_1k_input_tokens: float = 0.0015
    cost_per_1k_output_tokens: float = 0.002

    # ── LLM selection (from .env) ───────────────────────────────────
    use_remote_llm: bool = False
    open_router_api_key: str = ""
    openai_api_key: str = ""
    model_name: str = ""

    # ── Phase 4 decision-node settings ──────────────────────────────
    # Corpus lexicon (src/generation/lexicon.py): an acronym-shaped token
    # must occur at least this often in spec text to count as a domain term
    # (keeps one-off mentions like XML/HTTP out of the lexicon).
    lexicon_min_term_freq: int = 10

    # rewrite_query_node: hard cap on rewrites per question; after it the
    # flow terminates at abstain. Generation affinity: an acronym is 4G- or
    # 5G-affine when its per-token rate in one spec group is >= this ratio
    # times its rate in the other.
    max_rewrites: int = 1
    rewrite_generation_affinity_ratio: float = 3.0
    rewrite_4g_spec_ids: list[str] = ["24.301"]
    rewrite_5g_spec_ids: list[str] = ["23.501", "24.501"]

    # grade_sufficiency_node. Cross-encoder (ms-marco-MiniLM-L-6-v2) logits:
    # the six answerable golden-set questions measured top-1 4.6-8.4 in
    # eval/baseline/retrieval_metrics.json; irrelevant pairs score negative.
    # Below the top-1 floor no LLM call is made. The fallback threshold
    # counts a chunk relevant only when the relevance LLM call fails.
    sufficiency_top_n: int = 5
    sufficiency_min_relevant: int = 2
    sufficiency_min_top1_score: float = 0.0
    sufficiency_fallback_min_score: float = 3.0
    sufficiency_llm_max_tokens: int = 200
    sufficiency_chunk_char_cap: int = 1200
    sufficiency_filter_chunks: bool = True

    # triage_node. Tier 2 compares the query's max cosine against per-spec
    # clause centroids (one per clause prefix at this depth, cached to the
    # .npz below): < refuse_below -> refuse, >= accept_above -> retrieve,
    # in between -> one tier-3 LLM yes/no capped at triage_llm_max_tokens.
    # The two thresholds are PROVISIONAL until src/eval/triage_sweep.py has
    # been run against the BGE model (it writes the recommended values);
    # sweep_margin pads the golden-set extremes it derives them from.
    triage_centroids_path: str = "data/domain_centroids.npz"
    triage_centroid_clause_depth: int = 2
    triage_centroid_exclude_clauses: list[str] = ["0", "2"]  # frontmatter, References
    triage_refuse_below: float = 0.60
    triage_accept_above: float = 0.72
    triage_llm_max_tokens: int = 10
    triage_sweep_margin: float = 0.02

    # Per-chunk character cap on chunk text injected into the generate and
    # verify prompts (prompt tokens are 66% of measured spend).
    prompt_chunk_char_cap: int = 2400

    @property
    def project_root(self) -> Path:
        return _find_project_root()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
