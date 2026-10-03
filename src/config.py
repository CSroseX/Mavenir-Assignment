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
    fetch_k: int = 20
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

    # ── Cost estimation (USD per 1k tokens, clearly an ESTIMATE) ───
    cost_per_1k_input_tokens: float = 0.0015
    cost_per_1k_output_tokens: float = 0.002

    # ── LLM selection (from .env) ───────────────────────────────────
    use_remote_llm: bool = False
    open_router_api_key: str = ""
    openai_api_key: str = ""
    model_name: str = ""

    @property
    def project_root(self) -> Path:
        return _find_project_root()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
