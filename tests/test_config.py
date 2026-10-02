from src.config import Settings, get_settings


class TestSettingsDefaults:
    def test_qdrant_defaults(self):
        s = Settings()
        assert s.qdrant_url == "http://localhost:6333"
        assert s.qdrant_collection == "3gpp_specs"

    def test_model_defaults(self):
        s = Settings()
        assert s.embedding_model == "BAAI/bge-small-en-v1.5"
        assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert s.local_llm_model == "qwen3.5:4b"
        assert s.remote_llm_model_openrouter == "openai/gpt-oss-20b"
        assert s.remote_llm_model_openai == "gpt-3.5-turbo"

    def test_vector_size(self):
        s = Settings()
        assert s.vector_size == 384

    def test_retrieval_defaults(self):
        s = Settings()
        assert s.default_top_k == 5
        assert s.fetch_k == 20
        assert s.verify_top_k == 3

    def test_generation_defaults(self):
        s = Settings()
        assert s.generate_temperature == 0.3
        assert s.verify_temperature == 0.0
        assert s.verify_max_tokens == 2000
        assert s.max_retries == 3

    def test_chunking_defaults(self):
        s = Settings()
        assert s.chunk_soft_limit == 400
        assert s.chunk_hard_cap == 800

    def test_indexer_defaults(self):
        s = Settings()
        assert s.indexer_batch_size == 100

    def test_eval_defaults(self):
        s = Settings()
        assert s.eval_sleep_seconds == 5

    def test_llm_selection_defaults(self):
        s = Settings()
        assert s.use_remote_llm is False

    def test_project_root_is_valid(self):
        s = Settings()
        root = s.project_root
        assert (root / "src").is_dir()


class TestSettingsEnvOverride:
    def test_qdrant_url_override(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://remote-qdrant:6333")
        s = Settings()
        assert s.qdrant_url == "http://remote-qdrant:6333"

    def test_collection_name_override(self, monkeypatch):
        monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
        s = Settings()
        assert s.qdrant_collection == "test_collection"

    def test_top_k_override(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TOP_K", "10")
        s = Settings()
        assert s.default_top_k == 10

    def test_max_retries_override(self, monkeypatch):
        monkeypatch.setenv("MAX_RETRIES", "5")
        s = Settings()
        assert s.max_retries == 5

    def test_embedding_model_override(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        s = Settings()
        assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_vector_size_override(self, monkeypatch):
        monkeypatch.setenv("VECTOR_SIZE", "768")
        s = Settings()
        assert s.vector_size == 768

    def test_temperature_override(self, monkeypatch):
        monkeypatch.setenv("GENERATE_TEMPERATURE", "0.7")
        s = Settings()
        assert s.generate_temperature == 0.7

    def test_use_remote_llm_override(self, monkeypatch):
        monkeypatch.setenv("USE_REMOTE_LLM", "true")
        s = Settings()
        assert s.use_remote_llm is True

    def test_eval_sleep_override(self, monkeypatch):
        monkeypatch.setenv("EVAL_SLEEP_SECONDS", "10")
        s = Settings()
        assert s.eval_sleep_seconds == 10

    def test_chunk_limits_override(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SOFT_LIMIT", "500")
        monkeypatch.setenv("CHUNK_HARD_CAP", "1000")
        s = Settings()
        assert s.chunk_soft_limit == 500
        assert s.chunk_hard_cap == 1000


class TestGetSettings:
    def test_singleton_returns_same_instance(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_gives_fresh_instance(self):
        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1 is not s2
