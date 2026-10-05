from unittest.mock import MagicMock

from src.generation.nodes.retrieve import make_retrieve_node


class TestMakeRetrieveNode:
    def test_returns_chunks_and_scores_from_retriever(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"spec_id": "23.501", "clause_id": "6.2.1", "score": 0.95, "content": "AMF info"},
            {"spec_id": "23.501", "clause_id": "5.1", "score": 0.80, "content": "Overview"},
        ]

        node = make_retrieve_node(mock_retriever)
        result = node({"query": "What is the AMF?"})

        mock_retriever.search.assert_called_once()
        assert len(result["chunks"]) == 2
        assert result["retrieval_scores"] == [0.95, 0.80]

    def test_returns_empty_with_termination_reason_when_nothing_found(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        node = make_retrieve_node(mock_retriever)
        result = node({"query": "something obscure"})

        assert result["chunks"] == []
        assert result["retrieval_scores"] == []
        assert result["termination_reason"] == "insufficient_context"

    def test_no_termination_reason_when_chunks_found(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"spec_id": "23.501", "clause_id": "6.2.1", "score": 0.9, "content": "x"},
        ]

        node = make_retrieve_node(mock_retriever)
        result = node({"query": "test"})

        assert "termination_reason" not in result

    def test_lazy_construction_when_no_retriever_passed(self, monkeypatch):
        """When no retriever is given, the node lazily constructs one on
        first call. We mock the Retriever class to avoid needing Qdrant."""
        mock_instance = MagicMock()
        mock_instance.search.return_value = [
            {"spec_id": "24.501", "clause_id": "5.5.1", "score": 0.7, "content": "NAS"},
        ]
        mock_cls = MagicMock(return_value=mock_instance)

        import src.retrieval.retriever as retriever_mod
        monkeypatch.setattr(retriever_mod, "Retriever", mock_cls)

        node = make_retrieve_node(None)
        result = node({"query": "NAS procedure"})

        mock_cls.assert_called_once()
        assert result["chunks"][0]["spec_id"] == "24.501"

    def test_uses_default_top_k_from_config(self, monkeypatch):
        from src.config import get_settings
        get_settings.cache_clear()
        cfg = get_settings()

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        node = make_retrieve_node(mock_retriever)
        node({"query": "test"})

        _, kwargs = mock_retriever.search.call_args
        assert kwargs["top_k"] == cfg.default_top_k

    def test_scores_fallback_to_zero_when_missing(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"spec_id": "23.501", "clause_id": "6.2.1", "content": "no score field"},
        ]

        node = make_retrieve_node(mock_retriever)
        result = node({"query": "test"})

        assert result["retrieval_scores"] == [0.0]
