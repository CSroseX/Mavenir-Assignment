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


class TestRetrieveAfterRewrite:
    def test_uses_rewritten_query_not_original(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        node = make_retrieve_node(mock_retriever)
        node({"query": "What is the AMF?", "retrieval_queries": ["the AMF Access and Mobility Management Function"]})

        args, _ = mock_retriever.search.call_args
        assert args[0] == "the AMF Access and Mobility Management Function"

    def test_split_queries_are_interleaved_and_deduplicated(self):
        def search(q, top_k):
            if "24.301" in q:
                return [
                    {"spec_id": "24.301", "clause_id": "5.5.1", "score": 3.0, "content": "attach"},
                    {"spec_id": "24.301", "clause_id": "5.5.2", "score": 2.0, "content": "detach"},
                    {"spec_id": "23.501", "clause_id": "6.2.7", "score": 1.0, "content": "udm"},
                ]
            return [
                {"spec_id": "23.501", "clause_id": "6.2.7", "score": 9.0, "content": "udm"},
                {"spec_id": "23.501", "clause_id": "6.2.1", "score": 8.0, "content": "amf"},
            ]

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = search

        node = make_retrieve_node(mock_retriever)
        result = node({"query": "q", "retrieval_queries": ["MME attach TS 24.301", "UDM"]})

        assert mock_retriever.search.call_count == 2
        assert [c["clause_id"] for c in result["chunks"]] == ["5.5.1", "6.2.7", "5.5.2", "6.2.1"]

    def test_merge_caps_at_top_k(self):
        from src.generation.nodes.retrieve import merge_ranked

        a = [{"spec_id": "a", "clause_id": str(i), "content": f"a{i}"} for i in range(5)]
        b = [{"spec_id": "b", "clause_id": str(i), "content": f"b{i}"} for i in range(5)]
        merged = merge_ranked([a, b], 5)
        assert [c["spec_id"] for c in merged] == ["a", "b", "a", "b", "a"]

    def test_retriever_factory_is_used_lazily(self):
        mock_instance = MagicMock()
        mock_instance.search.return_value = []
        factory = MagicMock(return_value=mock_instance)

        node = make_retrieve_node(retriever_factory=factory)
        factory.assert_not_called()
        node({"query": "x"})
        node({"query": "y"})
        factory.assert_called_once()
