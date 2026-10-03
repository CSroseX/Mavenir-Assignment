"""Tests for Retriever chunk-loading: anchored glob and loud failure on zero chunks."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_chunk_dir(tmp_path, chunks_data=None):
    """Create a temporary project structure with data/chunks/."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    chunk_dir = data_dir / "chunks"
    chunk_dir.mkdir()
    if chunks_data is not None:
        for filename, data in chunks_data.items():
            with open(chunk_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f)
    return tmp_path


class TestRetrieverChunkLoading:
    """Test that the Retriever loads chunks from cfg.project_root and fails loudly."""

    def _import_and_patch(self, project_root):
        """Import retriever module with heavy deps mocked out."""
        mock_cfg = MagicMock()
        mock_cfg.project_root = Path(project_root)
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.embedding_model = "BAAI/bge-small-en-v1.5"
        mock_cfg.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return mock_cfg

    @patch("src.retrieval.retriever.CrossEncoder")
    @patch("src.retrieval.retriever.TextEmbedding")
    @patch("src.retrieval.retriever.QdrantClient")
    @patch("src.retrieval.retriever.get_settings")
    def test_raises_on_no_chunk_files(self, mock_settings, mock_qclient, mock_embed, mock_ce, tmp_path):
        project_root = _make_chunk_dir(tmp_path)
        mock_settings.return_value = self._import_and_patch(project_root)
        from src.retrieval.retriever import Retriever
        with pytest.raises(FileNotFoundError, match="No chunk files found"):
            Retriever()

    @patch("src.retrieval.retriever.CrossEncoder")
    @patch("src.retrieval.retriever.TextEmbedding")
    @patch("src.retrieval.retriever.QdrantClient")
    @patch("src.retrieval.retriever.get_settings")
    def test_raises_on_empty_chunk_files(self, mock_settings, mock_qclient, mock_embed, mock_ce, tmp_path):
        project_root = _make_chunk_dir(tmp_path, {"23.501_chunks.json": []})
        mock_settings.return_value = self._import_and_patch(project_root)
        from src.retrieval.retriever import Retriever
        with pytest.raises(ValueError, match="contained no clause IDs"):
            Retriever()

    @patch("src.retrieval.retriever.CrossEncoder")
    @patch("src.retrieval.retriever.TextEmbedding")
    @patch("src.retrieval.retriever.QdrantClient")
    @patch("src.retrieval.retriever.get_settings")
    def test_loads_clause_ids_from_project_root(self, mock_settings, mock_qclient, mock_embed, mock_ce, tmp_path):
        chunks = [
            {"clause_id": "5.1", "content": "test"},
            {"clause_id": "5.2", "content": "test"},
            {"clause_id": "5.2.1", "content": "test"},
        ]
        project_root = _make_chunk_dir(tmp_path, {"23.501_chunks.json": chunks})
        mock_settings.return_value = self._import_and_patch(project_root)
        from src.retrieval.retriever import Retriever
        r = Retriever()
        assert r.valid_clause_ids == {"5.1", "5.2", "5.2.1"}

    @patch("src.retrieval.retriever.CrossEncoder")
    @patch("src.retrieval.retriever.TextEmbedding")
    @patch("src.retrieval.retriever.QdrantClient")
    @patch("src.retrieval.retriever.get_settings")
    def test_loads_from_different_cwd(self, mock_settings, mock_qclient, mock_embed, mock_ce, tmp_path):
        """Retriever loads chunks regardless of cwd, using project_root."""
        chunks = [{"clause_id": "6.1", "content": "test"}]
        project_root = _make_chunk_dir(tmp_path, {"24.501_chunks.json": chunks})
        mock_settings.return_value = self._import_and_patch(project_root)
        original_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            from src.retrieval.retriever import Retriever
            r = Retriever()
            assert "6.1" in r.valid_clause_ids
        finally:
            os.chdir(original_cwd)
