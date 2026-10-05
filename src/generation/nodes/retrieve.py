"""retrieve_node: fetches relevant chunks from Qdrant as the first
graph node, replacing the external retriever.search() calls that
used to live in app.py, verify.py, and the eval runners."""

from __future__ import annotations

from typing import Any, Dict

from src.config import get_settings
from src.obs.logging import get_logger

log = get_logger(__name__)


def make_retrieve_node(retriever=None):
    """Return a retrieve_node closure, optionally bound to an existing
    Retriever instance so the caller controls construction and caching."""

    _retriever_holder = [retriever]

    def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
        if _retriever_holder[0] is None:
            from src.retrieval.retriever import Retriever
            _retriever_holder[0] = Retriever()

        cfg = get_settings()
        query = state["query"]
        chunks = _retriever_holder[0].search(query, top_k=cfg.default_top_k)
        retrieval_scores = [c.get("score", 0.0) for c in chunks]

        log.info("retrieve_node.done", query=query, num_chunks=len(chunks))

        result: Dict[str, Any] = {
            "chunks": chunks,
            "retrieval_scores": retrieval_scores,
        }
        if not chunks:
            result["termination_reason"] = "insufficient_context"
        return result

    return retrieve_node
