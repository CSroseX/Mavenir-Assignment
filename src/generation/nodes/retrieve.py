"""retrieve_node: fetches relevant chunks from Qdrant as the first
graph node, replacing the external retriever.search() calls that
used to live in app.py, verify.py, and the eval runners."""

from __future__ import annotations

from typing import Any, Dict, List

from src.config import get_settings
from src.obs.logging import get_logger

log = get_logger(__name__)


def _chunk_key(chunk: Dict[str, Any]):
    return (chunk.get("spec_id"), chunk.get("clause_id"), str(chunk.get("content", ""))[:200])


def merge_ranked(result_lists: List[List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
    """Round-robin merge of per-query ranked lists, deduplicated, capped at
    top_k. Interleaving (rather than sorting on score) guarantees each
    sub-query of a split cross-generation rewrite gets its best chunks into
    context — cross-encoder scores from different sub-queries aren't
    comparable anyway."""
    merged: List[Dict[str, Any]] = []
    seen = set()
    depth = max((len(r) for r in result_lists), default=0)
    for rank in range(depth):
        for results in result_lists:
            if rank < len(results):
                key = _chunk_key(results[rank])
                if key not in seen:
                    seen.add(key)
                    merged.append(results[rank])
            if len(merged) >= top_k:
                return merged
    return merged


def make_retrieve_node(retriever=None, *, retriever_factory=None):
    """Return a retrieve_node closure, optionally bound to an existing
    Retriever instance so the caller controls construction and caching.

    `retriever_factory` (a zero-arg callable) lets build_graph share one
    lazily-constructed Retriever between this node and triage, which reuses
    its embedding model.
    """

    _retriever_holder = [retriever]

    def _get_retriever():
        if _retriever_holder[0] is None:
            if retriever_factory is not None:
                _retriever_holder[0] = retriever_factory()
            else:
                from src.retrieval.retriever import Retriever
                _retriever_holder[0] = Retriever()
        return _retriever_holder[0]

    def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
        cfg = get_settings()
        query = state["query"]
        # After rewrite_query_node, retrieve on the rewritten query (or its
        # 4G/5G split); `query` itself stays the user's question.
        queries = state.get("retrieval_queries") or [query]
        retriever = _get_retriever()

        if len(queries) == 1:
            chunks = retriever.search(queries[0], top_k=cfg.default_top_k)
        else:
            chunks = merge_ranked(
                [retriever.search(q, top_k=cfg.default_top_k) for q in queries], cfg.default_top_k
            )
        retrieval_scores = [c.get("score", 0.0) for c in chunks]

        log.info("retrieve_node.done", query=query, retrieval_queries=queries, num_chunks=len(chunks))

        result: Dict[str, Any] = {
            "chunks": chunks,
            "retrieval_scores": retrieval_scores,
        }
        if not chunks:
            result["termination_reason"] = "insufficient_context"
        return result

    return retrieve_node
