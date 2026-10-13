"""GraphState: the shape of data threaded through the generate/verify graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # Core fields, populated on every run.
    query: str
    chunks: List[Dict[str, Any]]
    answer: str
    verification_passed: bool
    feedback: str
    retries: int  # legacy counter kept for app.py / eval scripts; see `attempt`

    # Phase 2: truthful counters and typed termination state.
    attempt: int
    grounding_retries: int
    transport_attempts: int
    termination_reason: Optional[str]
    abstention: Optional[Dict[str, Any]]  # set by abstain_node on any non-verified exit

    # Phase 4 decision-nodes.
    triage_decision: Optional[str]  # in_domain | out_of_domain
    triage_tier: Optional[str]  # lexical | centroid | llm | fail_open
    triage_score: Optional[float]  # max centroid cosine, when tier 2 ran
    triage_signals: Optional[Dict[str, List[str]]]
    retrieval_scores: Optional[List[float]]
    sufficiency: Optional[str]
    rewrites: Optional[int]
    # rewrite_query_node: what retrieve_node searches on after a rewrite
    # (one query, or a 4G/5G split); `query` stays the user's question.
    retrieval_queries: Optional[List[str]]
    rewrite_noop: Optional[bool]
    # grade_sufficiency_node: after_grade routes on n_relevant_chunks.
    n_relevant_chunks: Optional[int]
    relevance_flags: Optional[List[bool]]
    top1_cross_score: Optional[float]
    sufficiency_signal: Optional[str]  # llm | cross_encoder_floor | cross_encoder_fallback | no_chunks

    # Reserved for later phases; unpopulated today.
    citations: Optional[List[Dict[str, Any]]]


# Valid values for termination_reason once a terminal node is reached.
TERMINATION_REASONS = {
    "verified",
    "verification_failed",
    "insufficient_context",
    "out_of_domain",
    "verifier_unavailable",
}
