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

    # Reserved for later phases (Phase 4 decision-nodes); unpopulated today.
    triage_decision: Optional[str]
    retrieval_scores: Optional[List[float]]
    sufficiency: Optional[str]
    rewrites: Optional[int]
    citations: Optional[List[Dict[str, Any]]]


# Valid values for termination_reason once a terminal node is reached.
TERMINATION_REASONS = {
    "verified",
    "verification_failed",
    "insufficient_context",
    "out_of_domain",
    "verifier_unavailable",
}
