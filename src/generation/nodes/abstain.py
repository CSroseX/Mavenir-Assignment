"""abstain_node: the terminal node for every non-verified exit from the
generate/verify graph, replacing flag_unverified_node.

flag_unverified_node used to pass the last-generated answer through
unchanged and unbadged — the fact that verification never passed was
visible only in state fields (verification_passed / termination_reason),
never in the answer text itself. A caller inspecting only the answer
string had no way to tell a fully-grounded response from one that
failed verification three times.

abstain_node instead returns a structured abstention: which source
clauses were actually retrieved, which specific claims (if any) failed
entailment, and why the graph is ending here (termination_reason). It
does not fabricate a "the system doesn't know this" message when the
answer was in fact fine except for an operational hiccup (an
unavailable verifier) — that distinction is exactly what Phase 2's
typed exception boundary (see verify_claims_node) made possible to draw.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.generation.state import GraphState


def _found_clauses(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"spec_id": c.get("spec_id"), "clause_id": c.get("clause_id")}
        for c in chunks or []
    ]


def abstain_node(state: GraphState) -> Dict[str, Any]:
    """
    Terminal node for any path that did not end in a verified answer.

    termination_reason must already be set by the node that routed here
    (verify_claims_node sets "verifier_unavailable" on a transport
    failure; the router falls through to this node on exhausted
    grounding retries, in which case termination_reason defaults to
    "verification_failed" here).
    """
    chunks = state.get("chunks", [])
    feedback = state.get("feedback", "")
    termination_reason = state.get("termination_reason") or "verification_failed"

    unsupported_claims: List[str] = []
    if feedback:
        # feedback is built in verify_claims_node as a header line followed
        # by one "- claim" per unsupported claim; recover the list form so
        # abstain's structured output doesn't re-parse prose at the call site.
        unsupported_claims = [
            line[2:].strip()
            for line in feedback.splitlines()
            if line.strip().startswith("- ")
        ]

    return {
        "answer": state.get("answer", ""),
        "termination_reason": termination_reason,
        "abstention": {
            "found_clauses": _found_clauses(chunks),
            "unsupported_claims": unsupported_claims,
            "termination_reason": termination_reason,
        },
    }
