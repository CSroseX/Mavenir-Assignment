from src.generation.nodes.abstain import abstain_node


def make_chunks():
    return [
        {"spec_id": "23.501", "clause_id": "6.2.1", "content": "x"},
        {"spec_id": "24.501", "clause_id": "9.11.3.4", "content": "y"},
    ]


class TestAbstainNodeFoundClauses:
    def test_lists_found_clauses_from_chunks(self):
        state = {
            "answer": "The AMF handles registration.",
            "chunks": make_chunks(),
            "feedback": "",
            "termination_reason": "verification_failed",
        }
        result = abstain_node(state)
        assert result["abstention"]["found_clauses"] == [
            {"spec_id": "23.501", "clause_id": "6.2.1"},
            {"spec_id": "24.501", "clause_id": "9.11.3.4"},
        ]

    def test_empty_chunks_yields_empty_found_clauses(self):
        state = {"answer": "x", "chunks": [], "feedback": "", "termination_reason": "insufficient_context"}
        result = abstain_node(state)
        assert result["abstention"]["found_clauses"] == []


class TestAbstainNodeUnsupportedClaims:
    def test_parses_unsupported_claims_from_feedback(self):
        feedback = (
            "The following claims were NOT supported by the source text and must be removed or corrected:\n"
            "- The AMF uses a 10-second timer.\n"
            "- The UE retries exactly 3 times."
        )
        state = {
            "answer": "x",
            "chunks": make_chunks(),
            "feedback": feedback,
            "termination_reason": "verification_failed",
        }
        result = abstain_node(state)
        assert result["abstention"]["unsupported_claims"] == [
            "The AMF uses a 10-second timer.",
            "The UE retries exactly 3 times.",
        ]

    def test_no_feedback_yields_empty_unsupported_claims(self):
        state = {"answer": "x", "chunks": make_chunks(), "feedback": "", "termination_reason": "verifier_unavailable"}
        result = abstain_node(state)
        assert result["abstention"]["unsupported_claims"] == []


class TestAbstainNodeTerminationReason:
    def test_preserves_verifier_unavailable(self):
        state = {"answer": "x", "chunks": [], "feedback": "", "termination_reason": "verifier_unavailable"}
        result = abstain_node(state)
        assert result["termination_reason"] == "verifier_unavailable"
        assert result["abstention"]["termination_reason"] == "verifier_unavailable"

    def test_preserves_insufficient_context(self):
        state = {"answer": "x", "chunks": [], "feedback": "", "termination_reason": "insufficient_context"}
        result = abstain_node(state)
        assert result["termination_reason"] == "insufficient_context"

    def test_defaults_to_verification_failed_if_unset(self):
        state = {"answer": "x", "chunks": [], "feedback": ""}
        result = abstain_node(state)
        assert result["termination_reason"] == "verification_failed"

    def test_answer_passed_through_unchanged(self):
        state = {"answer": "The AMF handles registration.", "chunks": [], "feedback": ""}
        result = abstain_node(state)
        assert result["answer"] == "The AMF handles registration."
