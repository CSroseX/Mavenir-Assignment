from langgraph.graph import END

from src.generation.routers import after_grade, after_rewrite, after_triage, should_retry


class TestPhase4RoutersCrossProduct:
    """The Phase 4 routers over the cross-product of the state fields they
    read (and fields they must ignore)."""

    def test_after_triage(self):
        for decision in (None, "in_domain", "out_of_domain"):
            for tier in (None, "lexical", "centroid", "llm", "fail_open"):
                for termination_reason in (None, "out_of_domain"):
                    state = {"triage_decision": decision, "triage_tier": tier,
                             "termination_reason": termination_reason}
                    expected = "refuse" if decision == "out_of_domain" else "retrieve"
                    assert after_triage(state) == expected, state

    def test_after_rewrite(self):
        for noop in (None, False, True):
            for rewrites in (None, 0, 1, 2):
                state = {"rewrite_noop": noop, "rewrites": rewrites}
                assert after_rewrite(state) == ("abstain" if noop else "retrieve"), state

    def test_empty_retrieval_never_reaches_generate(self):
        # Empty retrieval grades to zero relevant chunks: rewrite, then abstain.
        from src.config import get_settings
        get_settings.cache_clear()
        cap = get_settings().max_rewrites
        assert after_grade({"chunks": [], "n_relevant_chunks": 0, "rewrites": 0}) == "rewrite_query"
        assert after_grade({"chunks": [], "n_relevant_chunks": 0, "rewrites": cap}) == "abstain"


class TestAfterRewrite:
    def test_real_rewrite_routes_to_retrieve(self):
        assert after_rewrite({"rewrite_noop": False, "rewrites": 1}) == "retrieve"

    def test_noop_routes_to_abstain(self):
        assert after_rewrite({"rewrite_noop": True, "rewrites": 1}) == "abstain"

    def test_missing_flag_routes_to_retrieve(self):
        assert after_rewrite({}) == "retrieve"


class TestShouldRetryVerifiedPasses:
    def test_verified_routes_to_end_with_zero_retries(self):
        state = {"verification_passed": True, "grounding_retries": 0}
        assert should_retry(state) == END

    def test_verified_routes_to_end_even_with_prior_retries(self):
        state = {"verification_passed": True, "grounding_retries": 2}
        assert should_retry(state) == END


class TestShouldRetryVerifierUnavailable:
    def test_transport_failure_routes_to_flag_unverified_immediately(self):
        # Even with zero grounding_retries used, a transport failure must
        # not be retried as if it were a grounding failure.
        state = {
            "verification_passed": False,
            "grounding_retries": 0,
            "termination_reason": "verifier_unavailable",
        }
        assert should_retry(state) == "flag_unverified"


class TestShouldRetryGroundingFailure:
    def test_below_max_retries_routes_to_generate(self, monkeypatch):
        from src.config import get_settings
        get_settings.cache_clear()
        state = {"verification_passed": False, "grounding_retries": 1, "termination_reason": None}
        assert should_retry(state) == "generate"

    def test_at_max_retries_routes_to_flag_unverified(self):
        from src.config import get_settings
        get_settings.cache_clear()
        cfg = get_settings()
        state = {
            "verification_passed": False,
            "grounding_retries": cfg.max_retries,
            "termination_reason": None,
        }
        assert should_retry(state) == "flag_unverified"

    def test_above_max_retries_routes_to_flag_unverified(self):
        from src.config import get_settings
        get_settings.cache_clear()
        cfg = get_settings()
        state = {
            "verification_passed": False,
            "grounding_retries": cfg.max_retries + 5,
            "termination_reason": None,
        }
        assert should_retry(state) == "flag_unverified"

    def test_falls_back_to_legacy_retries_field_if_grounding_retries_absent(self):
        from src.config import get_settings
        get_settings.cache_clear()
        cfg = get_settings()
        state = {"verification_passed": False, "retries": cfg.max_retries, "termination_reason": None}
        assert should_retry(state) == "flag_unverified"


class TestShouldRetryStateCrossProduct:
    """Exhaustive-ish cross product of the fields should_retry reads."""

    def test_all_combinations(self):
        from src.config import get_settings
        get_settings.cache_clear()
        max_retries = get_settings().max_retries

        for verified in (True, False):
            for grounding_retries in (0, 1, max_retries - 1, max_retries, max_retries + 1):
                for termination_reason in (None, "verifier_unavailable", "verification_failed"):
                    state = {
                        "verification_passed": verified,
                        "grounding_retries": grounding_retries,
                        "termination_reason": termination_reason,
                    }
                    result = should_retry(state)
                    if verified:
                        assert result == END, state
                    elif termination_reason == "verifier_unavailable":
                        assert result == "flag_unverified", state
                    elif grounding_retries >= max_retries:
                        assert result == "flag_unverified", state
                    else:
                        assert result == "generate", state
