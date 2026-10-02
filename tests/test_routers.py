from langgraph.graph import END

from src.generation.routers import should_retry


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
