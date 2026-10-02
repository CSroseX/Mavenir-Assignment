import json
import types

import httpx
import openai
import pytest

from src.generation import nodes as gen_nodes


def make_chunks():
    return [{"spec_id": "23.501", "clause_id": "6.2.1", "content": "The AMF handles registration management."}]


def make_response(content: str):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


class TestVerifyClaimsNodeTransportFailure:
    def test_429_sets_verifier_unavailable_without_incrementing_grounding_retries(self, monkeypatch):
        request = httpx.Request("POST", "http://x")
        response = httpx.Response(429, request=request)
        error = openai.APIStatusError("rate limited", response=response, body=None)

        def raise_429(*args, **kwargs):
            raise error

        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", raise_429)

        state = {
            "answer": "The AMF handles registration management.",
            "chunks": make_chunks(),
            "grounding_retries": 0,
            "transport_attempts": 0,
        }
        result = gen_nodes.verify_claims_node(state)

        assert result["termination_reason"] == "verifier_unavailable"
        assert result["verification_passed"] is False
        assert result["grounding_retries"] == 0
        assert result["transport_attempts"] == 1
        assert result["feedback"] == ""

    def test_connection_error_sets_verifier_unavailable(self, monkeypatch):
        request = httpx.Request("POST", "http://x")
        error = openai.APIConnectionError(request=request)

        def raise_conn(*args, **kwargs):
            raise error

        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", raise_conn)

        state = {"answer": "x", "chunks": make_chunks(), "grounding_retries": 2, "transport_attempts": 0}
        result = gen_nodes.verify_claims_node(state)

        assert result["termination_reason"] == "verifier_unavailable"
        assert result["grounding_retries"] == 2  # unchanged, not burned on a transport failure
        assert result["transport_attempts"] == 1

    def test_transport_failure_does_not_inject_fake_feedback(self, monkeypatch):
        request = httpx.Request("POST", "http://x")
        error = openai.APIConnectionError(request=request)
        monkeypatch.setattr(
            gen_nodes.client.chat.completions, "create", lambda *a, **k: (_ for _ in ()).throw(error)
        )
        state = {"answer": "x", "chunks": make_chunks(), "grounding_retries": 0, "transport_attempts": 0}
        result = gen_nodes.verify_claims_node(state)
        assert result["feedback"] == ""
        assert "could not complete" not in result["feedback"]


class TestVerifyClaimsNodeMalformedJson:
    def test_malformed_json_reasks_once_then_succeeds(self, monkeypatch):
        calls = {"count": 0}

        def flaky_create(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return make_response("not valid json")
            return make_response(json.dumps({"is_supported": True, "unsupported_claims": []}))

        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", flaky_create)

        state = {"answer": "x", "chunks": make_chunks(), "grounding_retries": 0, "transport_attempts": 0}
        result = gen_nodes.verify_claims_node(state)

        assert calls["count"] == 2
        assert result["verification_passed"] is True
        assert result["termination_reason"] == "verified"

    def test_malformed_json_twice_is_treated_as_transport_failure(self, monkeypatch):
        def always_bad(*args, **kwargs):
            return make_response("still not json")

        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", always_bad)

        state = {"answer": "x", "chunks": make_chunks(), "grounding_retries": 0, "transport_attempts": 0}
        result = gen_nodes.verify_claims_node(state)

        assert result["termination_reason"] == "verifier_unavailable"
        assert result["grounding_retries"] == 0
        assert result["transport_attempts"] == 2


class TestVerifyClaimsNodeGenuineGroundingFailure:
    def test_valid_json_unsupported_increments_grounding_retries_and_injects_feedback(self, monkeypatch):
        def unsupported(*args, **kwargs):
            return make_response(json.dumps({
                "is_supported": False,
                "reasoning": "claim not in source",
                "unsupported_claims": ["The AMF uses a 10-second timer."],
            }))

        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", unsupported)

        state = {"answer": "x", "chunks": make_chunks(), "grounding_retries": 0, "transport_attempts": 0}
        result = gen_nodes.verify_claims_node(state)

        assert result["verification_passed"] is False
        assert result["grounding_retries"] == 1
        assert result["termination_reason"] is None
        assert "10-second timer" in result["feedback"]

    def test_valid_json_supported_sets_verified(self, monkeypatch):
        def supported(*args, **kwargs):
            return make_response(json.dumps({"is_supported": True, "unsupported_claims": []}))

        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", supported)

        state = {"answer": "x", "chunks": make_chunks(), "grounding_retries": 0, "transport_attempts": 0}
        result = gen_nodes.verify_claims_node(state)

        assert result["verification_passed"] is True
        assert result["grounding_retries"] == 0
        assert result["termination_reason"] == "verified"


class TestGenerateNodeAttemptCounter:
    def test_first_pass_records_attempt_one(self, monkeypatch):
        monkeypatch.setattr(
            gen_nodes.client.chat.completions,
            "create",
            lambda *a, **k: make_response("The AMF handles registration management."),
        )
        state = {"query": "What is the AMF?", "chunks": make_chunks()}
        result = gen_nodes.generate_node(state)
        assert result["attempt"] == 1
        assert result["retries"] == 0

    def test_second_pass_records_attempt_two(self, monkeypatch):
        monkeypatch.setattr(
            gen_nodes.client.chat.completions,
            "create",
            lambda *a, **k: make_response("The AMF handles registration management."),
        )
        state = {"query": "What is the AMF?", "chunks": make_chunks(), "attempt": 1, "feedback": "fix it"}
        result = gen_nodes.generate_node(state)
        assert result["attempt"] == 2
        assert result["retries"] == 1
