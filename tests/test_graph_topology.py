"""End-to-end runs of the compiled Phase 4 graph with every external
dependency faked: retriever, embedder/centroids, the triage and grading
LLM seams, and the shared client generate/verify call. Asserts the node
path each question takes and how many LLM calls (hence tokens) it spent."""

import json
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import get_settings
from src.generation import nodes as gen_nodes
from src.generation.centroids import Centroids
from src.generation.graph import build_graph

DIM = 8


def unit(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


CENTROIDS = Centroids(vectors=np.vstack([unit(0)]), spec_ids=np.array(["23.501"]), clause_ids=np.array(["5.3"]))


def far_embed(texts):
    """Every query is orthogonal to the domain centroid (cosine 0)."""
    return np.vstack([unit(1) for _ in texts])


class FakeClient:
    """Stands in for gen_nodes.client.chat.completions.create: generate
    calls return prose, verify calls (json_object) return a verdict."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.generate_calls = 0
        self.verify_calls = 0

    def create(self, **kwargs):
        if kwargs.get("response_format") == {"type": "json_object"}:
            self.verify_calls += 1
            supported = self.verdicts.pop(0) if self.verdicts else False
            content = json.dumps({
                "is_supported": supported,
                "reasoning": "fake",
                "unsupported_claims": [] if supported else ["The AMF uses a 10-second timer."],
            })
        else:
            self.generate_calls += 1
            content = "The AMF handles registration and connection management."
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])


class FakeLLM:
    def __init__(self, response):
        self.response, self.calls = response, 0

    def __call__(self, messages, **kwargs):
        self.calls += 1
        return self.response(messages) if callable(self.response) else self.response


def relevance(n_true):
    """Fake judge: the first n_true passages in the prompt are relevant."""
    def respond(messages):
        n = messages[1]["content"].count("(Spec ")  # one header per passage
        return json.dumps({"relevant": [i < n_true for i in range(n)]})
    return respond


def chunks(score, n=5):
    return [
        {"spec_id": "23.501", "clause_id": f"6.2.{i}", "score": score, "cross_score": score,
         "content": f"AMF passage {i}"}
        for i in range(1, n + 1)
    ]


def run(graph, query):
    path, state = [], {"query": query}
    for output in graph.stream({"query": query}):
        for node, value in output.items():
            path.append(node)
            state.update(value)
    return path, state


@pytest.fixture
def client(monkeypatch):
    get_settings.cache_clear()
    fake = FakeClient(verdicts=[True])
    monkeypatch.setattr(gen_nodes.client.chat.completions, "create", fake.create)
    return fake


def make_graph(retriever, triage_llm="no", grade_llm=None):
    tl = FakeLLM(triage_llm)
    gl = FakeLLM(grade_llm or relevance(2))
    graph = build_graph(
        retriever=retriever,
        embed=far_embed,
        centroids_loader=lambda embed: CENTROIDS,
        triage_llm=tl,
        grade_llm=gl,
    )
    return graph, tl, gl


class TestOutOfDomain:
    def test_capital_of_france_ends_at_refuse_with_zero_llm_calls(self, client):
        retriever = MagicMock()
        graph, triage_llm, grade_llm = make_graph(retriever)

        path, state = run(graph, "What is the capital of France?")

        assert path == ["triage", "refuse"]
        assert state["termination_reason"] == "out_of_domain"
        assert state["triage_tier"] == "centroid"  # cosine 0 < refuse_below: no LLM needed
        assert client.generate_calls == 0 and client.verify_calls == 0
        assert triage_llm.calls == 0 and grade_llm.calls == 0  # zero tokens spent
        retriever.search.assert_not_called()
        assert state["chunks"] == []
        assert "out of scope" in state["answer"]


class TestSufficientContext:
    def test_happy_path(self, client):
        retriever = MagicMock()
        retriever.search.return_value = chunks(8.0)
        graph, _, grade_llm = make_graph(retriever, grade_llm=relevance(3))

        path, state = run(graph, "What is the role of the AMF?")

        assert path == ["triage", "retrieve", "grade_sufficiency", "generate", "verify"]
        assert state["termination_reason"] == "verified"
        assert state["attempt"] == 1
        assert grade_llm.calls == 1
        assert len(state["chunks"]) == 3  # only the chunks judged relevant reach generate

    def test_verification_failures_still_abstain(self, monkeypatch):
        get_settings.cache_clear()
        fake = FakeClient(verdicts=[False] * 10)
        monkeypatch.setattr(gen_nodes.client.chat.completions, "create", fake.create)
        retriever = MagicMock()
        retriever.search.return_value = chunks(8.0)
        graph, _, _ = make_graph(retriever)

        path, state = run(graph, "What is the role of the AMF?")

        max_retries = get_settings().max_retries
        assert path[:3] == ["triage", "retrieve", "grade_sufficiency"]
        assert path[3:-1] == ["generate", "verify"] * max_retries
        assert path[-1] == "abstain"
        assert state["termination_reason"] == "verification_failed"


class TestInsufficientContext:
    def test_poisoned_retrieval_routes_rewrite_then_abstain(self, client):
        """Retrieval returns only irrelevant chunks, before and after the
        rewrite: one rewrite, then an honest abstention — zero generate
        calls instead of 3 pointless regenerations."""
        retriever = MagicMock()
        retriever.search.return_value = chunks(8.0)
        graph, _, grade_llm = make_graph(retriever, grade_llm=relevance(0))

        path, state = run(graph, "What is the role of the AMF?")

        assert path == [
            "triage", "retrieve", "grade_sufficiency", "rewrite_query", "retrieve", "grade_sufficiency", "abstain",
        ]
        assert state["termination_reason"] == "insufficient_context"
        assert state["abstention"]["termination_reason"] == "insufficient_context"
        assert state["rewrites"] == 1
        assert client.generate_calls == 0 and client.verify_calls == 0
        assert grade_llm.calls == 2
        # The second retrieval ran on the rewritten query, not the original.
        second_query = retriever.search.call_args_list[1].args[0]
        assert "Access and Mobility Management Function" in second_query
        assert state["answer"]  # an explanation, not an empty string

    def test_low_cross_scores_abstain_without_any_llm_call(self, client):
        retriever = MagicMock()
        retriever.search.return_value = chunks(-9.0)
        graph, _, grade_llm = make_graph(retriever)

        path, state = run(graph, "What is the role of the AMF?")

        assert path[-1] == "abstain"
        assert state["termination_reason"] == "insufficient_context"
        assert grade_llm.calls == 0
        assert client.generate_calls == 0

    def test_empty_retrieval_rewrites_then_abstains(self, client):
        retriever = MagicMock()
        retriever.search.return_value = []
        graph, _, _ = make_graph(retriever)

        path, state = run(graph, "What is the role of the AMF?")

        assert path == [
            "triage", "retrieve", "grade_sufficiency", "rewrite_query", "retrieve", "grade_sufficiency", "abstain",
        ]
        assert state["termination_reason"] == "insufficient_context"

    def test_rewrite_that_changes_nothing_skips_the_second_retrieval(self, client):
        retriever = MagicMock()
        retriever.search.return_value = chunks(8.0)
        graph, _, _ = make_graph(retriever, grade_llm=relevance(0))

        # Tier 1 accepts on the title bigram; nothing to strip or expand.
        path, state = run(graph, "periodic registration update")

        assert path == ["triage", "retrieve", "grade_sufficiency", "rewrite_query", "abstain"]
        assert retriever.search.call_count == 1
        assert state["termination_reason"] == "insufficient_context"

    def test_rewrite_rescues_retrieval(self, client):
        retriever = MagicMock()
        retriever.search.return_value = chunks(8.0)
        verdicts = iter([0, 3])
        graph, _, _ = make_graph(
            retriever,
            grade_llm=lambda m: relevance(next(verdicts))(m),
        )

        path, state = run(graph, "What is the role of the AMF?")

        assert path == [
            "triage", "retrieve", "grade_sufficiency", "rewrite_query", "retrieve", "grade_sufficiency",
            "generate", "verify",
        ]
        assert state["termination_reason"] == "verified"
