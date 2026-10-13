from types import SimpleNamespace

import openai
import pytest

from src.obs import tracing


@pytest.fixture(autouse=True)
def no_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    tracing.reset_usage()
    yield
    tracing.reset_usage()


def make_fake_client(usage):
    def create(**kwargs):
        return SimpleNamespace(usage=usage, choices=[])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class TestFactoryWithoutLangfuse:
    def test_tracing_disabled_when_host_unset(self):
        assert tracing.tracing_enabled() is False

    def test_returns_plain_openai_client(self):
        client = tracing.make_llm_client(api_key="k", base_url="http://localhost:1/v1")
        assert isinstance(client, openai.OpenAI)

    def test_observe_bare_passes_through(self):
        def node(state):
            return {"seen": state["x"]}

        wrapped = tracing.observe(node)
        assert wrapped({"x": 1}) == {"seen": 1}
        assert wrapped.__name__ == "node"

    def test_observe_with_args_passes_through(self):
        def node(state):
            return {"seen": state["x"]}

        assert tracing.observe(name="generate")(node)({"x": 2}) == {"seen": 2}

    def test_observe_switches_to_langfuse_once_enabled_at_call_time(self, monkeypatch):
        # Decorated while tracing is off (as at import, before .env loads) ...
        calls = []

        def fake_lf_observe(fn, **kwargs):
            calls.append(kwargs)
            return lambda *a, **k: ("traced", fn(*a, **k))

        @tracing.observe(name="generate")
        def node(state):
            return state["x"]

        assert node({"x": 3}) == 3
        # ... then enabled: the next call goes through langfuse.observe.
        monkeypatch.setattr(tracing, "tracing_enabled", lambda: True)
        monkeypatch.setattr("langfuse.observe", fake_lf_observe)
        assert node({"x": 4}) == ("traced", 4)
        assert node({"x": 5}) == ("traced", 5)
        assert calls == [{"name": "generate"}]

    def test_update_trace_attributes_is_noop(self):
        tracing.update_trace_attributes(termination_reason="verified", attempt=1)


class TestUsageRecording:
    def test_usage_recorded_per_call(self):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        client = tracing._wrap_usage(make_fake_client(usage))

        cursor = tracing.usage_snapshot()
        client.chat.completions.create(model="m", messages=[])
        client.chat.completions.create(model="m", messages=[])

        records = tracing.usage_since(cursor)
        assert len(records) == 2
        assert records[0] == {"model": "m", "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

    def test_cursor_excludes_earlier_calls(self):
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        client = tracing._wrap_usage(make_fake_client(usage))
        client.chat.completions.create(model="m", messages=[])

        cursor = tracing.usage_snapshot()
        client.chat.completions.create(model="m", messages=[])
        assert len(tracing.usage_since(cursor)) == 1

    def test_response_without_usage_is_skipped(self):
        client = tracing._wrap_usage(make_fake_client(None))
        client.chat.completions.create(model="m", messages=[])
        assert tracing.usage_since(0) == []

    def test_live_client_create_is_wrapped(self):
        client = tracing.make_llm_client(api_key="k", base_url="http://localhost:1/v1")
        assert client.chat.completions.create.__wrapped__ is not None


class TestTraceAttributes:
    def test_collects_the_five_signals(self):
        state = {
            "termination_reason": "verified",
            "attempt": 2,
            "chunks": [{"cross_score": 6.1, "score": 7.0}, {"cross_score": 1.0}],
            "n_relevant_chunks": 3,
            "triage_decision": "retrieve",
        }
        assert tracing.trace_attributes(state) == {
            "termination_reason": "verified",
            "attempt": 2,
            "top1_cross_score": 6.1,
            "n_relevant_chunks": 3,
            "triage_decision": "retrieve",
        }

    def test_falls_back_to_score_and_drops_unset(self):
        assert tracing.trace_attributes({"chunks": [{"score": 4.2}]}) == {"top1_cross_score": 4.2}

    def test_empty_state(self):
        assert tracing.trace_attributes({}) == {}


class TestTracedNodeWithoutLangfuse:
    def test_passes_state_through(self):
        node = tracing.traced_node("retrieve", lambda state: {"chunks": [], "seen": state["query"]})
        assert node({"query": "q"}) == {"chunks": [], "seen": "q"}

    def test_trace_query_is_noop(self):
        with tracing.trace_query("q"):
            pass


@pytest.fixture
def sink_server():
    """Local HTTP endpoint that accepts (and discards) span exports, so the
    langfuse exporter neither needs a real backend nor retries after the test."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Sink)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


class TestOneQueryOneTrace:
    """Real langfuse SDK against a sink endpoint, not a Langfuse backend."""

    def test_node_spans_nest_under_each_query_root(self, monkeypatch, sink_server):
        pytest.importorskip("langfuse")
        from typing import TypedDict

        from langfuse import get_client
        from langgraph.graph import END, START, StateGraph

        monkeypatch.setenv("LANGFUSE_HOST", sink_server)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        class S(TypedDict, total=False):
            query: str
            n: int

        seen = []

        def make(name):
            def node(state):
                client = get_client()
                seen.append((name, client.get_current_trace_id(), client.get_current_observation_id()))
                return {"n": len(seen)}
            return node

        graph = StateGraph(S)
        graph.add_node("a", tracing.traced_node("a", make("a")))
        graph.add_node("b", tracing.traced_node("b", make("b")))
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        roots = []
        for q in ["q1", "q2"]:
            with tracing.trace_query(q):
                roots.append(get_client().get_current_trace_id())
                list(app.stream({"query": q}))

        assert roots[0] and roots[0] != roots[1]
        assert [s[1] for s in seen] == [roots[0], roots[0], roots[1], roots[1]]
        # Each node is its own span, not the root's.
        assert len({s[2] for s in seen}) == 4
        get_client().shutdown()
