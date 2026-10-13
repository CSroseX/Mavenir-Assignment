"""Single LLM client factory, optional Langfuse tracing, and token usage
accounting for every chat completion in the live path.

Tracing is optional and non-fatal: with LANGFUSE_HOST unset (or langfuse
not installed) make_llm_client() returns the plain openai.OpenAI client
and observe() is a no-op decorator. Importing langfuse.openai patches the
openai module in place, so it is only imported when tracing is enabled.

Token usage is recorded independently of Langfuse: every client returned
by make_llm_client() appends each response's usage to a process-wide log
(usage_snapshot() / usage_since()). The offline eval gate (RunRecorder)
reads that log, so it never depends on a tracing backend being up.
"""

from __future__ import annotations

import functools
import os
import threading
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from src.obs.logging import get_logger

log = get_logger(__name__)

_usage_lock = threading.Lock()
_usage_log: List[Dict[str, Any]] = []


def tracing_enabled() -> bool:
    if not os.environ.get("LANGFUSE_HOST"):
        return False
    try:
        import langfuse  # noqa: F401
    except ImportError:
        log.warning("tracing.langfuse_not_installed", hint='pip install -e ".[obs]"')
        return False
    return True


def _record_usage(response: Any, model: Optional[str]) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    with _usage_lock:
        _usage_log.append({
            "model": model,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", None) or 0,
            "total_tokens": getattr(usage, "total_tokens", None) or 0,
        })


def _wrap_usage(client: Any) -> Any:
    """Route client.chat.completions.create through _record_usage. The
    completions resource is cached on the client, so patching the instance
    attribute once covers every later call."""
    completions = client.chat.completions
    original_create = completions.create

    @functools.wraps(original_create)
    def create(*args, **kwargs):
        response = original_create(*args, **kwargs)
        _record_usage(response, kwargs.get("model"))
        return response

    completions.create = create
    return client


def make_llm_client(api_key: str, base_url: str) -> Any:
    """Return an OpenAI-compatible client: Langfuse-traced when LANGFUSE_HOST
    is set, the plain openai client otherwise. Usage is recorded either way."""
    if tracing_enabled():
        from langfuse.openai import OpenAI
        log.info("tracing.enabled", host=os.environ.get("LANGFUSE_HOST"))
    else:
        from openai import OpenAI
    return _wrap_usage(OpenAI(api_key=api_key, base_url=base_url))


def observe(func: Optional[Callable] = None, **observe_kwargs: Any) -> Callable:
    """langfuse.observe when tracing is enabled, otherwise a pass-through.
    Usable bare (@observe) or with arguments (@observe(name="generate")).

    The choice is made on each call, not at decoration time: .env is loaded
    by src.generation.nodes.core at import, so a node module decorated
    before that import would otherwise be stuck untraced."""

    def decorate(fn: Callable) -> Callable:
        traced: List[Callable] = []

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not tracing_enabled():
                return fn(*args, **kwargs)
            if not traced:
                from langfuse import observe as lf_observe
                traced.append(lf_observe(fn, **observe_kwargs))
            return traced[0](*args, **kwargs)

        return wrapper

    return decorate(func) if func is not None else decorate


def update_trace_attributes(**attributes: Any) -> None:
    """Attach attributes to the current Langfuse span; a no-op without
    tracing, and never raises — tracing must not break a query."""
    if not tracing_enabled():
        return
    try:
        from langfuse import get_client
        get_client().update_current_span(metadata=attributes)
    except Exception as e:  # tracing backend down or no active span
        log.debug("tracing.update_failed", error=str(e))


@contextmanager
def trace_query(query: str) -> Iterator[None]:
    """Root span for one question, so every node span nests under it and one
    query is one trace. A no-op without tracing; a tracing backend that
    fails to start a span never fails the query."""
    with ExitStack() as stack:
        if tracing_enabled():
            try:
                from langfuse import get_client
                stack.enter_context(get_client().start_as_current_observation(
                    name="rag_query", as_type="chain", input={"query": query},
                ))
            except Exception as e:
                log.debug("tracing.trace_query_failed", error=str(e))
        yield


def trace_attributes(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The decision signals recorded on node spans and on the query's root
    span: termination_reason, attempt, top1_cross_score, n_relevant_chunks,
    triage_decision. Keys whose value is not set yet are left out."""
    chunks = state.get("chunks") or []
    top1 = chunks[0].get("cross_score", chunks[0].get("score")) if chunks else None
    attributes = {
        "termination_reason": state.get("termination_reason"),
        "attempt": state.get("attempt"),
        "top1_cross_score": top1,
        "n_relevant_chunks": state.get("n_relevant_chunks"),
        "triage_decision": state.get("triage_decision"),
    }
    return {k: v for k, v in attributes.items() if v is not None}


def traced_node(name: str, node: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable:
    """Wrap a graph node in an observe() span named after it, recording the
    decision signals as of the node's output. LangGraph reads the node's
    signature, so the wrapper keeps the single `state` parameter."""

    @observe(name=name)
    def traced(state: Dict[str, Any]) -> Dict[str, Any]:
        update = node(state)
        if tracing_enabled():
            update_trace_attributes(**trace_attributes({**state, **(update or {})}))
        return update

    return traced


def usage_snapshot() -> int:
    """Opaque cursor into the usage log; pass it to usage_since()."""
    with _usage_lock:
        return len(_usage_log)


def usage_since(cursor: int) -> List[Dict[str, Any]]:
    with _usage_lock:
        return list(_usage_log[cursor:])


def reset_usage() -> None:
    with _usage_lock:
        _usage_log.clear()
