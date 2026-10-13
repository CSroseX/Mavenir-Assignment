"""Small LLM-call seam for the Phase 4 decision-nodes.

triage and grade_sufficiency each make one cheap, structured call. They go
through the same client/model as generate and verify (nodes/core.py) —
looked up at call time, so tests that monkeypatch
`gen_nodes.client.chat.completions.create` intercept these calls too — and
each node also accepts an injected `llm` callable with this signature, so
unit tests never need a client at all.
"""

from __future__ import annotations

import re
from typing import Dict, List

from openai import APIConnectionError, APIStatusError, APITimeoutError

# The failures a decision-node treats as "LLM unavailable" and degrades
# from (to its numeric/no-LLM fallback) instead of crashing the graph.
TRANSPORT_ERRORS = (APIStatusError, APIConnectionError, APITimeoutError)


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """One chat call on the shared client; returns the visible text with any
    <think> block stripped ("" if the model returned no content — e.g. a
    reasoning model that spent max_tokens before answering)."""
    from src.generation.nodes import core

    kwargs = {
        "model": core.MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = core.client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
