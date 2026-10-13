"""grade_sufficiency_node: decide, before generating, whether the retrieved
context can answer the question at all.

The baseline's dominant failure (17/18 ungrounded on first pass) was the
generate/verify loop regenerating against context that never contained the
answer — retries that cannot succeed. This node routes that case to one
query rewrite, then to an honest abstention, instead of 3 regenerations.

Two signals, cheapest first:
1. The cross-encoder logits retrieval already computed (`cross_score`,
   falling back to `score`). If even the top-1 chunk is below
   `sufficiency_min_top1_score`, nothing is relevant and no LLM call is made.
2. One cheap LLM call doing *binary* per-chunk relevance over the top-N
   ({"relevant": [true, false, ...]}) — far more reliable from small models
   than a 1-5 score. If that call fails (transport error, malformed JSON),
   the node degrades to counting chunks whose cross-encoder score clears
   `sufficiency_fallback_min_score`, rather than failing the question.

The count of relevant chunks is what after_grade routes on. When the
context is sufficient, only the chunks judged relevant are passed on to
generate/verify (`sufficiency_filter_chunks`) — prompt tokens are 66% of
spend, and fewer, better chunks is the plan's stated budget rule.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from src.config import get_settings
from src.generation.nodes.llm import TRANSPORT_ERRORS, chat_completion
from src.obs.logging import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You judge whether passages from 3GPP specifications are relevant to a question. "
    "A passage is relevant only if it contains information that directly helps answer the question; "
    "sharing vocabulary with the question is not enough. Do not use outside knowledge."
)


def cross_score(chunk: Dict[str, Any]) -> Optional[float]:
    """Raw cross-encoder logit for a chunk. Prefer `cross_score` over the
    final ranking `score`, which may include a lineage boost, so thresholds
    stay calibrated if that boost is turned on."""
    value = chunk.get("cross_score", chunk.get("score"))
    return float(value) if value is not None else None


def _passage_text(chunk: Dict[str, Any], cap: int) -> str:
    content = chunk.get("content", "")
    text = content if isinstance(content, str) else str(content)
    return text if len(text) <= cap else text[:cap] + " ..."


def build_relevance_messages(query: str, chunks: List[Dict[str, Any]], cap: int) -> List[Dict[str, str]]:
    passages = "\n\n".join(
        f"[{i}] (Spec {c.get('spec_id')}, clause {c.get('clause_id')})\n{_passage_text(c, cap)}"
        for i, c in enumerate(chunks, start=1)
    )
    user = (
        f"<QUESTION>\n{query}\n</QUESTION>\n\n<PASSAGES>\n{passages}\n</PASSAGES>\n\n"
        f'Return JSON only: {{"relevant": [...]}} with exactly {len(chunks)} booleans, '
        "one per passage, in order."
    )
    return [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_relevance(raw: str, n: int) -> List[bool]:
    """Parse the judge's JSON. Raises ValueError on anything but exactly n
    booleans (json.JSONDecodeError is a ValueError)."""
    payload = json.loads(raw)
    flags = payload.get("relevant") if isinstance(payload, dict) else None
    if not isinstance(flags, list) or len(flags) != n or not all(isinstance(f, bool) for f in flags):
        raise ValueError(f"expected {n} booleans under 'relevant', got {raw[:200]!r}")
    return flags


def make_grade_sufficiency_node(llm: Optional[Callable[..., str]] = None):
    """Return a grade_sufficiency_node; `llm` defaults to the shared client
    (nodes/llm.chat_completion) and is injectable for tests."""
    call_llm = llm or chat_completion

    def grade_sufficiency_node(state: Dict[str, Any]) -> Dict[str, Any]:
        cfg = get_settings()
        query = state["query"]
        top = (state.get("chunks") or [])[: cfg.sufficiency_top_n]
        scores = [cross_score(c) for c in top]
        known = [s for s in scores if s is not None]
        top1 = max(known) if known else None

        if not top:
            flags: List[bool] = []
            signal = "no_chunks"
        elif top1 is not None and top1 < cfg.sufficiency_min_top1_score:
            flags = [False] * len(top)
            signal = "cross_encoder_floor"
        else:
            try:
                raw = call_llm(
                    build_relevance_messages(query, top, cfg.sufficiency_chunk_char_cap),
                    max_tokens=cfg.sufficiency_llm_max_tokens,
                    temperature=0.0,
                    json_mode=True,
                )
                flags = parse_relevance(raw, len(top))
                signal = "llm"
            except (*TRANSPORT_ERRORS, ValueError) as e:
                log.warning("grade_sufficiency_node.llm_unavailable_fallback", error=str(e))
                flags = [s is not None and s >= cfg.sufficiency_fallback_min_score for s in scores]
                signal = "cross_encoder_fallback"

        n_relevant = sum(flags)
        sufficient = n_relevant >= cfg.sufficiency_min_relevant

        log.info(
            "grade_sufficiency_node.done",
            n_relevant=n_relevant,
            top1_cross_score=top1,
            signal=signal,
            sufficient=sufficient,
        )

        result: Dict[str, Any] = {
            "sufficiency": "sufficient" if sufficient else "insufficient",
            "n_relevant_chunks": n_relevant,
            "relevance_flags": flags,
            "top1_cross_score": top1,
            "sufficiency_signal": signal,
            # Cleared on success so a sufficient grade after a rewrite doesn't
            # inherit the first pass's insufficient_context.
            "termination_reason": None if sufficient else "insufficient_context",
        }
        if sufficient and cfg.sufficiency_filter_chunks:
            result["chunks"] = [c for c, keep in zip(top, flags) if keep]
        return result

    return grade_sufficiency_node


grade_sufficiency_node = make_grade_sufficiency_node()
