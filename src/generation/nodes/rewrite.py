"""rewrite_query_node: one deterministic, domain-structural query rewrite.

Runs when grade_sufficiency_node finds the retrieved context can't answer
the question. Not a generic LLM paraphrase (that costs tokens and drifts
off the 3GPP vocabulary the index was built from); three structural moves,
all driven by the corpus lexicon (src/generation/lexicon.py):

1. Strip conversational framing ("Can you explain how...", "What is the")
   down to the content-bearing phrase. Spec ids and clause references are
   kept verbatim — the retriever's targeted path keys on them.
2. Expand acronyms in both directions: AMF -> "Access and Mobility
   Management Function", and a long form already in the query gets its
   acronym appended, so dense retrieval sees both surface forms.
3. Split a cross-generation question (an EPS/4G term and a 5G term in the
   same query, e.g. "does the MME talk to the UDM?") into a 4G part and a
   5G part, retrieved separately. The 4G part is pinned to the EPS spec,
   so the 4G side of the comparison is actually in context — the
   cross-spec conflation this corpus was chosen to expose.

Hard-capped at `max_rewrites` (config); past the cap, or when the rewrite
would just reproduce the queries already tried, the node marks itself a
no-op and the graph terminates at abstain instead of re-retrieving.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.config import get_settings
from src.generation.lexicon import DomainLexicon, get_lexicon
from src.obs.logging import get_logger

log = get_logger(__name__)

_FRAMING_PATTERNS = [
    re.compile(r"^(?:hi|hello|hey)\b[\s,!.]*", re.IGNORECASE),
    re.compile(r"^please\s+", re.IGNORECASE),
    re.compile(r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?", re.IGNORECASE),
    re.compile(r"^i\s+(?:want|would\s+like|'d\s+like|need)\s+to\s+(?:know|understand)\s+", re.IGNORECASE),
    re.compile(
        r"^(?:tell\s+me(?:\s+about)?|explain(?:\s+to\s+me)?|describe|summari[sz]e|define|clarify)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:what|which|who|how|why|when|where)(?:'s|\s+(?:is|are|was|were|does|do|did|can|could|"
        r"should|would|will|shall|has|have))?\s+",
        re.IGNORECASE,
    ),
]
_TRAILING_RE = re.compile(r"(?:[\s,]+please)?[\s?.!]*$", re.IGNORECASE)
_SPEC_RE = re.compile(r"\b\d{2}\.\d{3}\b")


def strip_framing(query: str) -> str:
    """Peel conversational framing off the front of the query. Applies the
    patterns repeatedly ("Can you please explain what the ..." needs three
    passes); falls back to the original if nothing would be left."""
    text = query.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _FRAMING_PATTERNS:
            new = pattern.sub("", text, count=1)
            if new != text and new.strip():
                text = new
                changed = True
    text = _TRAILING_RE.sub("", text).strip()
    return text or query.strip()


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;:])", r"\1", text)
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
    return text.strip(" ,;:")


def _expansions(text: str, terms: List[str], lex: DomainLexicon) -> List[str]:
    """Surface forms to append for `terms` (acronyms): the long form when
    the text only has the acronym, the acronym when it only has the long form."""
    lower = text.lower()
    out: List[str] = []
    for term in terms:
        long_form = lex.long_forms.get(term)
        if long_form and long_form.lower() not in lower and long_form not in out:
            out.append(long_form)
        if not re.search(r"(?<![A-Za-z0-9-])" + re.escape(term) + r"(?![A-Za-z0-9])", text) and term not in out:
            out.append(term)
    return out


def _remove_terms(text: str, terms: List[str]) -> str:
    for term in terms:
        text = re.sub(r"(?<![A-Za-z0-9-])" + re.escape(term) + r"(?![A-Za-z0-9-])", "", text)
    return _tidy(text)


def rewrite_query(query: str, lex: Optional[DomainLexicon] = None) -> List[str]:
    """Deterministic rewrite: returns one query, or two (4G part, 5G part)
    for a cross-generation question."""
    cfg = get_settings()
    lex = lex or get_lexicon()

    core = strip_framing(query)
    terms = lex.find_acronyms(core)
    for _, acronym in lex.find_long_forms(core):
        if acronym not in terms:
            terms.append(acronym)
    # "TS" in "TS 24.501" is a document-type prefix, not a concept to expand.
    terms = [t for t in terms if not re.search(re.escape(t) + r"\s*\d{2}\.\d{3}", core)]

    terms_4g = [t for t in terms if lex.generation_of(t) == "4g"]
    terms_5g = [t for t in terms if lex.generation_of(t) == "5g"]

    if terms_4g and terms_5g:
        part_4g = _remove_terms(core, terms_5g)
        part_4g = _tidy(" ".join([part_4g] + _expansions(part_4g, terms_4g, lex)))
        # Pin the 4G part to the EPS spec when there is exactly one and the
        # user didn't already name a spec: "TS 24.301" puts the retriever
        # on its targeted (payload-filtered) path.
        if len(cfg.rewrite_4g_spec_ids) == 1 and not _SPEC_RE.search(query):
            part_4g = f"{part_4g} TS {cfg.rewrite_4g_spec_ids[0]}"

        part_5g = _remove_terms(core, terms_4g)
        part_5g = _tidy(" ".join([part_5g] + _expansions(part_5g, terms_5g, lex)))
        return [part_4g, part_5g]

    return [_tidy(" ".join([core] + _expansions(core, terms, lex)))]


def rewrite_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node: rewrite the query for one more retrieval pass.

    Writes `retrieval_queries` (consumed by retrieve_node; `query` itself
    stays the user's original question for generate/verify) and bumps
    `rewrites`. Sets `rewrite_noop` when the cap is already spent or the
    rewrite reproduces the queries already tried — after_rewrite routes
    that straight to abstain rather than paying for an identical retrieval.
    """
    cfg = get_settings()
    rewrites = state.get("rewrites") or 0
    if rewrites >= cfg.max_rewrites:
        log.info("rewrite_query_node.cap_reached", rewrites=rewrites, max_rewrites=cfg.max_rewrites)
        return {"rewrite_noop": True}

    previous = state.get("retrieval_queries") or [state["query"]]
    queries = rewrite_query(state["query"])
    noop = [q.strip().lower() for q in queries] == [q.strip().lower() for q in previous]

    log.info("rewrite_query_node.done", original=state["query"], rewritten=queries, noop=noop)
    return {
        "retrieval_queries": queries,
        "rewrites": rewrites + 1,
        "rewrite_noop": noop,
    }
