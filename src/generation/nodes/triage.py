"""triage_node: route each question to retrieve (in domain) or refuse.

The baseline spent 37.3% of all tokens answering questions that should
have been refused ("What is the capital of France?" -> "Paris"). Triage
removes that spend: an out-of-domain question ends at refuse with zero
generate calls. Three tiers, cheapest first; each either decides or hands
the question to the next:

1. Lexical/structural, no model — a spec id (TS 23.501), a clause reference
   that exists in the corpus, a corpus acronym (AMF, 5GMM), an acronym long
   form, or a clause-title bigram ("session establishment"). All from the
   corpus-derived lexicon (src/generation/lexicon.py), no hand-written
   keyword list. Tier 1 only ever *accepts*: a miss proves nothing.
2. Embedding centroid, no LLM — max cosine between the query and the
   per-spec clause centroids (src/generation/centroids.py), using the same
   BGE model the retriever already loaded. Below `triage_refuse_below`:
   refuse. At or above `triage_accept_above`: accept. Between: ambiguous.
3. LLM, only on the ambiguous band — one yes/no, `triage_llm_max_tokens`.

Failure policy is fail-open: if the centroids or the LLM are unavailable the
question goes to retrieve, where grade_sufficiency still guards generation.
A wrongly refused 3GPP question is a worse failure than a few wasted tokens.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from src.config import get_settings
from src.generation.lexicon import DomainLexicon, get_lexicon
from src.generation.nodes.llm import TRANSPORT_ERRORS, chat_completion
from src.obs.logging import get_logger

log = get_logger(__name__)

_SPEC_REF_RE = re.compile(r"(?i)\b(?:TS|TR|3GPP)\s*(\d{2}\.\d{3})\b|\b(\d{2}\.\d{3})\b")
_CLAUSE_REF_RE = re.compile(r"(?i)\b(?:clause|section|subclause|sec\.?)\s+(\d+(?:\.\d+[A-Za-z]?)+)")

REFUSAL_ANSWER = (
    "This question is out of scope: I can only answer questions about the 3GPP specifications "
    "TS 23.501 (5G system architecture), TS 24.501 (5G NAS) and TS 24.301 (EPS NAS)."
)

_TRIAGE_PROMPT = (
    "Is the following question about mobile telecommunication networks as standardised by 3GPP "
    "(for example 4G/EPS or 5G core network architecture, NAS signalling, mobility or session "
    "management)? Answer with exactly one word: yes or no.\n\nQuestion: {query}"
)


def lexical_signals(query: str, lex: DomainLexicon) -> Dict[str, List[str]]:
    """Tier-1 evidence that the query is in domain; empty lists mean none."""
    spec_refs = []
    for m in _SPEC_REF_RE.finditer(query):
        prefixed, bare = m.group(1), m.group(2)
        # "TS 29.500" is any 3GPP spec; a bare "23.501" must be one of ours.
        if prefixed or bare in lex.spec_ids:
            spec_refs.append(prefixed or bare)
    clause_refs = [c for c in _CLAUSE_REF_RE.findall(query) if c.rstrip(".") in lex.clause_ids]
    return {
        "spec_refs": spec_refs,
        "clause_refs": clause_refs,
        "acronyms": lex.find_acronyms(query),
        "long_forms": [phrase for phrase, _ in lex.find_long_forms(query)],
        "title_bigrams": lex.find_title_bigrams(query),
    }


def parse_yes_no(raw: str) -> Optional[bool]:
    word = re.match(r"\W*(yes|no)\b", raw.strip().lower())
    return None if word is None else word.group(1) == "yes"


def make_triage_node(
    embed: Optional[Callable] = None,
    centroids_loader: Optional[Callable] = None,
    llm: Optional[Callable[..., str]] = None,
    lexicon: Optional[DomainLexicon] = None,
):
    """Return a triage_node.

    embed: texts -> (n, d) array, the retriever's BGE model in the live graph.
    centroids_loader: embed -> Centroids | None; defaults to the cached
        data/domain_centroids.npz (built on first use if missing).
    llm: tier-3 callable; defaults to the shared client.
    All are resolved lazily on first use and cached in the closure.
    """
    call_llm = llm or chat_completion
    cache: Dict[str, Any] = {}

    def _centroids():
        if "centroids" not in cache:
            if centroids_loader is not None:
                loader = centroids_loader
            else:
                from src.generation.centroids import get_centroids as loader
            try:
                cache["centroids"] = loader(embed)
            except Exception as e:  # tier 2 is optional; never crash triage on it
                log.warning("triage_node.centroids_unavailable", error=str(e))
                cache["centroids"] = None
        return cache["centroids"]

    def _tier2_score(query: str) -> Optional[float]:
        if embed is None:
            return None
        centroids = _centroids()
        if centroids is None:
            return None
        try:
            vector = embed([query])[0]
        except Exception as e:
            log.warning("triage_node.embed_failed", error=str(e))
            return None
        score, spec_id, clause = centroids.max_similarity(vector)
        log.debug("triage_node.nearest_centroid", score=score, spec_id=spec_id, clause=clause)
        return score

    def triage_node(state: Dict[str, Any]) -> Dict[str, Any]:
        cfg = get_settings()
        query = state["query"]
        lex = lexicon or get_lexicon()

        signals = lexical_signals(query, lex)
        if any(signals.values()):
            decision, tier, score = "in_domain", "lexical", None
        else:
            score = _tier2_score(query)
            if score is not None and score < cfg.triage_refuse_below:
                decision, tier = "out_of_domain", "centroid"
            elif score is not None and score >= cfg.triage_accept_above:
                decision, tier = "in_domain", "centroid"
            else:
                # Ambiguous band, or tier 2 unavailable: one short yes/no.
                try:
                    raw = call_llm(
                        [{"role": "user", "content": _TRIAGE_PROMPT.format(query=query)}],
                        max_tokens=cfg.triage_llm_max_tokens,
                        temperature=0.0,
                    )
                    verdict = parse_yes_no(raw)
                except TRANSPORT_ERRORS as e:
                    log.warning("triage_node.llm_unavailable", error=str(e))
                    verdict = None
                if verdict is None:
                    decision, tier = "in_domain", "fail_open"
                else:
                    decision, tier = ("in_domain" if verdict else "out_of_domain"), "llm"

        log.info("triage_node.done", query=query, decision=decision, tier=tier, score=score)
        return {
            "triage_decision": decision,
            "triage_tier": tier,
            "triage_score": score,
            "triage_signals": {k: v for k, v in signals.items() if v},
            "termination_reason": "out_of_domain" if decision == "out_of_domain" else None,
        }

    return triage_node


def refuse_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Terminal node for out-of-domain questions: a fixed refusal, no
    retrieval and no LLM call."""
    return {
        "answer": REFUSAL_ANSWER,
        "chunks": [],
        "verification_passed": False,
        "termination_reason": "out_of_domain",
        "abstention": {
            "found_clauses": [],
            "unsupported_claims": [],
            "termination_reason": "out_of_domain",
        },
    }
