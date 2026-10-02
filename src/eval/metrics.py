"""Retrieval scoring: recall@k, MRR, nDCG@10 — strict and prefix-credit.

Promoted from eval/baseline/run_baseline.py's pair_in_top/normalize_pair,
which used exact (spec_id, clause_id) equality. That undercounts: some gold
clauses (e.g. 24.501/5.1.3.2, 23.501/5.15.2) exist in the corpus only as
descendants (5.1.3.2.1.1, 5.15.2.1, ...), so retrieving the correct child
scored as a total miss. Prefix-credit matching fixes that by also crediting
a retrieved clause that is a strict descendant of a gold clause.

Both variants are reported side by side so that any future retrieval-tuning
delta (Phase 5) is attributable to retrieval work, not to this scorer fix.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, TypedDict


class GoldPair(TypedDict):
    spec_id: str
    clause_id: str


class RetrievedItem(TypedDict, total=False):
    spec_id: str
    clause_id: str
    score: float


def normalize_pair(pair: dict) -> tuple[str, str]:
    return (pair.get("spec_id"), pair.get("clause_id"))


def is_prefix_match(gold_clause_id: str, retrieved_clause_id: str) -> bool:
    """True if retrieved_clause_id is gold_clause_id itself or a strict
    descendant of it (e.g. gold "5.15.2" matches retrieved "5.15.2.1")."""
    if retrieved_clause_id == gold_clause_id:
        return True
    return retrieved_clause_id.startswith(gold_clause_id + ".")


def _hit_rank(
    results: Sequence[RetrievedItem],
    gold_pair: tuple[str, str],
    strict: bool,
) -> int | None:
    """1-indexed rank of the first result matching gold_pair, or None."""
    gold_spec, gold_clause = gold_pair
    for rank, item in enumerate(results, start=1):
        if item.get("spec_id") != gold_spec:
            continue
        retrieved_clause = item.get("clause_id")
        if retrieved_clause is None:
            continue
        if strict:
            if retrieved_clause == gold_clause:
                return rank
        else:
            if is_prefix_match(gold_clause, retrieved_clause):
                return rank
    return None


def recall_at_k(
    results: Sequence[RetrievedItem],
    gold_pairs: Sequence[tuple[str, str]],
    k: int,
    strict: bool,
) -> float:
    """Fraction of gold_pairs covered by at least one hit in results[:k]."""
    if not gold_pairs:
        return 0.0
    window = results[:k]
    hits = 0
    for gold_pair in gold_pairs:
        rank = _hit_rank(window, gold_pair, strict)
        if rank is not None:
            hits += 1
    return hits / len(gold_pairs)


def mrr(
    results: Sequence[RetrievedItem],
    gold_pairs: Sequence[tuple[str, str]],
    strict: bool,
) -> float:
    """Reciprocal rank of the first result matching ANY gold pair."""
    best_rank = None
    for gold_pair in gold_pairs:
        rank = _hit_rank(results, gold_pair, strict)
        if rank is not None and (best_rank is None or rank < best_rank):
            best_rank = rank
    return 1.0 / best_rank if best_rank else 0.0


def ndcg_at_k(
    results: Sequence[RetrievedItem],
    gold_pairs: Sequence[tuple[str, str]],
    k: int,
    strict: bool,
) -> float:
    """Binary-relevance nDCG@k: a result is relevant if it matches any gold
    pair. Multiple gold pairs do not inflate per-rank relevance past 1.0."""
    if not gold_pairs:
        return 0.0
    window = results[:k]

    dcg = 0.0
    for rank, item in enumerate(window, start=1):
        relevant = any(
            (
                item.get("spec_id") == gp[0]
                and (
                    item.get("clause_id") == gp[1]
                    if strict
                    else is_prefix_match(gp[1], item.get("clause_id", ""))
                )
            )
            for gp in gold_pairs
        )
        if relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gold_pairs), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_question(
    results: Sequence[RetrievedItem],
    gold_pairs: Iterable[dict],
    k5: int = 5,
    k20: int = 20,
    k_ndcg: int = 10,
) -> dict:
    """Compute strict and prefix-credit metrics for one question."""
    pairs = [normalize_pair(p) for p in gold_pairs]
    out = {}
    for variant, strict in (("strict", True), ("prefix_credit", False)):
        out[variant] = {
            f"recall_at_{k5}": recall_at_k(results, pairs, k5, strict),
            f"recall_at_{k20}": recall_at_k(results, pairs, k20, strict),
            "mrr": mrr(results, pairs, strict),
            f"ndcg_at_{k_ndcg}": ndcg_at_k(results, pairs, k_ndcg, strict),
        }
    return out
