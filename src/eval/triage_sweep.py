"""Zero-LLM threshold sweep for triage tier 2 (embedding centroids).

Scores every golden-set question against the domain centroids with the
retriever's BGE model, sweeps the refuse threshold, and recommends the
(triage_refuse_below, triage_accept_above) pair:

- refuse_below  = min(lo, hi)  and  accept_above = max(lo, hi), where
  lo = lowest in-domain score - margin  (nothing in domain is refused by
       tier 2 alone, even with tier 1 switched off), and
  hi = highest out-of-scope score + margin  (nothing out of scope is
       accepted by tier 2 alone).
  If the classes separate, [hi, lo) is an empty-on-the-golden-set safety
  band; if they overlap, the overlap goes to the tier-3 LLM.

In-domain = `normal` + `adversarial`; the hard refusal gate is the
`out_of_scope` questions minus `borderline` ones (Q15, TCP/UDP), which are
reported but excluded. Tier-1 (lexical) decisions are reported too: they
need no model, and an out-of-scope question accepted lexically would bypass
tier 2 entirely.

Writes eval/triage/sweep.json and eval/triage/sweep.svg. Spends zero LLM
tokens; needs the BGE model (downloaded by fastembed on first use).

    python -m src.eval.triage_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import get_settings  # noqa: E402
from src.generation.lexicon import get_lexicon  # noqa: E402
from src.generation.nodes.triage import lexical_signals  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_SET_PATH = ROOT / "eval" / "golden" / "golden_set.v1.json"
OUT_DIR = ROOT / "eval" / "triage"


def gate_of(item: Dict[str, Any]) -> str:
    if item.get("borderline"):
        return "excluded"
    return "out" if item["type"] == "out_of_scope" else "in"


def tier1_rows(golden: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lex = get_lexicon()
    rows = []
    for item in golden:
        signals = {k: v for k, v in lexical_signals(item["question"], lex).items() if v}
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "type": item["type"],
            "gate": gate_of(item),
            "tier1_hit": bool(signals),
            "tier1_signals": signals,
        })
    return rows


def recommend(rows: List[Dict[str, Any]], margin: float) -> Dict[str, Any]:
    in_scores = [r["score"] for r in rows if r["gate"] == "in"]
    out_scores = [r["score"] for r in rows if r["gate"] == "out"]
    lo = min(in_scores) - margin
    hi = max(out_scores) + margin
    return {
        "triage_refuse_below": round(min(lo, hi), 4),
        "triage_accept_above": round(max(lo, hi), 4),
        "separable": max(out_scores) < min(in_scores),
        "min_in_domain_score": round(min(in_scores), 4),
        "max_out_of_scope_score": round(max(out_scores), 4),
        "margin": margin,
    }


def sweep(rows: List[Dict[str, Any]], start: float = 0.30, stop: float = 0.95, step: float = 0.005):
    n_steps = int(round((stop - start) / step)) + 1
    out = []
    for i in range(n_steps):
        t = round(start + i * step, 4)
        out.append({
            "threshold": t,
            "in_domain_refused_tier2_only": sum(1 for r in rows if r["gate"] == "in" and r["score"] < t),
            "in_domain_refused_with_tier1": sum(
                1 for r in rows if r["gate"] == "in" and not r["tier1_hit"] and r["score"] < t
            ),
            "out_of_scope_refused": sum(
                1 for r in rows if r["gate"] == "out" and not r["tier1_hit"] and r["score"] < t
            ),
        })
    return out


def simulate(rows: List[Dict[str, Any]], refuse_below: float, accept_above: float) -> List[Dict[str, Any]]:
    """Zero-LLM end-to-end triage decision per question; 'llm' marks the
    questions that would cost one tier-3 call."""
    for r in rows:
        if r["tier1_hit"]:
            r["decision"] = "retrieve (tier 1)"
        elif r["score"] < refuse_below:
            r["decision"] = "refuse (tier 2)"
        elif r["score"] >= accept_above:
            r["decision"] = "retrieve (tier 2)"
        else:
            r["decision"] = "llm (tier 3)"
    return rows


# ── plot ───────────────────────────────────────────────────────────────
# Static SVG, no plotting dependency. Categorical slots 1 and 2 of the
# default dataviz palette; text in neutral ink, never the series colour.
_IN, _OUT, _BORDER = "#2a78d6", "#eb6834", "#8a8a8a"
_INK, _MUTED, _GRID, _BG = "#1f1f1f", "#5c5c5c", "#e4e4e4", "#ffffff"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_svg(rows, curve, rec) -> str:
    W, H = 760, 520
    left, right = 190, 30
    scores = [r["score"] for r in rows]
    x0 = max(0.0, min(scores + [rec["triage_refuse_below"]]) - 0.05)
    x1 = min(1.0, max(scores + [rec["triage_accept_above"]]) + 0.05)

    def x(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * (W - left - right)

    lo, hi = rec["triage_refuse_below"], rec["triage_accept_above"]
    out: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
        f'<text x="20" y="28" font-size="15" font-weight="600" fill="{_INK}">'
        "Triage tier 2: max centroid cosine per golden-set question</text>",
        f'<text x="20" y="46" fill="{_MUTED}">Shaded band goes to the tier-3 LLM; '
        f"refuse below {lo:.3f}, accept at or above {hi:.3f}.</text>",
    ]

    # Panel A: dot strip.
    top_a, row_h = 70, 34
    lanes: List[Tuple[str, str, str]] = [
        ("in", "In domain (normal + adversarial)", _IN),
        ("out", "Out of scope (hard gate)", _OUT),
        ("excluded", "Borderline (excluded)", _BORDER),
    ]
    a_bottom = top_a + row_h * len(lanes)
    out.append(f'<rect x="{x(lo):.1f}" y="{top_a}" width="{max(x(hi) - x(lo), 0):.1f}" '
               f'height="{a_bottom - top_a}" fill="#f2f2f2"/>')
    for i, (gate, label, color) in enumerate(lanes):
        cy = top_a + row_h * i + row_h / 2
        out.append(f'<line x1="{left}" x2="{W - right}" y1="{cy}" y2="{cy}" stroke="{_GRID}"/>')
        out.append(f'<text x="{left - 10}" y="{cy + 4}" text-anchor="end" fill="{_INK}">{_esc(label)}</text>')
        for r in rows:
            if r["gate"] != gate:
                continue
            title = _esc(f"Q{r['id']} ({r['score']:.3f}): {r['question']}")
            out.append(f'<circle cx="{x(r["score"]):.1f}" cy="{cy}" r="5" fill="{color}" '
                       f'stroke="{_BG}" stroke-width="2"><title>{title}</title></circle>')
    for v, name in ((lo, "refuse below"), (hi, "accept above")):
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top_a - 6}" y2="{a_bottom}" '
                   f'stroke="{_MUTED}" stroke-dasharray="4 3"/>')
        out.append(f'<text x="{x(v):.1f}" y="{top_a - 10}" text-anchor="middle" fill="{_MUTED}">'
                   f"{name} {v:.3f}</text>")

    # Panel B: refusal counts as the refuse threshold sweeps.
    top_b, bottom_b = a_bottom + 60, H - 60
    n_in = sum(1 for r in rows if r["gate"] == "in")
    n_out = sum(1 for r in rows if r["gate"] == "out")
    y_max = max(n_in, n_out, 1)

    def y(v: float) -> float:
        return bottom_b - v / y_max * (bottom_b - top_b)

    out.append(f'<text x="20" y="{top_b - 18}" font-weight="600" fill="{_INK}">'
               "Questions refused by tier 2 alone, as the refuse threshold sweeps</text>")
    for tick in range(0, y_max + 1, max(1, y_max // 4)):
        out.append(f'<line x1="{left}" x2="{W - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{_GRID}"/>')
        out.append(f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="{_MUTED}">{tick}</text>')
    pts = [c for c in curve if x0 <= c["threshold"] <= x1]
    for key, color, label in (
        ("in_domain_refused_tier2_only", _IN, f"In domain wrongly refused (of {n_in})"),
        ("out_of_scope_refused", _OUT, f"Out of scope correctly refused (of {n_out})"),
    ):
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(c['threshold']):.1f},{y(c[key]):.1f}" for i, c in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        last = pts[-1]
        out.append(f'<text x="{x(last["threshold"]) - 4:.1f}" y="{y(last[key]) - 8:.1f}" '
                   f'text-anchor="end" fill="{_INK}">{_esc(label)}</text>')
    for v in (lo, hi):
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top_b}" y2="{bottom_b}" '
                   f'stroke="{_MUTED}" stroke-dasharray="4 3"/>')
    # Shared x axis.
    tick = x0
    step = 0.05
    while tick <= x1 + 1e-9:
        out.append(f'<text x="{x(tick):.1f}" y="{bottom_b + 18}" text-anchor="middle" fill="{_MUTED}">{tick:.2f}</text>')
        tick = round(tick + step, 4)
    out.append(f'<text x="{(left + W - right) / 2}" y="{bottom_b + 40}" text-anchor="middle" fill="{_INK}">'
               "max cosine to a domain centroid / refuse threshold</text>")
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-LLM triage threshold sweep over the golden set.")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    cfg = get_settings()

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    rows = tier1_rows(golden)

    print("Tier 1 (lexical, no model):")
    for r in rows:
        print(f"  Q{r['id']:>2} {r['gate']:<8} tier1={'HIT ' if r['tier1_hit'] else 'miss'} {r['tier1_signals']}")
    leaks = [r["id"] for r in rows if r["gate"] == "out" and r["tier1_hit"]]
    if leaks:
        print(f"WARNING: out-of-scope questions accepted lexically (bypass tier 2): {leaks}")

    try:
        from src.generation.centroids import get_centroids, load_default_embedder
        embed = load_default_embedder()
        centroids = get_centroids(embed)
    except Exception as e:  # model download blocked, no cache, etc.
        print(f"\nTier 2 sweep NOT run: could not load the embedding model / centroids ({type(e).__name__}: {e}).")
        print("Re-run where the BGE model can be downloaded; nothing was written.")
        sys.exit(2)

    vectors = embed([r["question"] for r in rows])
    for r, v in zip(rows, vectors):
        score, spec_id, clause = centroids.max_similarity(v)
        r["score"] = round(score, 4)
        r["nearest_centroid"] = f"{spec_id}/{clause}"

    curve = sweep(rows)
    rec = recommend(rows, cfg.triage_sweep_margin)
    simulate(rows, rec["triage_refuse_below"], rec["triage_accept_above"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": cfg.embedding_model,
        "n_centroids": int(len(centroids.vectors)),
        "golden_set": str(GOLDEN_SET_PATH.relative_to(ROOT)),
        "recommendation": rec,
        "questions": rows,
        "sweep": curve,
    }
    (out_dir / "sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "sweep.svg").write_text(render_svg(rows, curve, rec), encoding="utf-8")

    print("\nTier 2 scores:")
    for r in sorted(rows, key=lambda r: r["score"]):
        print(f"  {r['score']:.4f}  Q{r['id']:>2} {r['gate']:<8} {r['decision']:<18} {r['question'][:60]}")
    print(f"\nRecommendation: {rec}")
    print(f"Wrote {out_dir / 'sweep.json'} and {out_dir / 'sweep.svg'}")


if __name__ == "__main__":
    main()
