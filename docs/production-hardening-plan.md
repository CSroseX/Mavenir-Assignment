# Production-Hardening the 3GPP RAG Assistant

## Context

This repo answers 3GPP telecom questions (TS 23.501 / 24.501 / 24.301) using a
LangGraph `generate → verify` loop over Qdrant retrieval. The pipeline works, but
it has no production scaffolding: no config module, no logging, no tests, no CI,
no cost accounting, and only one decision point in the graph. It is also,
measurably, not behaving correctly — and nobody had run the full eval set to find
out.

**The measured starting state** (from the frozen `eval/baseline/` run, 18 questions):

| Metric | Value | Meaning |
|---|---|---|
| Verified first-pass | **1/18** | 17 answers could not be grounded in retrieved text |
| Out-of-scope refusals | **0/6** | Answers "What is the capital of France?" with "The capital of France is Paris." |
| Tokens per run | **124,701** | ~6,928/question, 2 LLM calls/question at 1-retry cap |
| **Tokens spent on questions that should have been refused** | **46,452 (37.3%)** | Pure waste |
| Mean recall@5 / @20 / MRR | 0.583 / 0.750 / 0.468 | Measured on only 6 of 18 questions |

Goal: make this a project that demonstrates production RAG engineering — real
evals, real observability, real decision-nodes — while **reducing** token spend,
since the LLM runs on a free-tier key that must not be exhausted.

### Four findings that shape the plan

1. **`retries` is off by one.** `nodes.py:171` increments on every verify call
   including the first, so a first-pass success records `retries: 1`. Every retry
   metric and the "self-corrected" badge is wrong. The stale `eval/results.json`
   (12 rows, all `verified_after_retry`, 100% pass rate) is an artifact of this
   plus an incomplete run that never reached a single `out_of_scope` question.

2. **The verifier cannot distinguish an outage from a hallucination.**
   `nodes.py:163-166` catches bare `Exception` → `verification_passed = False` +
   `feedback = "verification could not complete."` A 429 or timeout therefore
   burns up to 3 regenerations *and* injects that string into the next generate
   prompt as if it were factual critique.

3. **Two gold clauses do not exist as chunks** — verified against the corpus:
   `24.501/5.1.3.2` and `23.501/5.15.2` exist only as descendants
   (`5.15.2.1`, `5.15.2.2`, …). The scorer uses exact `(spec_id, clause_id)`
   equality (`run_baseline.py:43-47`), so retrieving the correct child at rank 2
   scores as a total miss. **Fix the scorer before tuning retrieval.**

4. **Two retrieval failures, two different mechanisms.** Q1 ("role of the AMF"):
   gold `23.501/6.2.1` is in the top-20 but not top-5 (r@20=1.0, r@5=0.0) — a
   pure *reranker* failure. Q6: gold never enters the candidate set at all
   (0.0 at both cutoffs) — a *first-stage recall* failure needing lexical search.
   `recall@20 >> recall@5` across the set says the reranker is the bottleneck.

### Decisions already made

- **All 6 phases.**
- **Langfuse, self-hosted** via docker-compose for tracing.
- **Custom LLM judges, no ragas.** Not a preference — measured: with
  dependencies resolved, pip backs off from ragas 0.4.3 to **ragas 0.0.6** (a
  prototype release) and pulls `protobuf 3.20.0` on this Python 3.14.3 env.
  Modern ragas cannot install here. Documented in the README as an evaluated
  trade-off with evidence. `pydantic-settings 2.15.0` is already installed, so
  the config module needs no new dependency.
- **Keep `eval/baseline/` frozen** as the "before" for the ablation table. The
  bad numbers get *fixed*, not hidden; freezing just keeps the delta legible.

### Token budget — a first-class constraint

Every phase must hold or reduce tokens per eval run. Rules:

- **Phase 4's triage node is the biggest win and is scheduled early for that
  reason** — it eliminates the 37.3% spent on out-of-scope questions.
- Prompt tokens are **66%** of all spend (chunk text dominates, avg 2,276/call).
  So: cap chunk characters injected into prompts, and prefer *fewer, better*
  chunks over more.
- Grading sufficiency *before* generating kills the 17/18 ungroundable-answer
  retry loops, which cannot succeed anyway — the context lacks the answer.
- Retrieval-only metrics (recall/MRR/nDCG) cost **zero** LLM tokens. Gate CI on
  those; never run LLM evals on a PR.
- Judge calls reuse one cheap prompt, run only on `workflow_dispatch`/nightly.
- Add `--limit N` / `--types normal` flags so iteration runs a 3-question subset,
  not all 18.

---

## Phase 1 — Foundations and an honest baseline

No LLM calls in this phase.

- ✅ `src/config.py` — `pydantic-settings` singleton. Absorbs the literals currently
  duplicated across files: `"3gpp_specs"`, `localhost:6333`, `localhost:11434`,
  embed/reranker model names, `top_k` (today 5 in `app.py`/`run_eval.py`, 3 in
  `verify.py`, 20 in baseline), `fetch_k`, retry caps, thresholds, log level,
  eval sleep.
- ✅ `src/obs/logging.py` — structlog; JSON to `logs/app.jsonl`, pretty console in
  dev. Replaces `print()` in `src/`, notably `nodes.py:79-81` (dumps the entire
  system prompt + all chunk text on *every* call) and `retriever.py:116-138`.
  **Update `CLAUDE.md` in the same commit** — it currently documents the verbose
  `[DEBUG]` prints as intentional.
- `pyproject.toml` — replaces unpinned `requirements.txt`; pin from the current
  working env; extras `dev`/`eval`/`obs`; ruff + pytest config. Add the missing
  `docling` (used by `parse.py`, never declared).
- ✅ `src/eval/metrics.py` — promote and **fix** the scorer from
  `run_baseline.py:43-47`: add **prefix-credit matching** so a retrieved
  descendant covers its gold ancestor. Report strict *and* prefix-credit side by
  side so the Phase 5 deltas are attributable to retrieval work, not the scorer
  fix. Add nDCG@10.
- ✅ `eval/golden/golden_set.v1.json` — merge `eval/eval_set.json` (18 Qs, joined by
  question *text*) and `eval/baseline/gold.json` (12 entries, joined by
  `question_id`). One record per question, stable `id`, fields: `question`,
  `type`, `expected_behavior`, `gold_clauses`, `retrieval_scored`,
  `corpus_answerable`, `borderline`, `notes`. Mark Q15 (TCP/UDP) `borderline`
  — the eval set itself says "refusal *or* out of scope unless referenced as
  transport". Versioned filename; never edit a released version in place.
- ✅ `tests/test_golden_set.py` — contract test asserting every `gold_clauses` entry
  resolves to ≥1 chunk (exact or descendant). **This test would have caught
  finding #3 on day one.** Plus `test_metrics.py`, `test_config.py`.
- `app.py` — `@st.cache_resource` on `get_retriever()` / `get_graph()`. Today
  `app.py:75,82` reload two ML models and reparse 2,644 chunks **per message**.
- ✅ `src/retrieval/retriever.py` — anchor the `data/chunks/*_chunks.json` glob
  (`retriever.py:31`) to repo root; a bare relative glob silently returns zero
  from any other cwd.
- ✅ `src/ingestion/indexer.py` — deterministic `chunk_id` (`sha1(spec|clause|ordinal)`)
  replacing per-run `uuid4()`. Snapshot-based CI in Phase 5 depends on this.
- `.github/workflows/ci.yml` job 1: ruff + pytest, no network, under a minute.

**Verify:** `pytest` green; contract test fails on a deliberately bad gold pair;
second chat message is visibly faster; retrieval metrics re-run and committed as
`eval/baselines/retrieval_baseline.json` with both scorer variants.

---

## Phase 2 — Reliability and truthful state

- ✅ `src/generation/state.py` — expand `GraphState`: `triage_decision`,
  `retrieval_scores`, `sufficiency`, `rewrites`, `attempt`,
  `transport_attempts`, `termination_reason`, `citations`.
- ✅ **Fix the off-by-one (finding #1):** `generate_node` owns the `attempt`
  counter; `verify_claims_node` becomes pure w.r.t. counters. First pass is
  `attempt == 1`; "self-corrected" means `attempt > 1 AND verification_passed`.
- ✅ **Fix the swallowed exception (finding #2):** typed boundary, not bare
  `Exception`.
  - `APIStatusError` 429 / `APIConnectionError` / `APITimeoutError` → transport
    failure. Backoff via the SDK's `max_retries` (set explicitly in config). On
    exhaustion: `termination_reason = "verifier_unavailable"`, **no** retry
    increment, **no** feedback injection.
  - `JSONDecodeError`/schema-invalid → one reask, then treat as transport.
  - Valid JSON with `is_supported: false` → the *only* case that increments
    retries and injects feedback.
  - Two counters: `grounding_retries` (semantic, drives routing) and
    `transport_attempts` (operational, observability only).
- ✅ `src/generation/nodes/abstain.py` replaces `flag_unverified_node`, which today
  passes a thrice-failed answer through unchanged and unbadged. Emits a
  structured abstention: what clauses *were* found, what couldn't be
  substantiated, and the `termination_reason` enum
  (`verified` | `verification_failed` | `insufficient_context` |
  `out_of_domain` | `verifier_unavailable`).
- ✅ Banned-phrase leak check becomes a real gate (today `nodes.py:99-103` only
  prints). Tighten to word-boundary regexes — the current bare substrings
  `"retrieved"`/`"chunks"` false-positive on legitimate 3GPP prose about
  retrieving subscription data. One targeted rephrase, then abstain.
- ✅ `src/generation/routers.py` — all conditional-edge functions, pure and
  unit-testable.
- `app.py` — one `render_assistant_message()` switching on `termination_reason`,
  replacing the badge logic currently duplicated and already drifting between
  `app.py:48-59` and `app.py:100-107`. Renders **all** terminal states; today
  `"Flagged Unverified"` is computed and never shown. Drop the substring refusal
  heuristic at `app.py:151`.
- ✅ `src/eval/runner.py` — rewrite of `eval/run_eval.py`. Resume keyed by `id` plus
  a content hash of the golden set (today: positional `start_index`, never
  validates `results[i].question == eval_set[i].question`). Replace
  `except: pass` (`run_eval.py:41-45`, silently restarts from zero then
  overwrites the good file) with `JSONDecodeError`-specific handling and
  **atomic temp-file-then-replace** writes. Add `--limit` / `--types` for cheap
  iteration.
- Delete the stale `eval/results.json`; one clean full 18-question run.

**Verify:** router unit tests over the state cross-product; fault-injection test
patching the client to raise `APIStatusError` 429 asserts
`termination_reason == "verifier_unavailable"`, attempt not incremented, no fake
feedback; a first-pass success records `attempt == 1`; full run yields 18 rows
and reaches all 6 `out_of_scope` questions. Expect the honest pass rate to be
*bad* — that is the Phase 4 setup.

---

## Phase 3 — Observability

- `docker-compose.yml` — Qdrant + Langfuse (+ Postgres/ClickHouse).
- `src/obs/tracing.py` — swap `from openai import OpenAI` for
  `from langfuse.openai import OpenAI` in a single client factory. This alone
  gives per-call tokens, latency, model and cost **in the live path**, not just
  in a monkeypatched baseline script. Then delete the
  `run_baseline.py:132-147` monkeypatch as dead code.
- `@observe()` on each node → one query = one trace with nested spans. Trace
  attributes: `termination_reason`, `attempt`, `top1_cross_score`,
  `n_relevant_chunks`, `triage_decision`.
- **Non-fatal and optional.** With `LANGFUSE_HOST` unset the factory returns the
  plain client and `@observe` no-ops. A portfolio project that crashes without a
  tracing backend is worse than one with no tracing.
- `src/obs/metrics.py` — `RunRecorder` writes self-contained
  `eval/runs/<run_id>/{run.json, per_question.jsonl, summary.json}`. Langfuse is
  for interactive inspection; it must **not** be a dependency of the eval gate.
  Records per-node latency, tokens in/out per call, estimated cost (price table
  in config, clearly marked an estimate), node path, `termination_reason`.
- `app.py` — third **Trace** tab: last query's node path, per-node latency,
  tokens, and decision signals. This is the most demo-visible artifact in the
  project; load the `dataviz` skill when building it. Dashboard reads
  `eval/runs/` instead of the single `results.json`.

**Verify:** one query → one Langfuse trace with the expected span tree and
non-zero tokens; everything still works with `LANGFUSE_HOST` unset;
`summary.json` totals match Langfuse; eval reproducible from disk with only
Qdrant running.

---

## Phase 4 — Decision-nodes

The core architectural work, and the biggest token saving. **Retrieval moves
inside the graph** — today `app.py:76`, `run_eval.py:57` and `verify.py:23` each
call `retriever.search()` before the graph with three different `top_k` values,
which makes retrieve-after-rewrite structurally impossible.

```
START → triage ──out_of_domain──→ refuse → END
          │ in_domain
          ↓
       retrieve → grade_sufficiency ──insufficient──→ rewrite_query → retrieve
          ↑                                               (max 1, then abstain)
          │ sufficient
          ↓
       generate → verify ──passed──→ END
                     │ failed, attempt<MAX → generate
                     │ failed, attempt>=MAX → abstain → END
```

**`triage_node` → {retrieve, refuse}.** Three tiers, cheapest first:
1. Lexical/structural, no LLM — spec id, clause ref, or a term from a
   **corpus-derived** lexicon (harvest `clause_title` tokens + abbreviations
   across all 2,644 chunks; data-derived, not a hand-written keyword list).
2. Embedding-centroid, no LLM — reuse the already-loaded BGE model; cosine
   against per-spec clause centroids cached to `data/domain_centroids.npz`.
3. LLM tier **only** on the ambiguous band: one yes/no, `max_tokens=10`.

Thresholds tuned on the golden set with a committed sweep plot, not guessed.
Q15 (TCP/UDP) excluded from the hard refusal gate as `borderline`.
**This is the 37.3% token saving** — out-of-scope goes from ~7.7k tokens to one
embedding.

**`grade_sufficiency_node` → {generate, rewrite_query, abstain}.** Keys on a
numeric signal already computed and currently discarded (`retriever.py:127`
cross-encoder scores — top-1 and the distribution shape) plus one cheap LLM call
doing *binary per-chunk* relevance over the top-5 (far more reliable from small
models than a 1–5 score). Route on the count: ≥2 relevant → generate; ≤1 with
rewrites left → rewrite; ≤1 exhausted → abstain. **This is the fix for the 17/18
failure mode** — regenerating against context that lacks the answer cannot
succeed, so those retries are pure waste converted into one rewrite or an honest
abstention.

**`rewrite_query_node` → retrieve.** Hard cap 1 (config). Domain-structural, not
generic paraphrase: expand acronyms both directions from the corpus lexicon
(`AMF` ↔ `Access and Mobility Management Function` — likely fixes Q6's NSSAI
miss), strip conversational framing to a noun phrase, split adversarial
cross-generation queries into 4G and 5G parts.

**`abstain_node`** — already built in Phase 2, now also the terminus for
`insufficient_context`.

**Verify:** "What is the capital of France?" terminates at `refuse` with **zero**
generate calls — assert on recorded node path *and* token count; refusal accuracy
0/6 → 6/6 (excluding borderline Q15). A question with deliberately poisoned
retrieval routes rewrite → abstain instead of 3 pointless regenerations. Tokens
per run drop measurably from 124,701. Threshold sweep committed.

---

## Phase 5 — Retrieval quality and judge validation

Four interventions, **one PR each with its own metric delta** — the ablation
table is worth more than any single final number.

1. **`fetch_k`** — `max(20, top_k)` (`retriever.py:98`) is degenerate: `top_k=5`
   and `top_k=20` both fetch 20. Make it `rerank_multiplier * top_k` (default 8
   → 40 candidates). Cheapest possible reranker improvement, and `recall@20`
   being far above `recall@5` says to do exactly this.
2. **BM25 + RRF fusion** (`src/retrieval/hybrid.py`, `rank-bm25` in-process over
   2,644 chunks). Targets Q6-class misses where dense embeddings miss exact
   acronyms. Qdrant-native sparse vectors noted as future work — needs a reindex
   for a result BM25 gets in an afternoon.
3. **Reranker A/B** — `BAAI/bge-reranker-base` vs the current
   `ms-marco-MiniLM-L-6-v2`, reporting both quality *and* latency. Baseline p50
   retrieval is 1.76s; showing the measured tradeoff is the point.
4. **Lineage-aware boost** — Q1's gold chunk is literally titled "AMF" and the
   query asks about the AMF. The `lineage` field is in every payload and unused
   at retrieval time. Cheap, interpretable, targets the measured failure.

All zero-LLM-token work.

**Golden set:** label the 6 adversarial questions (currently
`retrieval_scored: false`, which is why n=6 and one label bug moved the mean 8
points). Add **cross-spec coverage** — did retrieval surface ≥1 chunk from *each*
spec needed to detect the conflation? That metric is specific to this project's
stated thesis and appears in no RAG tutorial.

**Judge validation:** hand-label ~25 (answer, verdict) pairs and report
**Cohen's κ** for judge-vs-human agreement. Without it, any groundedness number
comes from an unvalidated instrument. Given the 1/18 baseline I expect the
current judge is *over-strict*, rejecting correct answers for obvious-but-unstated
entailments — measuring that justifies every subsequent judge-prompt change.
Replace the fake conflation metric (`run_eval.py:117`:
`adv_caught = retries > 0` measures verifier dissatisfaction, not conflation
detection, and post-fix-#1 would read 0%) with a real judge.

**CI job 2:** retrieval regression on PRs touching `src/retrieval/**`,
`src/ingestion/**`, `eval/golden/**`. Qdrant service container + committed
snapshot restore (`data/snapshots/`, via Qdrant's snapshot API — faster and more
reproducible than re-embedding). No-regression ratchet at −0.02 tolerance against
the committed baseline, bumped deliberately on improvement. Zero LLM tokens.

---

## Phase 6 — Polish

CI job 3 (`workflow_dispatch` + nightly LLM eval, never blocking a PR). README
rewrite: architecture diagram, before/after table, ablation table, the ragas
dependency finding, honest limitations. `CLAUDE.md` update.

---

## Cut — scope creep

- **ragas/deepeval as the backbone** — measured unavailable (0.0.6 fallback).
- Multi-query fan-out with RRF over generated variants — one rewrite proves it.
- Conversational memory — the golden set is single-turn; memory invalidates every
  metric built here.
- Qdrant-native sparse vectors — reindex risk for an afternoon's BM25 result.
- Prometheus + Grafana — Langfuse covers it; a second stack is infra theatre.
- Agentic multi-hop retrieval, semantic caching, Streamlit migration.
- Reranker fine-tuning on 18 questions — would overfit. Saying *why not* in the
  README is the stronger signal.
- Re-parsing PDFs / touching `tree_builder.py` — working; leave alone except
  `chunk_id`.

---

## End-to-end verification

1. `docker compose up -d` → Qdrant + Langfuse healthy.
2. `pytest` → unit + contract tests green, no network.
3. `python -m src.ingestion.indexer` → deterministic chunk ids, point count
   matches chunk-file total.
4. `streamlit run app.py` → ask an in-scope question: ✅ Grounded badge, Trace tab
   shows triage → retrieve → grade → generate → verify with per-node latency and
   tokens. Ask "What is the capital of France?" → 🚫 Outside 3GPP scope, zero
   generate calls.
5. `python -m src.eval.runner --limit 3` → cheap smoke run writes
   `eval/runs/<id>/`.
6. `python -m src.eval.runner` → full 18; refusal accuracy 6/6 (ex-borderline);
   tokens per run below the 124,701 baseline; `summary.json` totals match
   Langfuse.
7. `python -m src.eval.retrieval_report` → recall/MRR/nDCG + cross-spec coverage,
   strict and prefix-credit, compared against the ratchet.
8. Open a PR touching `src/retrieval/` → CI job 2 runs retrieval regression on the
   snapshot and blocks a deliberate regression.
