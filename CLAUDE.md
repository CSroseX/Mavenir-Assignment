# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG system that answers 3GPP telecom questions grounded in three specs: TS 23.501 (5G system architecture), TS 24.501 (5G NAS), TS 24.301 (EPS NAS). The three specs were chosen deliberately because they share terminology but differ in procedural detail — cross-spec conflation is the hallucination mode this system is built to defeat.

## Commands

`src/retrieval/retriever.py` and `src/ingestion/indexer.py` resolve `data/chunks/*_chunks.json` against `Settings().project_root` (see `src/config.py`), so they work from any cwd. Other scripts (`eval/baseline/*`) still assume the repo root.

```bash
# Qdrant must be live on :6333 for retrieval, the app, and all eval scripts
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

pip install -r requirements.txt   # docling is NOT in requirements.txt; pip install docling to re-parse PDFs

streamlit run app.py              # chat UI + eval dashboard

# Ingestion pipeline — each stage is a separate CLI taking explicit paths.
# Only re-run from the stage you changed; outputs are cached as JSON.
python src/ingestion/parse.py data/raw_pdfs/ts_123501v171100p.pdf data/parsed/ts_123501v171100p.json
python src/ingestion/tree_builder.py data/parsed/ts_123501v171100p.json data/chunks/23.501_tree.json
python src/ingestion/chunker.py data/chunks/23.501_tree.json data/chunks/23.501_chunks.json 23.501 17.11.0
python src/ingestion/indexer.py    # DROPS and recreates the 3gpp_specs collection, then indexes all chunk files

# Single-query end-to-end run (retrieval + generate/verify graph, verbose)
python src/generation/verify.py "What is the AMF according to TS 23.501?"

# Retrieval only, against four hardcoded sample queries
python src/retrieval/retriever.py

# Full eval suite (18 questions, 5s sleep between each for rate limits)
python eval/run_eval.py

python eval/baseline/preflight.py  # check Qdrant reachability, chunk/point count parity, LLM creds
python eval/baseline/run_baseline.py
```

`eval/run_eval.py` resumes from whatever is already in `eval/results.json` and appends. **Delete `eval/results.json` to force a clean run** — otherwise a completed file means zero questions execute. `python -m src.eval.runner [--limit N] [--types normal,adversarial] [--results-path PATH]` is the newer, safer replacement (see Eval section below) — prefer it for new runs; `run_eval.py` is kept as-is, not deleted.

## Architecture

Four stages, each writing a cached artifact the next stage reads. No orchestrator script ties them together.

**Ingestion (`src/ingestion/`)** — `parse.py` (Docling → `data/parsed/`) → `tree_builder.py` (→ `*_tree.json`) → `chunker.py` (→ `*_chunks.json`) → `indexer.py` (→ Qdrant).

- `tree_builder.py` rebuilds document hierarchy from Docling's flat block list using a **depth stack**: clause depth = dot count (`5.3.4` → 3); on a new heading, pop until the stack top has strictly smaller depth, leaving the true parent on top. This produces each clause's full `lineage`. The heading regex must tolerate letter-suffixed clause IDs (`4.2.5a`, `4.2.8.1A`) — an earlier numeric-only regex silently buried 37 real clauses. If you touch that regex, inspect what lands in the downgraded (non-clause text) pile rather than trusting the summary count.
- `chunker.py` uses a `len//4` token heuristic with SOFT_LIMIT 400 / HARD_CAP 800. **Tables are never split or flattened** — a table emits one chunk whose `content` is the raw 2D `grid`, and encountering one flushes the pending text buffer to preserve reading order.
- `indexer.py` prepends a lineage header (`Spec {spec_id} {version} Clause {clause_id} {clause_title}`) to every chunk before embedding, and stores the **entire chunk dict as the Qdrant payload** — so retrieval filters and the generator prompt both read metadata straight off the payload. Tables go through `format_table_for_embedding` (markdown-ish) for the vector only; the payload keeps the grid. Vectors are 384-dim (`BAAI/bge-small-en-v1.5` via FastEmbed, cosine, configured in `src/config.py`) — changing the embed model means changing `vector_size` and reindexing.
- `indexer.py` computes a deterministic `chunk_id = sha1(spec_id|clause_id|ordinal)` per chunk, where `ordinal` is the chunk's position among chunks sharing the same `(spec_id, clause_id)` — `clause_id` alone isn't unique (every frontmatter chunk shares `clause_id "0"`). The Qdrant point id is derived from the same digest, so re-running the indexer over unchanged chunk files reproduces identical ids and point counts.

**Retrieval (`src/retrieval/retriever.py`)** — one `Retriever` class, dual-path:

- `_parse_query` regex-detects a spec number (`23.501`, optionally `TS`-prefixed) and/or a clause reference (`clause|sec|section|subclause 5.5.1`).
- Targeted path builds a Qdrant payload `Filter`. Clause filtering is **prefix-expanded**: a hit on `5.5` matches `5.5` plus every descendant, resolved against `valid_clause_ids` loaded from the chunk JSON files at construction time. This is why the retriever needs `data/chunks/` on disk even though vectors live in Qdrant.
- Both paths fetch `max(fetch_k, top_k)` from Qdrant (`fetch_k` from `src/config.py`, still the degenerate `20` default — Phase 5's fix), then rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` and truncate to `top_k`. The returned dicts carry both `qdrant_score` and the final `score` (cross-encoder).
- Construction loads two models and prints progress — it is slow. `app.py` instantiates a fresh `Retriever` per query, which is a known inefficiency, not a design choice.

**Generation (`src/generation/`)** — LangGraph state machine in `graph.py`: `START → generate → verify → (END | generate | abstain)`. `GraphState` (`src/generation/state.py`) carries `query, chunks, answer, verification_passed, feedback, retries, attempt, grounding_retries, transport_attempts, termination_reason, abstention`, plus unpopulated placeholder fields (`triage_decision`, `retrieval_scores`, `sufficiency`, `rewrites`, `citations`) reserved for Phase 4. The conditional-edge router (`should_retry`) lives in `src/generation/routers.py`, pure and unit-tested independently of the graph; its `"flag_unverified"` return value is still the routing key name (unchanged, so the router needed no edit when the node behind it changed) but `graph.py` now wires it to `abstain`.

- `src/generation/nodes/` is a package, not a single file: `core.py` holds `generate_node`/`verify_claims_node`/the LLM client setup (what used to be `nodes.py`), `abstain.py` holds `abstain_node`, and `__init__.py` re-exports both plus `client`/`MODEL_NAME` so existing `from src.generation.nodes import X` and `from src.generation import nodes as gen_nodes; gen_nodes.client` call sites didn't need to change.
- `generate_node` owns the `attempt` counter (increments itself); a first-pass success records `attempt == 1`. The legacy `retries` field is still populated (mirrors `grounding_retries`) because `app.py` and the eval scripts read it for the "self-corrected" badge and retry-rate stats.
- `verify_claims_node` is a strict-entailment judge returning JSON (`is_supported`, `reasoning`, `unsupported_claims`) at temperature 0, with a **typed exception boundary**: `APIStatusError` / `APIConnectionError` / `APITimeoutError` are transport failures — they set `termination_reason = "verifier_unavailable"`, bump `transport_attempts`, and do **not** increment `grounding_retries` or inject feedback (an outage is no longer indistinguishable from a genuine grounding failure). Malformed JSON gets one reask, then is treated as transport. Only a valid JSON response with `is_supported: false` increments `grounding_retries`.
- Unsupported claims are fed back as `feedback` into the next `generate` call. The router sends `verifier_unavailable` straight to `abstain` (no point burning the retry budget on an outage) and otherwise allows up to `max_retries` grounding retries before falling through to `abstain_node`, which returns a **structured abstention** (`found_clauses`, `unsupported_claims` parsed back out of `feedback`, `termination_reason`) instead of silently passing the answer through — `termination_reason` is preserved as `"verifier_unavailable"` when that's why the graph ended here, rather than being overwritten with `"verification_failed"`. Still not surfaced in the chat UI itself (that's `app.py`'s job, #19, not yet done).
- The generate prompt deliberately hides the retrieval system from the user: a word-boundary banned-phrase gate (`_detect_banned_phrase` in `nodes/core.py`) is a **real gate**, not just a log line — one targeted rephrase attempt, then the leak is allowed to stand. Bare substring matching on "retrieved"/"chunks" was replaced because it false-positived on legitimate 3GPP prose (e.g. "the UE retrieved the subscription data"). `<think>` blocks are stripped with regex.
- `nodes/core.py` owns all LLM client setup, read from `.env` at import: `USE_REMOTE_LLM=false` pins a local Ollama endpoint (`localhost:11434`, `qwen3.5:4b`); true picks OpenRouter when `OPEN_ROUTER_API_KEY` is set (or the base URL mentions openrouter), else OpenAI-compatible with `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME`. Both generate and verify share one client and model.

**Eval (`eval/`)** — `eval_set.json` tags each question `normal | out_of_scope | adversarial`; metrics derive from those tags (refusal accuracy over `out_of_scope`, conflation catch rate over `adversarial`, where "caught" means `retries > 0`). `eval/golden/golden_set.v1.json` is the merged, versioned canonical set (18 questions, stable `id`, `gold_clauses`, `retrieval_scored`, `corpus_answerable`, `borderline`) joining `eval_set.json` and `eval/baseline/gold.json`; `tests/test_golden_set.py` is a contract test asserting every `gold_clauses` entry resolves to a real chunk (exact or descendant) via `src/eval/metrics.py`. `eval/baseline/` is a separate, self-contained diagnostic harness with hand-labelled `gold.json` (`question_id` → gold `(spec_id, clause_id)` pairs, some flagged `retrieval_scored: false`) that measures recall@5/@20, MRR, and retrieval latency percentiles. Baseline scripts write only inside `eval/baseline/` and are frozen as the "before" measurement.
- `src/eval/metrics.py` is the promoted, fixed scorer: exact `(spec_id, clause_id)` equality undercounted, since some gold clauses (`24.501/5.1.3.2`, `23.501/5.15.2`) exist in the corpus only as descendant chunks. It reports strict and **prefix-credit** matching (a retrieved clause counts if it's the gold clause or a strict descendant) side by side, plus nDCG@10.
- `src/eval/runner.py` supersedes `eval/run_eval.py` (kept, not deleted): resume is keyed by the golden set's stable `id` rather than list position, and is refused (`GoldenSetDriftError`) if the results file's recorded content hash of the golden set no longer matches the one on disk — the old positional resume never validated that a result actually corresponded to the question at that index. A corrupted results file raises `ResultsFileCorruptError` instead of the old bare `except: pass` that silently restarted from zero and overwrote the previous file. Writes are atomic (temp file + `os.replace`). Supports `--limit`/`--types` for cheap iteration.

## Conventions

- Entry-point scripts append the project root to `sys.path` before importing `src.*`; keep that in any new script.
- `src/obs/logging.py` provides `get_logger()` (structlog): JSON lines to `logs/app.jsonl`, pretty console in dev. `nodes.py` and `retriever.py` use it instead of `print()` for payload dumps, candidate lists, and warnings/errors. CLI entry points (`retriever.py`'s `__main__` block, missing-dependency messages) still use plain `print()` since that output is the point of running the script, not debug noise.
- `spec_id` is bare and dotted (`23.501`, no `TS`); `version` is a separate field (`17.11.0`). Both are passed as chunker CLI args, so a wrong invocation mislabels every chunk from that spec.
- Several files wrap `sys.stdout` in a UTF-8 `TextIOWrapper` — Windows console default encoding breaks on spec text.

## Planned work (to-dos)

The full production-hardening plan lives in [docs/production-hardening-plan.md](docs/production-hardening-plan.md) — read it before starting any of the phases below. It carries the measured baseline, the defect diagnoses, and the per-phase verification steps. **Phase 1 and Phase 2 are partially implemented** (see checkmarks below); Phases 3–6 are not started.

Measured starting state (frozen in `eval/baseline/`, 18 questions): 1/18 grounded on first pass, 0/6 out-of-scope refusals, 124,701 tokens per run, of which 46,452 (37%) are spent answering questions that should have been refused. Progress tracked via GitHub issues #1–46 on `CSroseX/Mavenir-Assignment` (one epic per phase, one sub-issue per bullet below).

- [ ] **Phase 1 — Foundations (partial — 7 of 10 sub-issues closed).** ✅ `src/config.py` (pydantic-settings) centralizing all literals (#2); ✅ merged `eval/golden/golden_set.v1.json` (#3); ✅ structlog in `src/obs/logging.py`, debug prints removed (#5); ✅ prefix-credit scorer fix in `src/eval/metrics.py` with nDCG@10 (#7); ✅ golden-set contract test resolving `gold_clauses` against real chunks (#8); ✅ repo-root-anchored chunk glob in `retriever.py` (#9, landed as part of #2's config work); ✅ deterministic `chunk_id` in `indexer.py` (#10). Still open: `@st.cache_resource` in `app.py` (#4), pinned `pyproject.toml` with `docling` added (#6), CI lint+unit job (#11). `eval/baselines/retrieval_baseline.json` (the committed scorer re-run) still needs Qdrant live to generate.
- [ ] **Phase 2 — Reliability and truthful state (partial — 7 of 9 sub-issues closed).** Done: ✅ the `retries`/`attempt` off-by-one fix (#14), ✅ typed exception boundary in the verifier distinguishing transport failures from grounding failures (#15), ✅ banned-phrase check as a real word-boundary gate (#17), ✅ pure routers extracted to `src/generation/routers.py` (#18), ✅ expanded `GraphState` (#13), ✅ `abstain_node` replacing `flag_unverified_node` (#16), ✅ `src/eval/runner.py` with id-keyed resume and atomic writes (#20). Still open: `render_assistant_message()` in `app.py` switching on `termination_reason` (#19); delete stale `eval/results.json` and do one clean 18-question run with the new runner (#21) — needs Qdrant live and spends real LLM tokens, not run in this environment.
- [ ] **Phase 3 — Observability.** docker-compose (Qdrant + Langfuse); `langfuse.openai` client swap; `@observe` on nodes; `RunRecorder` → `eval/runs/<run_id>/`; Trace tab in the dashboard.
- [ ] **Phase 4 — Decision-nodes.** Move retrieval inside the graph, then add `triage` (the 37% token saving), `grade_sufficiency` (the fix for 17/18), `rewrite_query` (capped at 1), and `abstain`.
- [ ] **Phase 5 — Retrieval quality and judge validation.** `fetch_k` (today's `max(fetch_k, top_k)` is degenerate with the default config); BM25 + RRF; reranker A/B; lineage boost — one PR each with its own metric delta. Label the 6 adversarial questions; add cross-spec coverage; Cohen's κ for judge-vs-human agreement; CI retrieval-regression ratchet.
- [ ] **Phase 6 — Polish.** Nightly LLM eval job; README with before/after and ablation tables; update this file.

`top_k`/URL/collection literals already live in `src/config.py` (Phase 1, done). The verbose `[DEBUG]` print convention is replaced by structlog (Phase 1, done, see Conventions above) — the Phase 3 bullet to "purge the prompt-dumping prints" is satisfied early.
