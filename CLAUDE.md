# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG system that answers 3GPP telecom questions grounded in three specs: TS 23.501 (5G system architecture), TS 24.501 (5G NAS), TS 24.301 (EPS NAS). The three specs were chosen deliberately because they share terminology but differ in procedural detail — cross-spec conflation is the hallucination mode this system is built to defeat.

## Commands

Everything runs from the repo root; scripts resolve `data/chunks/*_chunks.json` via relative glob, so running from a subdirectory silently yields zero chunks.

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

`eval/run_eval.py` resumes from whatever is already in `eval/results.json` and appends. **Delete `eval/results.json` to force a clean run** — otherwise a completed file means zero questions execute.

## Architecture

Four stages, each writing a cached artifact the next stage reads. No orchestrator script ties them together.

**Ingestion (`src/ingestion/`)** — `parse.py` (Docling → `data/parsed/`) → `tree_builder.py` (→ `*_tree.json`) → `chunker.py` (→ `*_chunks.json`) → `indexer.py` (→ Qdrant).

- `tree_builder.py` rebuilds document hierarchy from Docling's flat block list using a **depth stack**: clause depth = dot count (`5.3.4` → 3); on a new heading, pop until the stack top has strictly smaller depth, leaving the true parent on top. This produces each clause's full `lineage`. The heading regex must tolerate letter-suffixed clause IDs (`4.2.5a`, `4.2.8.1A`) — an earlier numeric-only regex silently buried 37 real clauses. If you touch that regex, inspect what lands in the downgraded (non-clause text) pile rather than trusting the summary count.
- `chunker.py` uses a `len//4` token heuristic with SOFT_LIMIT 400 / HARD_CAP 800. **Tables are never split or flattened** — a table emits one chunk whose `content` is the raw 2D `grid`, and encountering one flushes the pending text buffer to preserve reading order.
- `indexer.py` prepends a lineage header (`Spec {spec_id} {version} Clause {clause_id} {clause_title}`) to every chunk before embedding, and stores the **entire chunk dict as the Qdrant payload** — so retrieval filters and the generator prompt both read metadata straight off the payload. Tables go through `format_table_for_embedding` (markdown-ish) for the vector only; the payload keeps the grid. Vectors are 384-dim (`BAAI/bge-small-en-v1.5` via FastEmbed, cosine), hardcoded in both indexer and retriever — changing the embed model means changing `VectorParams(size=...)` and reindexing.

**Retrieval (`src/retrieval/retriever.py`)** — one `Retriever` class, dual-path:

- `_parse_query` regex-detects a spec number (`23.501`, optionally `TS`-prefixed) and/or a clause reference (`clause|sec|section|subclause 5.5.1`).
- Targeted path builds a Qdrant payload `Filter`. Clause filtering is **prefix-expanded**: a hit on `5.5` matches `5.5` plus every descendant, resolved against `valid_clause_ids` loaded from the chunk JSON files at construction time. This is why the retriever needs `data/chunks/` on disk even though vectors live in Qdrant.
- Both paths fetch `max(20, top_k)` from Qdrant, then rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` and truncate to `top_k`. The returned dicts carry both `qdrant_score` and the final `score` (cross-encoder).
- Construction loads two models and prints progress — it is slow. `app.py` instantiates a fresh `Retriever` per query, which is a known inefficiency, not a design choice.

**Generation (`src/generation/`)** — LangGraph state machine in `graph.py`: `START → generate → verify → (END | generate | flag_unverified)`. `GraphState` carries `query, chunks, answer, verification_passed, feedback, retries`.

- `verify_claims_node` is a strict-entailment judge returning JSON (`is_supported`, `reasoning`, `unsupported_claims`) at temperature 0. It increments `retries` itself, and **any exception sets `verification_passed = False`** — so an API/JSON failure is indistinguishable from a genuine grounding failure and burns a retry.
- Unsupported claims are fed back as `feedback` into the next `generate` call. Router allows up to 3 retries, then falls through to `flag_unverified_node`, which passes the answer through **unchanged and unmarked** — the "unverified" state is visible only via `verification_passed`/`status`, never in the answer text.
- The generate prompt deliberately hides the retrieval system from the user: a banned-phrase list (`chunks`, `retrieved`, `provided source`, …) is checked post-hoc and only warns; `<think>` blocks are stripped with regex.
- `nodes.py` owns all LLM client setup, read from `.env` at import: `USE_REMOTE_LLM=false` pins a local Ollama endpoint (`localhost:11434`, `qwen3.5:4b`); true picks OpenRouter when `OPEN_ROUTER_API_KEY` is set (or the base URL mentions openrouter), else OpenAI-compatible with `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME`. Both generate and verify share one client and model.

**Eval (`eval/`)** — `eval_set.json` tags each question `normal | out_of_scope | adversarial`; metrics derive from those tags (refusal accuracy over `out_of_scope`, conflation catch rate over `adversarial`, where "caught" means `retries > 0`). `eval/baseline/` is a separate, self-contained diagnostic harness with hand-labelled `gold.json` (`question_id` → gold `(spec_id, clause_id)` pairs, some flagged `retrieval_scored: false`) that measures recall@5/@20, MRR, and retrieval latency percentiles. Baseline scripts write only inside `eval/baseline/`.

## Conventions

- Entry-point scripts append the project root to `sys.path` before importing `src.*`; keep that in any new script.
- `src/obs/logging.py` provides `get_logger()` (structlog): JSON lines to `logs/app.jsonl`, pretty console in dev. `nodes.py` and `retriever.py` use it instead of `print()` for payload dumps, candidate lists, and warnings/errors. CLI entry points (`retriever.py`'s `__main__` block, missing-dependency messages) still use plain `print()` since that output is the point of running the script, not debug noise.
- `spec_id` is bare and dotted (`23.501`, no `TS`); `version` is a separate field (`17.11.0`). Both are passed as chunker CLI args, so a wrong invocation mislabels every chunk from that spec.
- Several files wrap `sys.stdout` in a UTF-8 `TextIOWrapper` — Windows console default encoding breaks on spec text.

## Planned work (to-dos)

The full production-hardening plan lives in [docs/production-hardening-plan.md](docs/production-hardening-plan.md) — read it before starting any of the phases below. It carries the measured baseline, the defect diagnoses, and the per-phase verification steps. **Nothing in it is implemented yet.**

Measured starting state (frozen in `eval/baseline/`, 18 questions): 1/18 grounded on first pass, 0/6 out-of-scope refusals, 124,701 tokens per run, of which 46,452 (37%) are spent answering questions that should have been refused.

- [ ] **Phase 1 — Foundations.** `src/config.py` (pydantic-settings, already installed); structlog; pinned `pyproject.toml` (add the missing `docling`); prefix-credit scorer fix; merged `eval/golden/golden_set.v1.json`; golden-set contract test; `@st.cache_resource` in `app.py`; repo-root-anchored chunk glob; deterministic `chunk_id`; CI lint+unit job. No LLM calls.
- [ ] **Phase 2 — Reliability and truthful state.** Fix the `retries` off-by-one; typed exception boundary in the verifier; `abstain_node` with a `termination_reason` enum; banned-phrase check as a real gate; `src/eval/runner.py` with id-keyed resume and atomic writes; one clean 18-question run.
- [ ] **Phase 3 — Observability.** docker-compose (Qdrant + Langfuse); `langfuse.openai` client swap; `@observe` on nodes; `RunRecorder` → `eval/runs/<run_id>/`; Trace tab in the dashboard; purge the prompt-dumping prints.
- [ ] **Phase 4 — Decision-nodes.** Move retrieval inside the graph, then add `triage` (the 37% token saving), `grade_sufficiency` (the fix for 17/18), `rewrite_query` (capped at 1), and `abstain`.
- [ ] **Phase 5 — Retrieval quality and judge validation.** `fetch_k` (today's `max(20, top_k)` is degenerate); BM25 + RRF; reranker A/B; lineage boost — one PR each with its own metric delta. Label the 6 adversarial questions; add cross-spec coverage; Cohen's κ for judge-vs-human agreement; CI retrieval-regression ratchet.
- [ ] **Phase 6 — Polish.** Nightly LLM eval job; README with before/after and ablation tables; update this file.

Two conventions in this file change when the plan lands: the verbose `[DEBUG]` print convention above is replaced by structlog in Phase 3, and `top_k`/URL/collection literals move into `src/config.py` in Phase 1.

### Not pursued

GitHub issues for these phases were requested but could not be created — the `github` MCP server fails to connect this session (`Authorization header is badly formatted`) and the `gh` CLI is not installed. The remote is `CSroseX/Mavenir-Assignment`.
