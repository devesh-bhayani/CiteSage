# CiteSage — Document QA with Verified Citations

Architecture & design context: **PROJECT.md** (what this is, how the pipeline flows, what's load-bearing).
Known issues & scoped fixes: **GAPS.md** (audit ordered by severity — check it before "discovering" a problem).

## Tech Stack
- Python 3.11+, uv, LangGraph, ChromaDB, rank_bm25, sentence-transformers
- all-MiniLM-L6-v2 (embeddings), cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker)
- FastAPI, Streamlit, RAGAS, pytest + pytest-vcr

## Key Commands
- Lint: `ruff check src/ --fix && black --target-version py311 src/`
- Test: `pytest tests/ -v` (149 tests; unit-only: `pytest tests/unit -q`, ~1 min)
- Eval (full, 40–90 min on Ollama): `python -m citesage.evaluation.run_eval --dataset tests/eval/golden_dataset.json --output reports/baseline_ollama.json --errors reports/baseline_ollama_errors.jsonl`
- Eval smoke (cheap, do this first): add `--subset 5` or `--category factual_lookup`
- Regression diff: `python -m citesage.evaluation.check_regression --baseline <old> --current <new>`
- Ingest: `python -m citesage.cli --ingest data/documents/`
- Query: `python -m citesage.cli "question"`
- Serve: `uvicorn citesage.api.main:app --reload`
- UI: `streamlit run src/citesage/ui/app.py`
- Use the venv python on this machine: `.venv/Scripts/python.exe`

## Critical Rules
- YOU MUST cite-or-decline: never generate answers unsupported by retrieved context.
- YOU MUST use token-based chunking via tiktoken length_function, NOT character count.
- YOU MUST retry all LLM calls with exponential backoff (max 3, base 1s).
- NEVER hardcode model names — use config.yaml. NEVER write inline prompt strings — YAML in `src/citesage/prompts/v1/`.
- Use the grader model (Haiku-class) for routing/grading/judging; generator model (Sonnet-class) for generation only.
- All components (chunker, retriever, embedder, reranker) must be swappable via config. Log token usage per query.
- Type hints on all functions. Google-style docstrings on public functions. Black, ruff, mypy (enforced by pre-commit — a failing hook blocks the commit).

## Conventions
- Per-package `CLAUDE.md` files in `src/citesage/*/` carry module-specific rules — read the one for the package you're editing. (Warning: some contain stale numbers; config.yaml is the source of truth for thresholds.)
- Structured logging only: `structlog.get_logger(__name__)`, event names namespaced like `graph.retrieve`, `pipeline.complete`.
- Config: pydantic models in `src/citesage/config/__init__.py`; `get_settings()` is lru_cached — restart the process (or `get_settings.cache_clear()`) after editing config.yaml.
- Errors from the API follow RFC 7807; loaders raise `ValueError` for user-input problems.
- Dataclasses for result types (`PipelineResult`, `VerificationResult`, `QueryCost`).

## Gotchas (things that WILL bite you)
- **Eval exit code 1 = targets missed, not a crash.** Read the printed summary before diagnosing a "failure".
- **Eval writes output files only at the very end.** A killed run leaves stale files; check the JSON `timestamp` field before trusting `reports/baseline_ollama.json`.
- **Ollama eval metrics vary ±6–10 pp between identical runs** (no temperature/seed pinning — GAPS.md #1). Never conclude anything from a single-run delta.
- **Don't pass an `options` dict to `ollama.chat()`** in `utils/llm_factory.py` without testing: it historically forced model reloads and broke `<think>` suppression. Related: `max_tokens` is currently a no-op on Ollama.
- **The `lru_cache` model loaders** (`_load_cross_encoder` in `retrieval/reranker.py`, `_load_embedder` in `ingestion/storage.py`) **prevent an OOM** that used to kill eval runs at ~query 20. Never remove them.
- **Changing chunk size/overlap or chunking logic silently invalidates all 65 `expected_source_chunks` IDs** in `tests/eval/golden_dataset.json` (IDs are SHA-256 of content+path).
- `generation/generator.py` is dead code (superseded by graph nodes) — don't extend it. `prompts/v1/verify_citations.yaml` (plural) is also unused; the live one is `verify_citation.yaml`.
- config.yaml is found by walking up from cwd; tests chdir to repo root via `tests/conftest.py`.
- First query per process is slow (~30–60 s model cold-load). Ollama timeout is 180 s (`CITESAGE_OLLAMA_TIMEOUT`) to survive generator↔grader model swaps.
- Windows: use here-strings/heredocs for multi-line commit messages; expect CRLF warnings from git (harmless).

## Current state (2026-06-15)
- Provider is `ollama` in config.yaml. The Anthropic "green run" (the only path to the Phase-3 targets, esp. p95 < 5 s) has never run — blocked on API credits.
- Latest committed baseline (`reports/baseline_ollama.json`): all 4 Phase-3 targets FAIL (acc 64.6%, cit-prec 36.9%, decline-recall 70%, p95 122 s). Local-model ceiling + single-doc corpus (GAPS.md #2). Don't quote these numbers as achievements.

## Compaction
Preserve: modified file list, current phase, active test failures, arch decisions.
