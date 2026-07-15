# GAPS.md — Honest audit

*Written 2026-06-15 at commit `9923c3a` after a full read. Ordered by severity. Each item ends with a fix scoped small enough for a single focused session.*

---

## 1. Eval is non-deterministic — no metric from it can be trusted (CRITICAL for the project's whole premise)

**What:** Two identical back-to-back Ollama eval runs (same code, same data, same config) produced accuracy 64.6% vs 58.5%, decline recall 70% vs 60%. Nothing pins sampling: `OllamaLLM.invoke()` (`src/citesage/utils/llm_factory.py`) deliberately passes no options dict (it caused model reloads), so temperature/seed are model defaults — for the generator, the relevance grader, the citation judge, **and the eval's answer grader**. Randomness compounds across 4 LLM roles.
**Why it matters:** The project's flagship claim is measured quality. A ±6–10 pp noise floor makes tuning changes (like the `decline_threshold` -3.0 → -2.5 commit) unmeasurable — that change was "validated" against noise.
**Fix (scoped):** In `OllamaLLM.invoke`, add `options={"temperature": 0, "seed": 42}` to the `client.chat()` call; run `--subset 10` twice and diff outputs to confirm determinism AND confirm think-suppression still works (the code comment warns options may break it — if it does, set temperature via a custom Modelfile instead). One file, one test loop.

## 2. Corpus is 1 document / 4 chunks — retrieval metrics are theater (CRITICAL for credibility)

**What:** `data/documents/` holds one file; Chroma has 4 chunks. `retrieval.rerank_candidates: 15` exceeds the entire corpus. The 65-query eval "retrieves" from a pool where random choice gets ~25% precision.
**Why it matters:** "Hybrid retrieval + RRF + reranking" cannot be demonstrated, and anyone who inspects the repo sees it immediately.
**Fix (scoped):** Ingest 15–30 real documents (`python -m citesage.cli --ingest <dir>`). Chunk IDs are content-addressed, so existing golden `expected_source_chunks` stay valid — the eval gets *harder* (real distractors), not broken. Then re-run eval. No code changes.

## 3. BM25 pickle + Chroma client reconstructed on every query

**What:** `retrieve_node` (`src/citesage/graph/nodes.py:145`) builds `Retriever()` per query → `ChromaStore()` (new PersistentClient) + `BM25Index.load()` (full pickle deserialize from disk) every single query. Same anti-pattern that caused the model-reload OOM (fixed in `83bee9f`) — models are now cached, the stores are not.
**Why it matters:** Wasted latency on the p50 path and per-query allocation churn; grows with corpus size (gap #2's fix makes this worse).
**Fix (scoped):** Add a module-level `@lru_cache` `_get_retriever()` in `graph/nodes.py` (mirroring `_load_cross_encoder`), use it in `retrieve_node`. Invalidate cache on ingest (call `.cache_clear()` from `IngestPipeline` or accept staleness with a `# ponytail:`-style note). One file + one test.

## 4. `max_tokens` silently ignored on Ollama

**What:** `OllamaLLM.__init__` stores `num_predict` but `invoke()` never passes it (`src/citesage/utils/llm_factory.py:88-105`). The citation judge requests `max_tokens=16`, the grader 512 — Ollama ignores both and can generate unbounded output.
**Why it matters:** Latency (a chatty judge inflates p95) and silent API-contract violation; on Anthropic the same argument works, so behavior diverges by provider.
**Fix (scoped):** Bundle with gap #1: pass `options={"num_predict": self.num_predict, "temperature": 0, "seed": 42}` and verify think-suppression survives. If options truly must stay off, delete the `num_predict` parameter and document the limitation in the class docstring.

## 5. No CI at all, despite CI-shaped tooling

**What:** No `.github/` directory. Yet `run_eval` returns exit 1 on missed targets, `check_regression.py` exists "(used in CI)" per README, and pre-commit hooks are configured.
**Why it matters:** 149 tests only run when someone remembers; the regression gate has no enforcement point; the README overclaims.
**Fix (scoped):** Add one `.github/workflows/test.yml`: checkout, `uv sync`, `pytest tests/unit tests/integration`. Skip the eval job (needs Ollama/credits). ~30 lines.

## 6. Four copies of LLM retry logic; two copies of `_format_sources` and `_merge_usage`

**What:** Exponential-backoff retry is reimplemented in `graph/nodes.py:_llm_invoke_with_retry`, `generation/generator.py:_invoke_with_retry`, `generation/citation_verifier.py:_llm_judge`, and `evaluation/run_eval.py:AnswerGrader.grade` (subtly different: 3 attempts vs loop-else, different logging). `_format_sources` exists in both `nodes.py` and `generator.py`; `_merge_usage` in both `nodes.py` and `citation_verifier.py`.
**Why it matters:** A retry-behavior fix (e.g. retrying only on `OllamaConnectionError`) must be applied 4 places; they will drift.
**Fix (scoped):** Move retry + `_merge_usage` into `utils/llm_factory.py` (or a new `utils/llm_retry.py`), import everywhere. Pure refactor, tests exist to catch breakage.

## 7. Dead code: `generation/generator.py` and `prompts/v1/verify_citations.yaml`

**What:** `Generator` class is imported nowhere (grep confirms); the LangGraph nodes superseded it. `verify_citations.yaml` (plural) is never loaded — only `verify_citation.yaml` (singular) is.
**Why it matters:** A newcomer (or a smaller model) will "fix" or extend the dead path and wonder why nothing changes. This nearly duplicates the fast-path node's logic.
**Fix (scoped):** Delete `src/citesage/generation/generator.py` and `prompts/v1/verify_citations.yaml`; fix any `generation/__init__.py` re-exports; run tests.

## 8. Untested critical paths

**What:** No unit tests for: `retrieval/reranker.py` (threshold/skip_threshold behavior), `retrieval/retriever.py` (where-filtering, `_matches_where`), `utils/llm_factory.py` (provider dispatch, Ollama error mapping, `</think>` stripping), `evaluation/run_eval.py` metric math (`check_citations`, `compute_metrics`, `_parse_grade_response`), `graph/nodes.py:_parse_grade_indices`. Existing tests cover chunker, RRF, citation verifier, config, cost tracker, and stubbed graph routing/API.
**Why it matters:** The two metric functions define the project's success criteria — a regression there was already shipped once (boolean citation precision bug, fixed in Phase 3 tuning). The `</think>`-stripping and grade-parsing code paths are the fragile text-munging kind.
**Fix (scoped):** One new file `tests/unit/test_eval_metrics.py` covering `check_citations` + `_parse_grade_response` + `_parse_grade_indices` with table-driven cases (~15 cases). Highest value per line of any fix here. A second session can add `test_llm_factory.py` with a fake ollama client.

## 9. Security (all LOW severity in the local-dev context, flag before any real deployment)

- **Non-constant-time API key compare** — `received != expected` in `api/main.py:require_api_key`. Fix: `secrets.compare_digest`. One line.
- **BM25 index is `pickle.load`ed** (`ingestion/storage.py:BM25Index.load`) — arbitrary code execution if `data/bm25_index.pkl` is attacker-supplied. Fine locally; unacceptable if the data dir is ever shared/user-writable. Fix later: version-tagged JSON + rebuilding, or at minimum a docstring warning.
- **Rate-limit key embeds the first 12 chars of the API key** (`_rate_limit_key`) — key material can leak into limiter storage/logs. Fix: hash the header (`sha256(header)[:12]`).
- **`/health` reveals whether `ANTHROPIC_API_KEY` is set** to unauthenticated callers. Debatable; consider gating the `has_credentials` field behind auth.
- `.env` handling is **correct** (gitignored, only `.env.example` tracked) — no action.

## 10. Docs/config disagree with code (each one is a future wrong decision)

- `CitationVerifier.WEAK_THRESHOLD = 0.55` but its own docstring, the module docstring, and `nodes.py:verify_citations_node` docstring all say **0.3** (`generation/citation_verifier.py:86-89`).
- `graph/CLAUDE.md` says routing threshold **0.7**; `config.yaml` has **0.8**.
- `RetrievalConfig.decline_threshold` code default **-5.0** (`config/__init__.py:64`) vs yaml **-2.5** — code default is dead weight that misleads.
- `graph/pipeline.py` docstring promises "fast p50 ~2 s" — true only for the never-run Anthropic provider; Ollama p50 is ~6 s.
- `pyproject.toml` description is the scaffold placeholder "Add your description here".
- Junk file in repo root: `Usersadmin.claudesettings.json` (0 bytes, committed by accident).
**Fix (scoped):** Single doc-sync commit: correct the four docstrings/CLAUDE.md numbers, delete the junk file, write a real pyproject description. No behavior change.

## 11. Half-finished / aspirational features

- **README roadmap items** (SSE streaming via `astream_events`, weekly eval cron, feedback loop, multi-tenant): unstarted. `sse-starlette` is installed but unused — remove the dep or ship streaming.
- **RAGAS** integration exists behind `--ragas` but is unexercised (needs an eval-capable LLM; never run since Ollama switch).
- **`--rebuild` BM25 flag** documented in ingestion CLAUDE.md; `IngestPipeline.rebuild_bm25` exists but no CLI flag wires to it (`cli.py` has no `--rebuild`).
- **VCR/pytest-recording** is a dev dependency and `.gitignore` reserves `tests/fixtures/cassettes/`, but no cassette tests exist.
**Fix (scoped):** Pick per item: wire `--rebuild` into `cli.py` (10 lines), drop `sse-starlette` until streaming lands.

## 12. Eval budget cap is a no-op on Ollama

**What:** `BUDGET_CAP_USD = 2.0` stops the run when cumulative cost hits the cap — but Ollama pricing is configured as $0, so the cap never triggers locally. Harmless today; on Anthropic the cap will stop a full 65-query run mid-flight if Sonnet pricing makes a run cost >$2 (plausible: ~2.7k tokens/query avg).
**Why it matters:** The first funded Anthropic run may silently stop early and produce a partial report that reads like a full one (`items_completed` < `total_queries` is recorded, but the summary doesn't shout).
**Fix (scoped):** Before the Anthropic run: estimate cost (65 × avg tokens × Sonnet pricing), pass `--budget` explicitly, and add a loud `*** PARTIAL RUN ***` banner to `print_summary` when `items_completed < total_queries`. ~10 lines.
