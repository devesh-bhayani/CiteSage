# GAPS.md — Honest audit

*Written 2026-06-15 at commit `9923c3a` after a full read. Ordered by severity. Each item ends with a fix scoped small enough for a single focused session.*

---

## 1. ~~Eval is non-deterministic~~ — FIXED 2026-06-15

**What it was:** Two identical back-to-back Ollama eval runs (same code, data, config) produced accuracy 64.6% vs 58.5%, decline recall 70% vs 60%. Nothing pinned sampling: `OllamaLLM.invoke()` passed no options dict, so temperature/seed were model defaults across all four LLM roles (generator, relevance grader, citation judge, eval grader). Randomness compounded.

**Fix shipped:** `OllamaLLM.invoke()` now passes `options={"temperature": 0, "seed": _OLLAMA_SEED}` (`src/citesage/utils/llm_factory.py`); seed defaults to 42, override via `CITESAGE_OLLAMA_SEED`.

**Verified, not assumed:**
- Wrapper level: same prompt ×3 → 1 distinct output (was 3).
- End-to-end: `--subset 10` run twice → **every** aggregate and per-category metric identical (accuracy 0.9500 / 0.9500, citation precision 0.7000 / 0.7000, decline recall 1.0 / 1.0, fast-path ratio 0.9 / 0.9). Per-query records byte-identical once `latency_ms` (wall-clock) is excluded.
- The old code comment claiming options "breaks thinking suppression" was **empirically false**: `<think>` tags leak with *and* without options; the `</think>` strip is the actual defense. Comment corrected.
- No model-reload penalty: call latencies flat at 2.9/2.9/3.1 s.

**Residual:** baselines recorded before this commit (incl. `reports/baseline_ollama.json`, acc 64.6%) still carry the old ±6–10 pp noise floor — they are not comparable to post-pinning runs. Re-baseline before drawing any conclusion from a delta. Note the pinning makes runs *reproducible*, not *correct*: it freezes one sample of model behaviour, so absolute scores can still shift if the model or prompts change.

## 2. ~~Corpus is 1 document / 4 chunks~~ — FIXED 2026-07-17, surfaced a new finding (was CRITICAL)

**What it was:** `data/documents/` held one file; Chroma had 4 chunks. `retrieval.rerank_candidates: 15` exceeded the entire corpus, so "hybrid retrieval + RRF + reranking" was never actually exercised — every retrieval trivially returned all 4 chunks regardless of query.

**Fix shipped:** Added 15 original markdown documents (`data/documents/*.md` — RNNs, LSTM/GRU, BERT, GPT, word embeddings, optimizers, dropout/regularization, LayerNorm/BatchNorm, tokenization, positional encoding variants, ViT, RAG, fine-tuning/PEFT, quantization, diffusion models) covering adjacent ML/DL topics chosen to give real, topically-related distractors against the existing 65-question eval set. Corpus is now 16 documents / 34 chunks. `transformer_architecture.md` was left untouched.

**Verified, not assumed:**
- All 4 chunk IDs referenced by `tests/eval/golden_dataset.json` still resolve in Chroma post-ingest (content-addressed IDs, unaffected by adding unrelated files).
- Spot queries confirm topical discrimination: "How does dropout work?" → `regularization_dropout.md` top-ranked (score 4.03); "What is the self-attention formula?" → `transformer_architecture.md` (0.99); a query below the fast-path confidence bar still correctly ranked the right document first among real competitors, rather than failing.
- 149 unit/integration tests unaffected (isolated fixtures).

**New finding surfaced by this fix (not a regression — this is the corpus finally being large enough to reveal it):** across the 55 answerable golden questions, the expected source chunk reaches the reranker's candidate pool (top-15, post RRF fusion) only **80%** of the time (44/55), and survives into the final top-5 handed to generation only **58%** of the time (32/55). With the old 4-chunk corpus this was ~100% by construction, which is exactly why the old baseline's citation/accuracy numbers couldn't be trusted as retrieval quality signal.

**Follow-up — RESOLVED 2026-07-20, see gap #13.** The tuning call below was made with an offline sweep (`scripts/retrieval_recall.py`): `rerank_candidates` 15 → 20, everything else unchanged. The sweep also showed the knobs are not the real constraint — the cross-encoder is. Original text kept for context:

`retrieval.bm25_top_k` / `vector_top_k` (20 each) and `rerank_candidates` (15) in `config.yaml` were never tuned against a real multi-document corpus — they were sized when 4 chunks was the entire corpus. Whether to raise them (more recall headroom, more reranker cost) or leave them (representative of a real deployment's retrieval budget) is a tuning call, not a bug fix — don't silently change these without re-running eval to check the effect, per the "don't retune thresholds against a noisy metric" lesson from GAPS.md #1. A full 65-query LLM eval against this new corpus has **not** been run yet; `reports/baseline_ollama.json` still reflects the old 4-chunk corpus and should not be compared against a future run on this expanded one.

## 3. ~~BM25 pickle + Chroma client reconstructed on every query~~ — FIXED 2026-07-18

**What it was:** `retrieve_node` builds `Retriever()` per query → new `chromadb.PersistentClient` + full BM25 pickle deserialize from disk, every single query. Same anti-pattern that caused the model-reload OOM (`83bee9f`).

**Fix shipped:** Cached at the storage layer (`src/citesage/ingestion/storage.py`), NOT by caching the `Retriever` object — integration tests patch `nodes.Retriever` at ~10 sites and rely on per-call construction, so a cached Retriever would freeze mocks. Instead: `_get_chroma_client(path)` (`lru_cache`) and an mtime-tagged `_BM25_CACHE` — `BM25Index.load()` stats the file and reuses the cached instance only when mtime matches, so a re-ingest from *another process* (CLI ingest while the API serves) is picked up automatically; in-process `save()` refreshes the entry. The empty/missing-file case is never cached.

**Verified:** warm `Retriever()` construction 6.8 s → **~2 ms**; coherence script covers load-after-save freshness, external-rewrite detection (mtime bump → reload), and missing-file behavior; 175 tests pass. (One test-script false alarm worth remembering: a 2-doc BM25 corpus gives IDF = ln(1) = 0 for a term in 1 of 2 docs — all scores zero. Use ≥3 docs when sanity-checking BM25.)

## 4. `max_tokens` silently ignored on Ollama

**What:** `OllamaLLM.__init__` stores `num_predict` but `invoke()` never passes it (`src/citesage/utils/llm_factory.py:88-105`). The citation judge requests `max_tokens=16`, the grader 512 — Ollama ignores both and can generate unbounded output.
**Why it matters:** Latency (a chatty judge inflates p95) and silent API-contract violation; on Anthropic the same argument works, so behavior diverges by provider.
**Correction (2026-06-15):** the original advice here — "bundle with gap #1, pass `num_predict` in the options dict" — is **wrong and must not be followed**. While fixing #1 it turned out qwen3 models emit reasoning *before* their answer, so honouring `max_tokens=16` from the citation judge would truncate mid-reasoning and destroy the YES/NO/PARTIAL verdict. `max_tokens` being ignored is what currently keeps the judge working.
**Fix (scoped):** Do NOT pass `num_predict`. Instead make the contract honest: drop the unused `num_predict` parameter from `OllamaLLM.__init__`, and document in the class docstring that token caps are not honoured on Ollama because thinking models need headroom. If a cap is ever genuinely needed, size it well above the reasoning budget (hundreds, not 16) and re-verify the judge still returns a verdict.

## 5. ~~No CI at all~~ — FIXED 2026-07-18

**Fix shipped:** `.github/workflows/test.yml` — ubuntu, uv (cached), ruff + black check, then `pytest tests/unit tests/integration` (175 tests; LLM calls stubbed). HuggingFace model cache keyed on config.yaml. The eval job is deliberately excluded (needs a live Ollama daemon or funded Anthropic credits — noted in the workflow). First green run on GitHub not yet observed — check the Actions tab after the next push; the lockfile was resolved on Windows, so a Linux-only dependency hiccup on first run is the most likely failure mode.

## 6. Four copies of LLM retry logic; two copies of `_format_sources` and `_merge_usage`

**What:** Exponential-backoff retry is reimplemented in `graph/nodes.py:_llm_invoke_with_retry`, `generation/generator.py:_invoke_with_retry`, `generation/citation_verifier.py:_llm_judge`, and `evaluation/run_eval.py:AnswerGrader.grade` (subtly different: 3 attempts vs loop-else, different logging). `_format_sources` exists in both `nodes.py` and `generator.py`; `_merge_usage` in both `nodes.py` and `citation_verifier.py`.
**Why it matters:** A retry-behavior fix (e.g. retrying only on `OllamaConnectionError`) must be applied 4 places; they will drift.
**Fix (scoped):** Move retry + `_merge_usage` into `utils/llm_factory.py` (or a new `utils/llm_retry.py`), import everywhere. Pure refactor, tests exist to catch breakage.

## 7. Dead code: `generation/generator.py` and `prompts/v1/verify_citations.yaml`

**What:** `Generator` class is imported nowhere (grep confirms); the LangGraph nodes superseded it. `verify_citations.yaml` (plural) is never loaded — only `verify_citation.yaml` (singular) is.
**Why it matters:** A newcomer (or a smaller model) will "fix" or extend the dead path and wonder why nothing changes. This nearly duplicates the fast-path node's logic.
**Fix (scoped):** Delete `src/citesage/generation/generator.py` and `prompts/v1/verify_citations.yaml`; fix any `generation/__init__.py` re-exports; run tests.

## 8. Untested critical paths

**What:** No unit tests for: `retrieval/reranker.py` (threshold/skip_threshold behavior), `retrieval/retriever.py` (where-filtering, `_matches_where`), `utils/llm_factory.py` (provider dispatch, Ollama error mapping, `</think>` stripping), `evaluation/run_eval.py:compute_metrics`. Existing tests cover chunker, RRF, citation verifier, config, cost tracker, and stubbed graph routing/API.
**Partially fixed 2026-07-18:** `tests/unit/test_eval_metrics.py` (26 table-driven cases) now covers `check_citations` (incl. a regression test for the historic boolean-precision bug), `_parse_grade_response`, and `_parse_grade_indices` (incl. the `'["1"]'`-parses-to-`[]` sharp edge, documented as current behavior with guidance not to delete it).
**Remaining (scoped):** `test_llm_factory.py` with a fake ollama client (provider dispatch, `OllamaConnectionError` mapping, `</think>` strip), and cases for `compute_metrics` aggregation.

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

## 12. ~~Eval budget cap is a no-op on Ollama~~ — FIXED 2026-07-23

**Fix shipped:**
- `--estimate-cost [REPORT]` on `run_eval` projects a funded run from a prior report's measured token usage against `config.yaml` pricing, then exits without touching the pipeline. Exit code 0 = fits the budget, 1 = over, so it works as a pre-flight gate.
- Loud `*** PARTIAL RUN - n/N QUERIES COMPLETED ***` banner in `print_summary` whenever `items_completed < total_queries`, stating that the metrics cover the completed subset only and must not be committed as a baseline. ASCII-only so it survives a Windows cp1252 console.
- `tests/unit/test_budget_guard.py` (11 cases) pins both, including the off-by-one (64/65 still warns), the ASCII constraint, and the projection arithmetic. Pricing is stubbed so the budget assertions don't silently skip themselves when the configured provider is free.

**The number Phase 4 existed to produce:** measured from `reports/baseline_ollama.json` (78,452 in / 85,999 out over 65 queries = 1,207 in / 1,323 out per query), a full 65-query Anthropic run projects to **$1.53 upper bound** (all tokens at Sonnet $3/$15 per M) against the **$2.00** cap — it fits, with ~31% headroom. Lower bound at Haiku rates is $0.13; true cost sits between, since generation runs on Sonnet and grading on Haiku.

**Read the headroom carefully before spending:** the upper bound is *not* conservative in every direction. It scales Ollama-measured tokens, and qwen3's output count is inflated by `<think>` reasoning blocks Claude won't emit — that pushes the real bill *down*. But nothing in the projection covers retries (3 attempts per call on failure), RAGAS if `--ragas` is passed, or a prompt change between now and the run — all of which push *up*. $1.53 of a $2.00 cap is not much room for a surprise. Pass `--budget` explicitly and re-run `--estimate-cost` immediately before the funded run rather than trusting this figure.

**Verified:** `--estimate-cost` run against both the live Ollama config (correctly projects $0.00, free models) and a temporarily-swapped Anthropic config (the $1.53 above; config.yaml restored, `git diff` clean afterwards).

**Original finding, for context:**

**What:** `BUDGET_CAP_USD = 2.0` stops the run when cumulative cost hits the cap — but Ollama pricing is configured as $0, so the cap never triggers locally. Harmless today; on Anthropic the cap will stop a full 65-query run mid-flight if Sonnet pricing makes a run cost >$2 (plausible: ~2.7k tokens/query avg).
**Why it matters:** The first funded Anthropic run may silently stop early and produce a partial report that reads like a full one (`items_completed` < `total_queries` is recorded, but the summary doesn't shout).
**Fix (scoped):** Before the Anthropic run: estimate cost (65 × avg tokens × Sonnet pricing), pass `--budget` explicitly, and add a loud `*** PARTIAL RUN ***` banner to `print_summary` when `items_completed < total_queries`. ~10 lines.

## 13. The cross-encoder reranker is the retrieval ceiling — and its scores don't separate answerable from unanswerable (NEW 2026-07-20, HIGH)

**What:** Measured offline with `scripts/retrieval_recall.py` over the 55 answerable golden questions (no LLM calls; BM25/vector rankings and cross-encoder scores precomputed per question, so configs are compared by re-slicing).

Retrieval reaches the right chunk; the reranker then throws it away. With `rerank_candidates: 20`, the expected chunk is in the candidate pool **84%** of the time but survives into the final top-5 only **60%** of the time. Where `ms-marco-MiniLM-L-6-v2` ranks the correct chunk across the whole 34-chunk corpus:

| CE rank of correct chunk | share |
|---|---|
| 1 | 29% |
| 2–5 | 29% (→ 58% reach top-5) |
| 6–15 | 18% |
| **16+ (below half the corpus)** | **24%** |

Final top-5 recall equals the reranker's own top-5 rate almost exactly — fusion and pool size are not the binding constraint. Miss classification at the tuned config: **13 rerank-drop, 8 both-miss, 1 rrf-dilution.**

**Why the config knobs can't fix it:** the sweep raised pool recall to 87% (candidates 30, bm25/vector 30) and final recall still sat at 60% — the cross-encoder re-buries the chunk regardless of what it is handed. `rerank_top_k` 5 → 7 does lift final recall to 64%, but it adds two more distractor slots to the generation context, which works directly against citation precision (the metric this was meant to rescue). Not taken.

**Reranker bake-off (offline, same 55 questions, pool held at 20):**

| model | final top-5 recall | rerank 20 candidates (CPU) |
|---|---|---|
| `ms-marco-MiniLM-L-6-v2` (current) | 60% | **385 ms** |
| `ms-marco-MiniLM-L-12-v2` | 62% | — |
| `BAAI/bge-reranker-base` | 53% (worse) | — |
| `BAAI/bge-reranker-v2-m3` | **71%** | **12,860 ms** |

**Why the obvious upgrade is rejected:** `bge-reranker-v2-m3` is the only model that meaningfully beats the incumbent (+11pp), but at 12.9 s to rerank 20 candidates on CPU it is 33× slower than L-6 and **2.5× the entire 5 s p95 target on its own, before a single LLM token**. The GPU is already contended by Ollama. `L-12` buys +2pp (one question — noise at n=55) for roughly double the compute. So no swap.

**Second finding, arguably worse:** cross-encoder scores barely separate answerable from unanswerable questions, which is the root cause of decline-recall 60% — not a bad threshold value. Top-1 score distributions:

| model | answerable top-1 (p25 / med) | unanswerable top-1 (p25 / med / max) |
|---|---|---|
| ms-marco L-6 | 0.535 / 2.518 | -3.310 / **-2.797** / 2.860 |
| bge-v2-m3 | 0.841 / 0.923 | 0.220 / 0.441 / 0.933 |

Unanswerable top-1 has median **-2.797** against a `decline_threshold` of **-2.5**, so roughly half of unanswerable questions score above the gate and are never auto-declined. This is *not* threshold-tunable: answerable p25 (0.535) sits below unanswerable max (2.860), so any gate strict enough to catch the leakers false-declines a large slice of answerable questions — exactly the -3.0 → -2.5 oscillation already recorded in `config.yaml`. Both models leak 1/10 unanswerable above the answerable median, so **bge buys recall, not discrimination**.

**Confirmed end-to-end 2026-07-20 (this is the useful part).** A full deterministic 65/65 Ollama eval with `rerank_candidates: 20` reproduced the previous baseline almost exactly — accuracy **70.0% → 70.0%**, decline recall **60.0% → 60.0%**, citation precision **25.07% → 25.17%**, and *every* per-category accuracy identical to three decimals (ambiguous 0.333, exact_term 0.722, factual_lookup 0.849, multi_hop 0.429, unanswerable 0.600). p95 122.8 s vs 123.7 s, fast-path ratio unchanged at 0.646.

That null is the evidence, not a disappointment: pool recall rose 78% → 84% and **nothing downstream moved at all**, because the extra chunks entering the candidate pool are re-buried by the cross-encoder before they ever reach generation. Retrieval breadth is not the constraint — the reranker is. Any future work that tunes `bm25_top_k` / `vector_top_k` / `rerank_candidates` and reports a metric change should be treated as suspect until the reranker itself changes.

**Fix (scoped, none of it cheap — this is a "know the ceiling" entry, not a quick win):**
- Accept ~60% final recall as the local-CPU ceiling and stop tuning retrieval knobs against it.
- If a GPU is ever free: re-time `bge-reranker-v2-m3` on CUDA; +11pp recall is worth real money if it fits the latency budget there.
- Decline routing needs a signal that isn't the cross-encoder score (the LLM grader already exists for this) — treat `decline_threshold` as a coarse pre-filter, not the decision.
- Do not read a citation-precision change after this commit as a retrieval win: `rerank_candidates` 15 → 20 changes the *pool*, not what generation sees.
