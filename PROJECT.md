# PROJECT.md — CiteSage

*Knowledge-transfer document. Written 2026-06-15 after a full codebase read at commit `9923c3a`. Read this before touching anything.*

## What this is

CiteSage is a **document question-answering system that refuses to hallucinate**. You ingest documents (PDF/MD/HTML/TXT), ask questions, and get answers where every claim carries a `[Source N]` citation pointing to a real chunk of a real file — or an explicit decline when the corpus can't support an answer ("cite-or-decline").

It is a **portfolio project** by devesh-bhayani demonstrating production-grade RAG engineering: hybrid retrieval, score-based routing, citation verification, cost tracking, and a real evaluation harness. It is not deployed anywhere; it runs locally.

## Tech stack and why

| Piece | Why |
|---|---|
| Python 3.11+, `uv` | Modern packaging; `uv.lock` is committed |
| **LangGraph** | The query pipeline is a state machine with conditional routing (fast/thorough/decline paths + a retry loop). LangGraph gives that shape natively |
| **ChromaDB** (persistent, cosine) | Local vector store, zero infra |
| **rank_bm25** (pickled to `data/bm25_index.pkl`) | Keyword recall to complement dense vectors; exact-term queries (numbers, names, formulas) fail on embeddings alone |
| **sentence-transformers** | `all-MiniLM-L6-v2` embeddings + `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, both tiny/CPU-friendly |
| **tiktoken** | Token-based chunking (600 tokens, 100 overlap) — a hard project rule, never character-based |
| **Provider abstraction** (`utils/llm_factory.py`) | Same pipeline runs on Anthropic (Sonnet generator + Haiku grader) or local Ollama (`qwen3:8b` + `qwen3-small`). Switch = one line in `config.yaml` |
| FastAPI + slowapi | API with key auth, rate limiting, RFC 7807 errors |
| Streamlit (`ui/app.py`) | Thin demo front-end over the API |
| structlog | Every stage logs structured events (`graph.retrieve`, `pipeline.complete`, …) |
| pytest (149 tests) | Unit + integration; LLM calls stubbed in integration tests |

## Architecture

```
INGESTION (offline)
  file → loaders.py (whitelist .pdf/.md/.html/.txt, 50MB cap)
       → chunker.py (tiktoken, SHA-256 chunk IDs → idempotent re-ingest)
       → storage.py: ChromaDB upsert  +  BM25 pickle save

QUERY (LangGraph state machine — graph/pipeline.py, graph/nodes.py)
  question
    → retrieve   (BM25 top-20 + vector top-20 → RRF fuse k=60 → top 15)
    → rerank     (cross-encoder scores ALL candidates, no threshold here)
    → route on top score:                        [config.yaml thresholds]
        ≥ confidence_threshold (0.8) → FAST:     1 generator call → verify
        < decline_threshold  (-2.5)  → DECLINE:  canned answer, 0 LLM calls
        in between                   → THOROUGH: grade_relevance (grader LLM
                                       filters chunks) → generate → verify
                                       → if confidence "low" & retry_count<1:
                                         transform_query (rewrite) → retrieve
  → PipelineResult {answer, citations, confidence, path_taken, cost}
```

**Citation verification** (`generation/citation_verifier.py`) is hybrid: deterministic token-overlap check first (threshold 0.55); only "weak" citations escalate to a per-citation LLM judge (YES/NO/PARTIAL). Runs on **both** paths — skipping it on the fast path was once the top cause of bad citation precision.

**Cost/usage** flows through `RAGState.model_usage` → `utils/cost_tracker.py` → logged per query and aggregated by the API `/stats` endpoint.

**Evaluation** (`evaluation/run_eval.py`, ~1000 lines): runs the 65-item golden dataset (`tests/eval/golden_dataset.json`, 5 categories, hand-verified) through the real pipeline, grades answers with an LLM judge, computes citation precision `|cited ∩ expected| / |cited|`, decline recall, latency percentiles, cost. `check_regression.py` diffs two reports (5% tolerance). Reports live in `reports/`.

## Key design decisions (inferred, with reasoning)

1. **Routing is by reranker score, not an LLM's opinion** — deterministic, free, and logged. The three thresholds in `config.yaml` are the pipeline's control surface.
2. **All heavyweight objects are constructed inside node functions, made safe by process-level caches.** `retrieve_node` builds a fresh `Retriever()` per query; the actual models are memoized (`_load_cross_encoder`, `_load_embedder` via `lru_cache` — commit `83bee9f`). This caching **fixed an OOM that killed eval runs at ~query 20** (CrossEncoder reloaded 6×/query before). Don't remove it.
3. **Ollama gets a hand-rolled client wrapper** (`OllamaLLM` in `llm_factory.py`) instead of `ChatOllama`, because langchain-ollama silently strips `<think>` output from qwen3-family models. The wrapper passes `think=False`, strips residual `</think>` blocks, and **deliberately does not pass an options dict** (it forced model reloads). Consequence: `max_tokens`/`num_predict` is ignored on Ollama.
4. **180 s Ollama timeout** (`CITESAGE_OLLAMA_TIMEOUT`, commit `74966aa`): generator and grader are different models, so every query forces a model swap in Ollama; cold reloads blew the old 60 s timeout and killed runs. Dead-daemon detection is unaffected (connection-refused is instant).
5. **Grader JSON parse failure → keep nothing** (`_parse_grade_indices` returns `[]` → decline). Safe-by-default: the grader is the last defense on unanswerable questions.
6. **Prompts are versioned YAML** (`prompts/v1/*.yaml`, `load_prompt()` with lru_cache). Never inline strings — hard rule.
7. **Chunk IDs are SHA-256(text + source_file)** → re-ingesting is idempotent, and the golden dataset can reference stable chunk IDs.
8. **Eval exits 1 when targets are missed** — it's a CI gate by design. Exit 1 ≠ crash. Check the printed summary/log tail before assuming failure.

## Current quality status (be honest with yourself)

- All 149 tests pass. The pipeline runs end-to-end on Ollama, 65/65 eval queries, no crashes.
- **All 4 Phase-3 eval targets fail on Ollama** (latest committed baseline `reports/baseline_ollama.json`: accuracy 64.6% vs ≥85%, citation precision 36.9% vs ≥90%, decline recall 70% vs ≥85%, p95 122 s vs <5 s). This is a local-model ceiling; the intended "green" run uses the Anthropic provider and has never been executed (blocked on API credits).
- **Ollama decoding is now pinned** (temperature=0, seed=42) so eval runs are reproducible (GAPS.md #1, fixed). Baselines recorded before that fix carry a ±6–10 pp noise floor and aren't comparable to post-pinning runs.
- **The corpus is now 16 documents / 34 chunks** (GAPS.md #2, fixed 2026-07-17) — 15 original ML/DL notes added alongside the original `transformer_architecture.md` to give retrieval real distractors. This surfaced a genuine finding: the golden set's expected chunk reaches the reranker's candidate pool only 80% of the time and survives to the final top-5 only 58% of the time — retrieval difficulty that a 4-chunk corpus could never reveal. `reports/baseline_ollama.json` still reflects the *old* 4-chunk corpus; no eval has been run against the expanded one yet.

## Critical paths (load-bearing — change with care)

- `graph/nodes.py` + `graph/pipeline.py` — the whole query flow and routing thresholds.
- `utils/llm_factory.py` — provider switch, Ollama quirk handling, timeout. Subtle; every behavior in it exists because something broke.
- `retrieval/reranker.py` / `ingestion/storage.py` — contain the `lru_cache` model loaders that prevent the OOM.
- `ingestion/chunker.py` — chunk IDs feed the golden dataset; changing chunking **silently invalidates all 65 expected-chunk references**.
- `config.yaml` thresholds — tiny numeric edits swing eval metrics by tens of points.
- `evaluation/run_eval.py` metric definitions — citation precision was once mis-measured (boolean-subset averaged); the current formula matches the Phase-3 spec. Don't "simplify" it back.

Safe to change casually: `ui/app.py`, `cli.py` output formatting, `scripts/compare_retrieval.py`, docs.

## Non-obvious gotchas

- `config.yaml` is found by **walking up from cwd** (`config/__init__.py`), overridable via `CITESAGE_CONFIG`. Tests chdir to project root in `conftest.py` and clear the settings cache per test.
- `get_settings()` is `lru_cache`d — config edits need a process restart (or `get_settings.cache_clear()`).
- First query in a process is slow: sentence-transformers + cross-encoder + Chroma all cold-load (~30–60 s on this machine).
- On Windows: git warns CRLF on touched files (harmless); black needs `--target-version py311`; pre-commit runs ruff/black/mypy on commit and **will block** commits.
- `python -m citesage.evaluation.run_eval` supports `--subset N` and `--category X` for cheap smoke runs — use them before a full 40–90 min Ollama run.
- The eval **overwrites** `--output`/`--errors` files only at the very end of the run; a killed run leaves the old files untouched.
- `generation/generator.py` (`Generator` class) is **dead code** — the LangGraph nodes reimplemented it. Don't extend it thinking it's live.
