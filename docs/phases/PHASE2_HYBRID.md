# Phase 2: Hybrid Retrieval + LangGraph Pipeline

## Goal
Lift retrieval quality above the Phase 1 vector-only baseline and route
queries by retrieval confidence, not by an LLM's opinion.

## Built
1. **BM25 + vector retrieval** — both run independently at top_k=20.
   BM25 index serialized to `data/bm25_index.pkl` alongside ChromaDB.
2. **Reciprocal Rank Fusion** (k=60) deduplicates by `chunk_id` before
   fusion. See [src/citesage/retrieval/rrf.py](../../src/citesage/retrieval/rrf.py).
3. **Cross-encoder reranking** with `ms-marco-MiniLM-L-6-v2` on the top
   15 RRF candidates, returning the top 5. See
   [src/citesage/retrieval/reranker.py](../../src/citesage/retrieval/reranker.py).
4. **LangGraph two-tier pipeline** — see
   [src/citesage/graph/pipeline.py](../../src/citesage/graph/pipeline.py):
   - `top_score ≥ confidence_threshold` → **fast path** (one generator call)
   - `-3.0 < top_score < 0.8` → **thorough path** (grade → generate → verify, with one retry)
   - `top_score ≤ decline_threshold` → **decline** (zero LLM calls)
5. **Citation verifier** — hybrid token-overlap gate + LLM judge on weak
   matches. See [src/citesage/generation/citation_verifier.py](../../src/citesage/generation/citation_verifier.py).

## Done When
- ✓ `test_graph_routing.py` integration test passes
- ✓ All Phase 1 guarantees preserved (cite-or-decline, token chunking)
- ✓ Routing is score-driven and logged per query
- ✓ Provider-agnostic LLM factory in place
  ([src/citesage/utils/llm_factory.py](../../src/citesage/utils/llm_factory.py))
