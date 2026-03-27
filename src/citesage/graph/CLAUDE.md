# LangGraph Rules
FAST PATH: retrieve → rerank → generate → END (3 nodes)
THOROUGH PATH: retrieve → rerank → grade → generate → verify → END (6 nodes, conditional rewrite loop)
Route by reranker score, not LLM call. Threshold 0.7 (configurable).
State: question, retrieved_chunks, reranked_chunks, reranker_top_score, answer, citations, confidence, path_taken, retry_count, error.
Guard rails: max 1 rewrite, max 1 regeneration. After max: confidence="low", not silent failure.
Latency targets (honest): fast p50 ~2s, thorough p50 ~5s. Measure first, optimize later.
