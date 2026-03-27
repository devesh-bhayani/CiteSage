# Retrieval Rules
Hybrid: BM25 top_k=20 + Vector top_k=20 → RRF fusion (k=60) → dedup → rerank top 15 → return top 5.
RRF: score(doc) = sum(1/(k + rank_i)). k configurable in config.yaml.
Dedup by chunk_id BEFORE reranking. Cross-encoder: ms-marco-MiniLM-L-6-v2.
Metadata filtering: ChromaDB where filters for vector. Post-filter BM25 before RRF.
Log per query: bm25_count, vector_count, overlap_count, rerank_scores, latency_ms.
