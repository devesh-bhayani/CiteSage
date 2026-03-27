---
name: eval-debug
description: Debug evaluation failures step by step
---
1. Load failing test case from golden_dataset.json
2. Run retrieval independently: check BM25, vector, RRF
3. Check dedup — same chunk wasting reranker slots?
4. Inspect reranker scores — relevant chunk in top 5?
5. Check which path was taken (fast vs thorough)
6. If retrieval good, run generation independently
7. Run deterministic citation check, then LLM verifier
8. Diagnose: chunking, prompt, threshold, or data issue?
9. Add failure case to regression tests
