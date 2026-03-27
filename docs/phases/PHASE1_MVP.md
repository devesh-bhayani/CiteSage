# Phase 1: Working MVP (Week 1-2)
## Goal: CLI tool — ingest docs → query → cited answer
1. PDF + Markdown loaders with token-based chunking (tiktoken!)
2. ChromaDB storage with rich metadata + BM25 serialized to disk
3. Vector retrieval (top_k=10) + generation with [Source N] citations
4. Decline-to-answer when no relevant chunks found
5. CLI: python -m citesage.cli "What is X?"
## Done When:
- Ingest PDF + markdown, query, get cited answer
- System declines when chunks aren't relevant
- Chunks verified as token-sized (not character-sized)
- BM25 index persists across restarts
- All unit tests pass
