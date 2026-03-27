# CiteSage — Document QA with Verified Citations

## Tech Stack
- Python 3.11+, uv, LangGraph, ChromaDB, rank_bm25, sentence-transformers
- all-MiniLM-L6-v2 (embeddings), cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker)
- FastAPI, Streamlit, RAGAS, pytest + pytest-vcr

## Code Style
- Type hints on all functions. Google-style docstrings on public functions.
- Black, ruff, mypy. NEVER hardcode model names — use config.yaml.
- NEVER write inline prompt strings — use YAML in src/citesage/prompts/.

## Key Commands
- Lint: ruff check src/ --fix && black src/
- Test: pytest tests/ -v
- Eval: python -m citesage.evaluation.run_eval
- Serve: uvicorn citesage.api.main:app --reload

## Critical Rules
- YOU MUST cite-or-decline: never generate answers unsupported by retrieved context.
- YOU MUST use token-based chunking via tiktoken length_function, NOT character count.
- YOU MUST retry all LLM calls with exponential backoff (max 3, base 1s).
- Use Haiku for cheap tasks (routing, grading). Sonnet for generation only.
- All components (chunker, retriever, embedder, reranker) must be swappable via config.
- Log token usage per query.

## Compaction
Preserve: modified file list, current phase, active test failures, arch decisions.
