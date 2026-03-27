---
name: rag-component
description: Scaffold a new RAG pipeline component with tests and typing
---
When creating a new component:
1. Create in appropriate src/citesage/ subdirectory
2. Type hints on all functions (TypedDict for state, Pydantic for API)
3. Google-style docstring with Args, Returns, Example
4. Corresponding test in tests/unit/ (use pytest-vcr if LLM involved)
5. Structured logging with structlog
6. Retry + backoff on all external calls
7. Register in __init__.py
8. Run ruff + black + mypy after
