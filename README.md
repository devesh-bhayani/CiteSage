# CiteSage

Document QA with **verified citations**. Ask a question against your own
corpus; get an answer where every claim links back to a specific chunk of a
specific source — or an honest decline when the corpus can't support one.

Built as a portfolio demonstration of production-grade RAG:

- **Hybrid retrieval** — BM25 + dense vectors, fused with Reciprocal Rank
  Fusion, then reranked by a cross-encoder.
- **Two-tier routing** — a fast single-LLM path for confident retrievals,
  and a slower grade → generate → verify loop when the top score is
  borderline. Routing is by score, not by a model's opinion.
- **Cite-or-decline** — the pipeline refuses to answer when retrieval can't
  back the question, rather than hallucinating.
- **Provider-agnostic** — swap `provider: anthropic` / `provider: ollama`
  in `config.yaml`. Same code path, same prompts, same metrics.
- **Instrumented** — per-query token and cost tracking, structured
  `structlog` output, a 65-item golden eval set, and a regression gate.

---

## Architecture

```
        ingest                             query
          │                                  │
          ▼                                  ▼
  chunk (tiktoken)              ┌──── retrieve (BM25 + vector) ──┐
          │                     │              │                 │
          ▼                     │              ▼                 │
  embed + metadata ──► ChromaDB │        RRF fuse + rerank       │
          │                     │              │                 │
          └─► BM25 pickle ──────┘              ▼                 │
                                   top score ≥ 0.8 ?             │
                                   │              │              │
                              yes  │              │  no          │
                                   ▼              ▼              ▼
                              generate      grade → generate → verify
                              (1 LLM call)      │           │      │
                                   │           ▼           ▼      │
                                   └──► cite or decline ◄──┘◄─────┘
                                             │
                                             ▼
                              answer  +  [Source N] citations
                              ( confidence, path_taken, cost )
```

Full pipeline description: [docs/architecture/overview.md](docs/architecture/overview.md).

---

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv pip install -e .
cp .env.example .env           # then set ANTHROPIC_API_KEY if using the cloud provider
```

### Ingest some documents

```bash
python -m citesage.cli --ingest data/documents/
```

### Ask a question

```bash
python -m citesage.cli "What is the self-attention formula?"
```

### Run the API

```bash
uvicorn citesage.api.main:app --reload
```

| Endpoint       | Purpose                                           |
| -------------- | ------------------------------------------------- |
| `POST /query`  | Question → cited answer                           |
| `POST /ingest` | Upload a `.pdf` / `.md` / `.html` / `.txt` file   |
| `GET /health`  | Chroma + BM25 + provider checks (public)          |
| `GET /stats`   | In-process token / cost / path-split counters     |

Set `CITESAGE_API_KEY` to require an `X-API-Key` header; leave unset for
open-mode local dev.

### Run the Streamlit UI

```bash
streamlit run src/citesage/ui/app.py
```

Three tabs: Query, Ingest, Stats. Points at the API URL set by
`CITESAGE_API_URL` (default `http://localhost:8000`).

---

## Provider configuration

`config.yaml` controls the provider. Both paths use the same prompts,
graph, and metrics:

```yaml
provider: "anthropic"   # or "ollama"

models:
  generator: "claude-sonnet-4-20250514"      # or "qwen3:8b" for Ollama
  grader:    "claude-haiku-4-5-20251001"     # or "qwen3-small:latest"
```

Ollama runs fully offline (free, slower). Anthropic hits the Phase 3
latency SLO (p95 < 5 s). Pick whichever matches your constraints.

---

## Evaluation

The golden dataset (`tests/eval/golden_dataset.json`) contains 65 manually
verified Q/A pairs across five categories: factual lookup, multi-hop,
unanswerable, ambiguous, and exact-term.

```bash
python -m citesage.evaluation.run_eval \
  --dataset tests/eval/golden_dataset.json \
  --output  reports/baseline_scores.json \
  --errors  reports/baseline_errors.jsonl
```

Phase 3 exit targets:

| Metric              | Target | How it's measured                                 |
| ------------------- | ------ | ------------------------------------------------- |
| Accuracy            | ≥ 85 % | LLM-judge grade over 65 queries                   |
| Citation precision  | ≥ 90 % | Mean `\|cited ∩ expected\| / \|cited\|` per query |
| Decline recall      | ≥ 85 % | Unanswerable questions correctly declined         |
| p95 latency         | < 5 s  | End-to-end, per query                             |

See [docs/phases/PHASE3_EVALUATION.md](docs/phases/PHASE3_EVALUATION.md) for
the full methodology and [reports/](reports/) for historical runs.

Regression gate (used in CI):

```bash
python -m citesage.evaluation.check_regression \
  --new reports/baseline_scores.json \
  --old reports/baseline_scores.json
```

---

## Development

```bash
uv sync
uv pip install -e .

# Lint + format
ruff check src/ --fix && black src/

# Tests
pytest tests/ -v
```

The test suite has three tiers:

- `tests/unit/` — deterministic unit tests (chunker, RRF, citation
  verifier, config, cost tracker)
- `tests/integration/` — graph routing + FastAPI surface with stubbed
  LLM calls
- `tests/eval/` — the golden dataset

---

## Project layout

```
src/citesage/
├── api/          FastAPI app (query, ingest, health, stats)
├── cli.py        Command-line entry point
├── config/       Pydantic-validated config.yaml loader
├── evaluation/   Golden-dataset runner, LLM judge, regression gate
├── generation/   Generator + citation verifier
├── graph/        LangGraph pipeline, nodes, state, routing
├── ingestion/    Loaders, tiktoken-based chunker, ChromaDB storage, BM25
├── prompts/      Versioned YAML prompts (no inline prompt strings)
├── retrieval/    BM25 + vector retriever, RRF fusion, cross-encoder rerank
├── ui/           Streamlit front-end
└── utils/        LLM factory (provider-agnostic), cost tracker
```

---

## Roadmap

- [ ] SSE streaming on `POST /query` via LangGraph `astream_events`
- [ ] Weekly non-VCR eval cron
- [ ] Query feedback loop (thumbs up/down → new golden entries)
- [ ] Multi-tenant corpus isolation

---

## License

MIT.
