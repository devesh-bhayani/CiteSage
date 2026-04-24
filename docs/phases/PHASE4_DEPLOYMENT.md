# Phase 4: Deployment — API, UI, Monitoring

## Goal
Ship the CiteSage pipeline behind a documented HTTP surface and a
browsable UI, with per-request tracing and usage counters.

## Built
1. **FastAPI application** at
   [src/citesage/api/main.py](../../src/citesage/api/main.py) with four
   endpoints:
   - `POST /query` — wraps `run_pipeline`, returns answer + citations +
     confidence + path + cost.
   - `POST /ingest` — multipart upload, extension whitelist, 50 MB cap,
     streams to a temp file, calls `IngestPipeline.ingest_file`.
   - `GET /health` — public; checks Chroma path, BM25 index, and
     provider credentials.
   - `GET /stats` — in-process counters (queries, declines, fast/thorough
     split, total tokens, total cost).
2. **Security** — X-API-Key auth (bypassed when `CITESAGE_API_KEY` unset,
   for dev), slowapi rate limit 10/min + 100/hour keyed by API key or
   client IP, input sanitization (strip C0/C1 control chars, 1000-char
   question cap), RFC 7807 error shape, per-request `X-Request-ID` bound
   into structlog.
3. **Streamlit UI** at
   [src/citesage/ui/app.py](../../src/citesage/ui/app.py) with Query,
   Ingest, and Stats tabs. Talks to the API over HTTP; reads
   `CITESAGE_API_URL` / `CITESAGE_API_KEY` from the environment.
4. **Integration tests** at
   [tests/integration/test_api.py](../../tests/integration/test_api.py):
   health shape, query happy/declined/error paths, control-char
   sanitization, stats accounting, ingest whitelist, auth enforcement.

## Done When
- ✓ `uvicorn citesage.api.main:app` boots and `curl /health` returns 200
- ✓ `streamlit run src/citesage/ui/app.py` loads and can round-trip a query
- ✓ `pytest tests/integration/test_api.py` passes
- ✓ RFC 7807 error shape on 4xx/5xx
- ✓ Non-auth endpoints (/health) reachable without key even when auth
  is enabled

## Stretch (follow-up)
- SSE streaming on `POST /query` via LangGraph `astream_events`
- Persistent stats (Redis) for multi-worker deployments
- Weekly non-VCR eval cron per the evaluation CLAUDE.md rules
