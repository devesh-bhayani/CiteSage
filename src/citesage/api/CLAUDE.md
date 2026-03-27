# API Rules
Auth: X-API-Key header. Per-key rate limit: 10/min, 100/hour. Audit log per request.
Upload safety: whitelist .pdf/.md/.html/.txt, MIME check, max 50MB, sandbox PDF parsing.
Prompt injection: sanitize input, strip control chars, limit 1000 chars, log suspicious patterns.
Streaming: SSE + LangGraph astream_events. Errors: RFC 7807, never expose traces.
Endpoints: POST /query, POST /ingest (admin), GET /health (public), GET /stats.
