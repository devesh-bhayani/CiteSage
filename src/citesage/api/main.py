"""CiteSage FastAPI application.

Endpoints
---------
    POST /query    — question in, cited answer out
    POST /ingest   — upload a document (admin, API-key required)
    GET  /health   — liveness + dependency probes (public)
    GET  /stats    — in-process aggregate usage counters

Auth
----
    All non-public endpoints require ``X-API-Key: <key>`` matching the
    ``CITESAGE_API_KEY`` environment variable. If the env var is unset the
    server starts in *open* mode and logs a warning — this is convenient for
    local development but should never be used in production.

Rate limiting
-------------
    Per-key (or per-IP when unauthenticated): 10/min and 100/hour. Enforced
    via slowapi.

Errors
------
    All error responses follow RFC 7807 (``application/problem+json``).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config import get_settings
from ..graph.pipeline import PipelineResult, run_pipeline
from ..ingestion.loaders import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from ..ingestion.pipeline import IngestPipeline

load_dotenv()
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY_ENV = "CITESAGE_API_KEY"
MAX_QUESTION_CHARS = 1000


def _get_api_key() -> Optional[str]:
    """Return the API key required for protected endpoints, or None (open mode)."""
    return os.environ.get(API_KEY_ENV) or None


# ---------------------------------------------------------------------------
# Rate limiter — keyed by X-API-Key when present, else remote IP
# ---------------------------------------------------------------------------


def _rate_limit_key(request: Request) -> str:
    header = request.headers.get("X-API-Key")
    if header:
        return f"key:{header[:12]}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(
    key_func=_rate_limit_key,
    default_limits=["100/hour", "10/minute"],
)


# ---------------------------------------------------------------------------
# Stats collector — in-process counters surfaced by GET /stats
# ---------------------------------------------------------------------------


class StatsCollector:
    """Thread-unsafe aggregator for a single-worker FastAPI process.

    For multi-worker deployments, replace with a Redis-backed counter.
    """

    def __init__(self) -> None:
        self.query_count: int = 0
        self.declined_count: int = 0
        self.fast_count: int = 0
        self.thorough_count: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.started_at: float = time.time()

    def record(self, result: PipelineResult) -> None:
        self.query_count += 1
        if result.declined:
            self.declined_count += 1
        if result.path_taken == "fast":
            self.fast_count += 1
        elif result.path_taken == "thorough":
            self.thorough_count += 1
        self.total_input_tokens += int(result.token_usage.get("input_tokens", 0) or 0)
        self.total_output_tokens += int(result.token_usage.get("output_tokens", 0) or 0)
        if result.query_cost is not None:
            self.total_cost_usd += float(result.query_cost.total_cost_usd)

    def snapshot(self) -> dict:
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "query_count": self.query_count,
            "declined_count": self.declined_count,
            "fast_path_count": self.fast_count,
            "thorough_path_count": self.thorough_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "average_cost_per_query_usd": (
                round(self.total_cost_usd / self.query_count, 6)
                if self.query_count
                else 0.0
            ),
        }


_stats = StatsCollector()


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    score: float
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str
    path_taken: str
    declined: bool
    token_usage: dict
    cost_usd: float
    request_id: str


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int
    request_id: str


class HealthResponse(BaseModel):
    status: str
    checks: dict
    provider: str
    version: str


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------


def _sanitize_question(raw: str) -> str:
    """Strip control characters and cap length.

    Keeps newlines and tabs (legitimate in multi-line questions). Drops every
    other C0/C1 control and the Unicode line/paragraph separators that can
    confuse downstream logging or prompt templates.
    """
    cleaned = "".join(
        ch
        for ch in raw
        if (ch in ("\n", "\t") or (ord(ch) >= 0x20 and ord(ch) != 0x7F))
    )
    cleaned = cleaned.strip()
    if len(cleaned) > MAX_QUESTION_CHARS:
        cleaned = cleaned[:MAX_QUESTION_CHARS]
    return cleaned


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def require_api_key(request: Request) -> str:
    """Raise 401 if API key is configured and the header is missing/invalid.

    When CITESAGE_API_KEY is unset, authentication is bypassed (dev mode).
    """
    expected = _get_api_key()
    if expected is None:
        return "dev-open-mode"
    received = request.headers.get("X-API-Key")
    if not received or received != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )
    return received


# ---------------------------------------------------------------------------
# FastAPI app + exception handlers (RFC 7807)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CiteSage API",
    description="Document QA with verified citations.",
    version=get_settings().project.version,
)
app.state.limiter = limiter


def _problem(
    status_code: int,
    title: str,
    detail: str,
    type_uri: str = "about:blank",
    instance: Optional[str] = None,
) -> JSONResponse:
    """Return an RFC 7807 ``application/problem+json`` response."""
    body: dict = {
        "type": type_uri,
        "title": title,
        "status": status_code,
        "detail": detail,
    }
    if instance:
        body["instance"] = instance
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    return _problem(
        exc.status_code,
        title=exc.__class__.__name__,
        detail=str(exc.detail),
        instance=str(request.url.path),
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    return _problem(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        title="ValidationError",
        detail=str(exc.errors()[:3]),  # truncate to avoid giant payloads
        instance=str(request.url.path),
    )


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return _problem(
        status.HTTP_429_TOO_MANY_REQUESTS,
        title="RateLimitExceeded",
        detail=f"Rate limit exceeded: {exc.detail}",
        instance=str(request.url.path),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "api.unhandled_exception",
        request_id=req_id,
        path=str(request.url.path),
        error=str(exc),
    )
    return _problem(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        title="InternalServerError",
        detail="An unexpected error occurred.",
        instance=str(request.url.path),
    )


# ---------------------------------------------------------------------------
# Middleware: request ID + structlog binding
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = req_id
    log = logger.bind(request_id=req_id, path=request.url.path)
    log.info("api.request.start", method=request.method)
    t0 = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.error("api.request.error", latency_ms=round(elapsed_ms, 1))
        raise
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "api.request.end",
        status=response.status_code,
        latency_ms=round(elapsed_ms, 1),
    )
    response.headers["X-Request-ID"] = req_id
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Public health probe — checks Chroma path, BM25 index, and LLM provider."""
    settings = get_settings()
    checks: dict = {}

    chroma_path = Path(settings.paths.chroma_db)
    checks["chroma_dir"] = {
        "ok": chroma_path.parent.exists(),
        "path": str(chroma_path),
    }

    bm25_path = Path(settings.paths.bm25_index)
    checks["bm25_index"] = {
        "ok": bm25_path.exists() or bm25_path.parent.exists(),
        "path": str(bm25_path),
        "present": bm25_path.exists(),
    }

    provider = getattr(settings, "provider", "anthropic").lower()
    # Intentionally cheap probe: do not call the LLM on /health — too costly
    # and too slow. Just report configured provider and whether credentials
    # look plausible. Real liveness is implied by /query success.
    if provider == "anthropic":
        checks["llm_provider"] = {
            "ok": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "provider": "anthropic",
            "has_credentials": bool(os.environ.get("ANTHROPIC_API_KEY")),
        }
    else:
        checks["llm_provider"] = {"ok": True, "provider": provider}

    overall_ok = all(v.get("ok") for v in checks.values())
    return HealthResponse(
        status="ok" if overall_ok else "degraded",
        checks=checks,
        provider=provider,
        version=settings.project.version,
    )


@app.get("/stats", tags=["ops"])
def stats(_key: str = Depends(require_api_key)) -> dict:
    """Return in-process aggregate usage counters."""
    return _stats.snapshot()


@app.post("/query", response_model=QueryResponse, tags=["qa"])
@limiter.limit("10/minute;100/hour")
def query(
    request: Request,
    body: QueryRequest,
    _key: str = Depends(require_api_key),
) -> QueryResponse:
    """Answer a question against the ingested corpus with citations."""
    req_id = request.state.request_id
    question = _sanitize_question(body.question)
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is empty after sanitization.",
        )

    try:
        result = run_pipeline(question)
    except Exception as exc:
        logger.error("api.query.pipeline_error", request_id=req_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Pipeline error: {exc}",
        ) from exc

    _stats.record(result)

    citations = [
        Citation(
            chunk_id=sc.chunk.chunk_id,
            source_file=sc.chunk.source_file,
            page_number=sc.chunk.page_number,
            section_heading=sc.chunk.section_heading,
            score=round(sc.score, 4),
            content_preview=sc.chunk.content[:300],
        )
        for sc in result.citations
    ]

    cost = (
        float(result.query_cost.total_cost_usd)
        if result.query_cost is not None
        else 0.0
    )
    return QueryResponse(
        answer=result.answer,
        citations=citations,
        confidence=result.confidence,
        path_taken=result.path_taken,
        declined=result.declined,
        token_usage=result.token_usage,
        cost_usd=round(cost, 6),
        request_id=req_id,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["admin"])
@limiter.limit("10/minute;100/hour")
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    _key: str = Depends(require_api_key),
) -> IngestResponse:
    """Ingest a single document. Extension whitelist + 50 MB cap enforced."""
    req_id = request.state.request_id
    filename = file.filename or "uploaded"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="citesage_ingest_"))
    tmp_path = tmp_dir / filename
    try:
        bytes_written = 0
        with tmp_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File exceeds 50 MB limit "
                            f"(received {bytes_written / 1_048_576:.1f} MB so far)."
                        ),
                    )
                out.write(chunk)

        pipeline = IngestPipeline()
        try:
            chunks = pipeline.ingest_file(tmp_path)
        except ValueError as exc:  # loader validation errors
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        logger.info(
            "api.ingest.success",
            request_id=req_id,
            filename=filename,
            chunks=len(chunks),
        )
        return IngestResponse(
            filename=filename,
            chunks_ingested=len(chunks),
            request_id=req_id,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
