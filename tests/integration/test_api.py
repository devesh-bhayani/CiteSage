"""Integration tests for the CiteSage FastAPI app.

The run_pipeline LLM calls are monkey-patched to a stub so these tests stay
fast and provider-agnostic. We are testing the HTTP surface (schemas, auth,
error shape, upload safety, stats accounting) — not the retrieval quality.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from citesage.graph.pipeline import PipelineResult
from citesage.ingestion.models import Chunk
from citesage.retrieval._types import ScoredChunk
from citesage.utils.cost_tracker import QueryCost

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_chunk(chunk_id: str = "chunk-1") -> ScoredChunk:
    ch = Chunk(
        chunk_id=chunk_id,
        content="Self-attention computes weighted sums of value vectors.",
        source_file="transformer.md",
        page_number=2,
        section_heading="Self-Attention",
        chunk_index=0,
        doc_type="md",
        ingestion_timestamp="2026-04-01T00:00:00Z",
        content_hash="abc",
        token_count=10,
    )
    return ScoredChunk(chunk=ch, score=0.92)


def _fake_pipeline_result(declined: bool = False) -> PipelineResult:
    return PipelineResult(
        answer=(
            "Declined — no relevant sources."
            if declined
            else "Self-attention uses Q, K, V projections [Source 1]."
        ),
        citations=[] if declined else [_fake_chunk()],
        confidence="low" if declined else "high",
        path_taken="declined" if declined else "fast",
        declined=declined,
        token_usage={"input_tokens": 100, "output_tokens": 20},
        query_cost=QueryCost(
            total_cost_usd=0.0012,
            total_input_tokens=100,
            total_output_tokens=20,
            model_breakdown=[],
        ),
    )


@pytest.fixture()
def client(monkeypatch):
    """TestClient with the pipeline stubbed and any API key cleared."""
    monkeypatch.delenv("CITESAGE_API_KEY", raising=False)

    # Import inside the fixture so env changes apply before module-level
    # _get_api_key is resolved. The app itself reads the env per-request.
    from citesage.api import main as api_main

    # Reset the stats collector so tests don't bleed into each other.
    api_main._stats.__init__()  # type: ignore[misc]

    def _fake_run(question: str) -> PipelineResult:
        declined = "does not exist" in question.lower()
        return _fake_pipeline_result(declined=declined)

    monkeypatch.setattr(api_main, "run_pipeline", _fake_run)

    return TestClient(api_main.app)


@pytest.fixture()
def authed_client(monkeypatch):
    """TestClient with API key auth enforced."""
    monkeypatch.setenv("CITESAGE_API_KEY", "secret-test-key")

    from citesage.api import main as api_main

    api_main._stats.__init__()  # type: ignore[misc]
    monkeypatch.setattr(
        api_main, "run_pipeline", lambda q: _fake_pipeline_result(False)
    )
    return TestClient(api_main.app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_is_public(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {"status", "checks", "provider", "version"}
        assert "chroma_dir" in body["checks"]
        assert "bm25_index" in body["checks"]
        assert "llm_provider" in body["checks"]

    def test_health_reports_configured_provider(self, client):
        body = client.get("/health").json()
        assert body["provider"] in ("anthropic", "ollama")


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_happy_path(self, client):
        r = client.post("/query", json={"question": "What is self-attention?"})
        assert r.status_code == 200
        body = r.json()
        assert body["declined"] is False
        assert body["path_taken"] == "fast"
        assert len(body["citations"]) == 1
        cite = body["citations"][0]
        assert cite["source_file"] == "transformer.md"
        assert cite["page_number"] == 2
        assert cite["section_heading"] == "Self-Attention"
        assert "request_id" in body

    def test_declined_path(self, client):
        r = client.post(
            "/query",
            json={
                "question": "What does the document say about something that does not exist?"
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["declined"] is True
        assert body["path_taken"] == "declined"
        assert body["citations"] == []

    def test_empty_question_rejected(self, client):
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_oversize_question_rejected(self, client):
        r = client.post("/query", json={"question": "x" * 1500})
        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_control_chars_stripped_but_accepted(self, client):
        # Bell + null interspersed should be stripped and the query still work.
        r = client.post(
            "/query",
            json={"question": "What\x00is\x07 self-attention?"},
        )
        assert r.status_code == 200

    def test_query_updates_stats(self, client):
        before = client.get("/stats").json()
        client.post("/query", json={"question": "What is X?"})
        after = client.get("/stats").json()
        assert after["query_count"] == before["query_count"] + 1
        assert after["fast_path_count"] == before["fast_path_count"] + 1
        assert after["total_input_tokens"] > before["total_input_tokens"]

    def test_pipeline_error_is_502(self, client, monkeypatch):
        from citesage.api import main as api_main

        def _boom(_q):
            raise RuntimeError("backend unreachable")

        monkeypatch.setattr(api_main, "run_pipeline", _boom)
        r = client.post("/query", json={"question": "anything"})
        assert r.status_code == 502
        assert r.headers["content-type"].startswith("application/problem+json")


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_rejects_unsupported_extension(self, client):
        r = client.post(
            "/ingest",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
        )
        assert r.status_code == 415
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_accepts_markdown(self, client, monkeypatch):
        # Stub IngestPipeline to avoid touching ChromaDB in tests.
        from citesage.api import main as api_main

        class _FakePipeline:
            def ingest_file(self, path):
                assert path.exists()
                return [object(), object(), object()]  # 3 chunks

        monkeypatch.setattr(api_main, "IngestPipeline", _FakePipeline)

        r = client.post(
            "/ingest",
            files={"file": ("note.md", b"# hello\n\ncontent", "text/markdown")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["filename"] == "note.md"
        assert body["chunks_ingested"] == 3

    def test_missing_file_returns_422(self, client):
        r = client.post("/ingest", files={})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_missing_key_rejected(self, authed_client):
        r = authed_client.post("/query", json={"question": "anything"})
        assert r.status_code == 401
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_wrong_key_rejected(self, authed_client):
        r = authed_client.post(
            "/query",
            json={"question": "anything"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_correct_key_accepted(self, authed_client):
        r = authed_client.post(
            "/query",
            json={"question": "anything"},
            headers={"X-API-Key": "secret-test-key"},
        )
        assert r.status_code == 200

    def test_health_open_even_with_auth_enabled(self, authed_client):
        r = authed_client.get("/health")
        assert r.status_code == 200
