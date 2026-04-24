"""Unit tests for the config loader.

Covers:
1. config.yaml loads into a fully-typed Settings object.
2. All expected fields are present with correct types.
3. ChunkingConfig.for_doc_type falls back to default for unknown types.
4. get_settings() is cached (same object returned).
5. Cache can be cleared and re-loaded.
6. Missing config file raises FileNotFoundError.
7. CITESAGE_CONFIG env var overrides path resolution.
8. vector_score_threshold and confidence_threshold are distinct.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from citesage.config import (
    Settings,
    get_settings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(path: Path) -> Path:
    """Write the smallest valid config.yaml to *path* and return the file path."""
    cfg = {
        "project": {"name": "test", "version": "0.0.0"},
        "models": {
            "generator": "claude-sonnet-test",
            "grader": "claude-haiku-test",
            "embedder": "all-MiniLM-L6-v2",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
        "chunking": {
            "default": {"size": 512, "overlap": 50, "strategy": "recursive"},
        },
        "retrieval": {
            "vector_top_k": 10,
            "bm25_top_k": 10,
            "rerank_top_k": 3,
            "rrf_k": 60,
            "vector_score_threshold": 0.3,
            "confidence_threshold": 0.7,
        },
        "prompts": {"version": "v1", "path": "src/citesage/prompts"},
        "paths": {
            "chroma_db": "data/chroma",
            "bm25_index": "data/bm25_index.pkl",
            "documents": "data/documents",
        },
    }
    cfg_path = path / "config.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    return cfg_path


# ---------------------------------------------------------------------------
# Tests: loading the real config
# ---------------------------------------------------------------------------


class TestRealConfig:
    def test_loads_without_error(self):
        """get_settings() must succeed with the project's config.yaml."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_project_fields(self):
        s = get_settings()
        assert s.project.name == "citesage"
        assert isinstance(s.project.version, str)

    def test_models_fields(self):
        s = get_settings()
        # generator and grader model names depend on the configured provider
        # (anthropic → "claude-*", ollama → e.g. "qwen3:8b"); the test only
        # asserts they are non-empty strings and the local-only models are
        # pinned to their expected values.
        assert isinstance(s.models.generator, str) and s.models.generator
        assert isinstance(s.models.grader, str) and s.models.grader
        assert s.models.embedder == "all-MiniLM-L6-v2"
        assert "cross-encoder" in s.models.reranker

    def test_chunking_default(self):
        s = get_settings()
        assert s.chunking.default.size == 600
        assert s.chunking.default.overlap == 100
        assert s.chunking.default.strategy == "recursive"

    def test_retrieval_fields_types(self):
        s = get_settings()
        assert isinstance(s.retrieval.vector_top_k, int)
        assert isinstance(s.retrieval.bm25_top_k, int)
        assert isinstance(s.retrieval.rerank_top_k, int)
        assert isinstance(s.retrieval.rrf_k, int)
        assert isinstance(s.retrieval.vector_score_threshold, float)
        assert isinstance(s.retrieval.confidence_threshold, float)

    def test_thresholds_are_distinct(self):
        """vector_score_threshold and confidence_threshold serve different roles."""
        s = get_settings()
        assert s.retrieval.vector_score_threshold < s.retrieval.confidence_threshold

    def test_paths_fields(self):
        s = get_settings()
        assert s.paths.chroma_db
        assert s.paths.bm25_index
        assert s.paths.documents

    def test_prompts_fields(self):
        s = get_settings()
        assert s.prompts.version == "v1"
        assert s.prompts.path


# ---------------------------------------------------------------------------
# Tests: caching behaviour
# ---------------------------------------------------------------------------


class TestCaching:
    def test_same_object_returned_on_repeat_calls(self):
        """get_settings() must return the same cached instance."""
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_cache_cleared_reloads(self):
        """After cache_clear(), get_settings() returns a fresh instance."""
        a = get_settings()
        get_settings.cache_clear()
        b = get_settings()
        # Different object (new load), but identical content
        assert a is not b
        assert a.project.name == b.project.name


# ---------------------------------------------------------------------------
# Tests: path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_env_var_override(self, tmp_path, monkeypatch):
        """CITESAGE_CONFIG env var must take precedence over cwd walk-up."""
        cfg_file = _write_minimal_config(tmp_path)
        monkeypatch.setenv("CITESAGE_CONFIG", str(cfg_file))
        get_settings.cache_clear()
        s = get_settings()
        assert s.project.name == "test"

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        """A path that doesn't exist must raise FileNotFoundError."""
        monkeypatch.setenv("CITESAGE_CONFIG", str(tmp_path / "nonexistent.yaml"))
        get_settings.cache_clear()
        with pytest.raises(FileNotFoundError):
            get_settings()


# ---------------------------------------------------------------------------
# Tests: ChunkingConfig.for_doc_type
# ---------------------------------------------------------------------------


class TestChunkingForDocType:
    def test_known_type_returns_default(self):
        """An unrecognised doc_type must return the default config."""
        s = get_settings()
        cfg = s.chunking.for_doc_type("pdf")
        assert cfg == s.chunking.default

    def test_unknown_type_returns_default(self):
        s = get_settings()
        cfg = s.chunking.for_doc_type("totally_unknown_type_xyz")
        assert cfg == s.chunking.default

    def test_per_type_override(self, tmp_path, monkeypatch):
        """A per-doc-type override in config must be returned when the type matches."""
        cfg = {
            "project": {"name": "t", "version": "0"},
            "models": {
                "generator": "g",
                "grader": "gr",
                "embedder": "e",
                "reranker": "r",
            },
            "chunking": {
                "default": {"size": 600, "overlap": 100, "strategy": "recursive"},
                "pdf": {"size": 400, "overlap": 50, "strategy": "recursive"},
            },
            "retrieval": {
                "vector_top_k": 10,
                "bm25_top_k": 10,
                "rerank_top_k": 3,
                "rrf_k": 60,
                "vector_score_threshold": 0.3,
                "confidence_threshold": 0.7,
            },
            "prompts": {"version": "v1", "path": "p"},
            "paths": {"chroma_db": "c", "bm25_index": "b", "documents": "d"},
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(cfg, fh)
        monkeypatch.setenv("CITESAGE_CONFIG", str(cfg_path))
        get_settings.cache_clear()
        s = get_settings()
        pdf_cfg = s.chunking.for_doc_type("pdf")
        assert pdf_cfg.size == 400
        assert pdf_cfg.overlap == 50
        # Markdown falls back to default
        md_cfg = s.chunking.for_doc_type("markdown")
        assert md_cfg.size == 600
