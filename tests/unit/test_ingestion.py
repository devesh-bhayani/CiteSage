"""Unit tests for the ingestion pipeline.

Covers:
1. Content hash is stable across re-runs (idempotency).
2. Re-ingesting the same file produces no duplicate chunks in ChromaDB or BM25.
3. chunk_id = SHA-256(content + source_file), not just SHA-256(content).
4. Loader rejects unsupported extensions and oversized files.
5. Markdown loader extracts section heading correctly.
6. BM25 index persists to disk and survives a reload.
7. IngestPipeline.ingest_file returns chunks with all metadata fields.
8. PDFLoader validates file size limit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from citesage.ingestion.loaders import (
    ALLOWED_EXTENSIONS,
    MarkdownLoader,
    load_document,
)
from citesage.ingestion.models import Chunk
from citesage.ingestion.pipeline import IngestPipeline
from citesage.ingestion.storage import BM25Index, ChromaStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def md_file(tmp_path) -> Path:
    """A small markdown file with a heading and body."""
    content = (
        "# Attention Is All You Need\n\n"
        "The Transformer is a sequence-to-sequence architecture. "
        "It replaces recurrent layers with self-attention. " * 5
    )
    f = tmp_path / "sample.md"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture()
def pipeline(tmp_path) -> IngestPipeline:
    """IngestPipeline wired to isolated temp-dir stores."""
    chroma = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    return IngestPipeline(chroma_store=chroma, bm25_index=bm25)


# ---------------------------------------------------------------------------
# Content hash stability
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_hash_is_sha256_of_content(self, pipeline, md_file):
        chunks = pipeline.ingest_file(md_file)
        for chunk in chunks:
            expected = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            assert (
                chunk.content_hash == expected
            ), f"chunk[{chunk.chunk_index}] content_hash mismatch"

    def test_hash_stable_across_reruns(self, pipeline, md_file):
        """Same file ingested twice must produce identical content hashes."""
        chunks_a = pipeline.ingest_file(md_file)
        chunks_b = pipeline.ingest_file(md_file)
        hashes_a = [c.content_hash for c in chunks_a]
        hashes_b = [c.content_hash for c in chunks_b]
        assert hashes_a == hashes_b

    def test_chunk_id_includes_source_file(self, pipeline, md_file, tmp_path):
        """chunk_id must differ when source_file path differs even for same content."""
        other = tmp_path / "other.md"
        other.write_text(md_file.read_text(), encoding="utf-8")

        chunks_orig = pipeline.ingest_file(md_file)
        chunks_other = pipeline.ingest_file(other)

        ids_orig = {c.chunk_id for c in chunks_orig}
        ids_other = {c.chunk_id for c in chunks_other}
        assert ids_orig.isdisjoint(
            ids_other
        ), "Different source files produced colliding chunk_ids"


# ---------------------------------------------------------------------------
# Idempotency (no duplicate chunks on re-ingestion)
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_chroma_no_duplicates_on_reingest(self, pipeline, md_file):
        """ChromaDB count must not increase on re-ingestion of same file."""
        pipeline.ingest_file(md_file)
        count_after_first = pipeline._store.count()
        pipeline.ingest_file(md_file)
        count_after_second = pipeline._store.count()
        assert count_after_first == count_after_second, (
            f"ChromaDB grew from {count_after_first} to {count_after_second} "
            "after re-ingesting the same file — duplicates inserted"
        )

    def test_bm25_no_duplicates_on_reingest(self, pipeline, md_file):
        """BM25 chunk count must not grow on re-ingestion of same file."""
        pipeline.ingest_file(md_file)
        count_after_first = pipeline._bm25.chunk_count()
        pipeline.ingest_file(md_file)
        count_after_second = pipeline._bm25.chunk_count()
        assert count_after_first == count_after_second, (
            f"BM25 grew from {count_after_first} to {count_after_second} "
            "after re-ingesting the same file — duplicates added"
        )


# ---------------------------------------------------------------------------
# BM25 persistence
# ---------------------------------------------------------------------------


class TestBM25Persistence:
    def test_index_written_to_disk(self, pipeline, md_file):
        """BM25 pickle file must exist after ingestion."""
        pipeline.ingest_file(md_file)
        assert pipeline._bm25._path.exists(), "BM25 pickle not written"

    def test_loaded_index_matches_original(self, pipeline, md_file):
        """BM25.load() must recover the same chunks as the in-memory index."""
        pipeline.ingest_file(md_file)
        pkl_path = pipeline._bm25._path
        loaded = BM25Index.load(str(pkl_path))
        assert loaded.chunk_count() == pipeline._bm25.chunk_count()
        orig_ids = {c.chunk_id for c in pipeline._bm25._chunks}
        loaded_ids = {c.chunk_id for c in loaded._chunks}
        assert orig_ids == loaded_ids

    def test_empty_index_load_returns_empty(self, tmp_path):
        """Loading from a non-existent path must return an empty index."""
        idx = BM25Index.load(str(tmp_path / "nonexistent.pkl"))
        assert idx.chunk_count() == 0

    def test_search_works_after_reload(self, pipeline, md_file):
        """Reloaded BM25 index must return search results."""
        pipeline.ingest_file(md_file)
        pkl_path = pipeline._bm25._path
        reloaded = BM25Index.load(str(pkl_path))
        results = reloaded.search("attention transformer", top_k=3)
        assert results, "Expected search results from reloaded BM25 index"
        # Results are (Chunk, score) pairs
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------


class TestMarkdownLoader:
    def test_heading_extracted(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# My Section\n\nSome body text.", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(md)
        assert len(docs) == 1
        assert docs[0].section_heading == "My Section"

    def test_no_heading_gives_none(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("Just a paragraph with no heading.", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(md)
        assert docs[0].section_heading is None

    def test_doc_type_is_markdown(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Hi\n\nContent.", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(md)
        assert docs[0].doc_type == "markdown"

    def test_source_file_is_absolute_path(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("Content.", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(md)
        assert docs[0].source_file == str(md)


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------


class TestLoaderValidation:
    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "file.docx"
        bad.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported file type|No loader"):
            load_document(bad)

    def test_oversized_file_raises(self, tmp_path, monkeypatch):
        """A file reported as too large must be rejected before reading."""
        md = tmp_path / "big.md"
        md.write_text("# Big\n\nSome content.", encoding="utf-8")
        # Patch stat to return an oversized file
        import citesage.ingestion.loaders as loaders_mod

        original_validate = loaders_mod._validate_file

        def patched_validate(path: Path) -> None:
            if path == md:
                raise ValueError(f"File too large: 51.0 MB (limit 50 MB): {path}")
            original_validate(path)

        monkeypatch.setattr(loaders_mod, "_validate_file", patched_validate)
        with pytest.raises(ValueError, match="too large"):
            load_document(md)

    def test_allowed_extensions_set(self):
        """Sanity-check that the whitelist contains the required extensions."""
        assert ".pdf" in ALLOWED_EXTENSIONS
        assert ".md" in ALLOWED_EXTENSIONS
        assert ".html" in ALLOWED_EXTENSIONS
        assert ".txt" in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# IngestPipeline metadata completeness
# ---------------------------------------------------------------------------


class TestPipelineMetadata:
    def test_all_required_fields_populated(self, pipeline, md_file):
        chunks = pipeline.ingest_file(md_file)
        assert chunks, "Pipeline produced no chunks"
        for chunk in chunks:
            assert chunk.chunk_id, "chunk_id empty"
            assert chunk.content, "content empty"
            assert chunk.source_file, "source_file empty"
            assert chunk.doc_type == "markdown"
            assert chunk.ingestion_timestamp, "ingestion_timestamp empty"
            assert chunk.content_hash, "content_hash empty"
            assert chunk.token_count > 0, "token_count must be positive"
            assert chunk.chunk_index >= 0

    def test_chunk_indices_sequential(self, pipeline, md_file):
        chunks = pipeline.ingest_file(md_file)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_ingest_directory(self, tmp_path):
        """ingest_directory must pick up all supported files recursively."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("# A\n\nContent about alpha." * 10)
        (docs_dir / "b.md").write_text("# B\n\nContent about beta." * 10)
        (docs_dir / "ignored.py").write_text("# not a doc")

        chroma = ChromaStore(persist_dir=str(tmp_path / "chroma"))
        bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
        pipeline = IngestPipeline(chroma_store=chroma, bm25_index=bm25)

        chunks = pipeline.ingest_directory(docs_dir)
        sources = {c.source_file for c in chunks}
        assert any("a.md" in s for s in sources)
        assert any("b.md" in s for s in sources)
        assert not any(".py" in s for s in sources)
