"""Unit tests for TokenChunker and ChunkerFactory.

Key invariants verified:
1. Every chunk's token_count field matches the actual tiktoken count.
2. No chunk exceeds the configured token limit (size).
3. Character length CAN exceed the token limit — confirming token-based splitting.
4. Chunk IDs are deterministic (same input → same ID).
5. Different source files produce different IDs for identical content.
6. chunk_index is sequential starting at 0.
7. All metadata fields are populated.
8. Empty documents produce zero chunks.
9. ChunkerFactory returns the same instance for the same doc_type (caching).
10. ChunkerFactory falls back to default config for unknown doc_types.
"""

from __future__ import annotations

import hashlib

import pytest
import tiktoken

from citesage.config import ChunkingStrategyConfig
from citesage.ingestion.chunker import ChunkerFactory, TokenChunker, _make_chunk_id
from citesage.ingestion.models import Document


# ---------------------------------------------------------------------------
# Shared encoding (module-level to avoid repeated init overhead)
# ---------------------------------------------------------------------------

ENC = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_config() -> ChunkingStrategyConfig:
    """50-token chunk size for tests that need many chunks."""
    return ChunkingStrategyConfig(size=50, overlap=10, strategy="recursive")


@pytest.fixture()
def default_config() -> ChunkingStrategyConfig:
    """600-token chunk size matching config.yaml defaults."""
    return ChunkingStrategyConfig(size=600, overlap=100, strategy="recursive")


@pytest.fixture()
def short_doc() -> Document:
    return Document(
        content="Hello world. This is a short test document.",
        source_file="tests/fixtures/short.md",
        doc_type="markdown",
    )


@pytest.fixture()
def long_doc() -> Document:
    """~400-token document: 'hello' × 400 (each 'hello' is 1 token)."""
    return Document(
        content=" ".join(["hello"] * 400),
        source_file="tests/fixtures/long.md",
        doc_type="markdown",
    )


@pytest.fixture()
def unicode_doc() -> Document:
    """Document with multi-byte characters to stress token vs char counting."""
    # Each emoji is 1 token but 2+ bytes.  100 repetitions = ~100 tokens but > 100 chars.
    content = "The café serves crêpes. " * 30
    return Document(
        content=content,
        source_file="tests/fixtures/unicode.md",
        doc_type="markdown",
    )


# ---------------------------------------------------------------------------
# TokenChunker — token count correctness
# ---------------------------------------------------------------------------


class TestTokenCounts:
    def test_token_count_field_matches_tiktoken(self, small_config, long_doc):
        """chunk.token_count must equal len(enc.encode(chunk.content))."""
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(long_doc)
        assert chunks, "Expected at least one chunk from long_doc"
        for chunk in chunks:
            actual = len(ENC.encode(chunk.content))
            assert chunk.token_count == actual, (
                f"chunk[{chunk.chunk_index}]: stored token_count={chunk.token_count} "
                f"but tiktoken counted {actual}"
            )

    def test_no_chunk_exceeds_token_limit(self, small_config, long_doc):
        """Every chunk must fit within the configured token limit."""
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(long_doc)
        for chunk in chunks:
            assert chunk.token_count <= small_config.size, (
                f"chunk[{chunk.chunk_index}] has {chunk.token_count} tokens, "
                f"exceeds limit of {small_config.size}"
            )

    def test_splitting_is_token_based_not_character_based(self, small_config):
        """Character length of a chunk must be ABLE to exceed the token limit.

        With size=50 and 'hello' (5 chars, 1 token) repeated 400 times:
        - Token-based split → ~50 'hello' tokens → ~249 characters per chunk.
        - Character-based split → ≤50 characters per chunk (≈ 10 words max).

        We assert that at least one chunk has more characters than the token
        limit, which is impossible under character-based splitting.
        """
        doc = Document(
            content=" ".join(["hello"] * 400),
            source_file="test.md",
            doc_type="markdown",
        )
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(doc)
        assert chunks

        oversized = [c for c in chunks if len(c.content) > small_config.size]
        assert oversized, (
            "No chunk had more characters than the token limit. "
            "This suggests character-based splitting was used instead of token-based."
        )

    def test_unicode_token_count_correct(self, small_config, unicode_doc):
        """Token counts must be correct even for non-ASCII content."""
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(unicode_doc)
        for chunk in chunks:
            actual = len(ENC.encode(chunk.content))
            assert chunk.token_count == actual


# ---------------------------------------------------------------------------
# TokenChunker — determinism and idempotency
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_produces_same_chunk_ids(self, small_config, short_doc):
        """Running the chunker twice on the same document must yield identical IDs."""
        chunker = TokenChunker(small_config)
        ids_a = [c.chunk_id for c in chunker.chunk(short_doc)]
        ids_b = [c.chunk_id for c in chunker.chunk(short_doc)]
        assert ids_a == ids_b

    def test_different_source_file_different_id(self):
        """SHA-256(text + source_a) ≠ SHA-256(text + source_b)."""
        text = "Identical content, different origin"
        id_a = _make_chunk_id(text, "file_a.pdf")
        id_b = _make_chunk_id(text, "file_b.pdf")
        assert id_a != id_b

    def test_chunk_id_formula(self):
        """chunk_id == SHA-256(chunk_text + source_file) in hex."""
        text = "some chunk text"
        source = "doc.pdf"
        expected = hashlib.sha256((text + source).encode("utf-8")).hexdigest()
        assert _make_chunk_id(text, source) == expected


# ---------------------------------------------------------------------------
# TokenChunker — metadata completeness
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_chunk_indices_are_sequential(self, default_config, long_doc):
        chunker = TokenChunker(default_config)
        chunks = chunker.chunk(long_doc)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_source_file_propagated(self, small_config, short_doc):
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(short_doc)
        for chunk in chunks:
            assert chunk.source_file == short_doc.source_file

    def test_doc_type_propagated(self, small_config, short_doc):
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(short_doc)
        for chunk in chunks:
            assert chunk.doc_type == short_doc.doc_type

    def test_ingestion_timestamp_present(self, small_config, short_doc):
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(short_doc)
        for chunk in chunks:
            assert chunk.ingestion_timestamp, "ingestion_timestamp must not be empty"

    def test_content_hash_is_sha256_of_content(self, small_config, short_doc):
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(short_doc)
        for chunk in chunks:
            expected = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            assert chunk.content_hash == expected

    def test_page_number_and_heading_propagated(self, small_config):
        doc = Document(
            content="Some content about neural networks in this chapter.",
            source_file="paper.pdf",
            doc_type="pdf",
            page_number=7,
            section_heading="Neural Networks",
        )
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.page_number == 7
            assert chunk.section_heading == "Neural Networks"


# ---------------------------------------------------------------------------
# TokenChunker — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_content_returns_empty_list(self, small_config):
        doc = Document(content="", source_file="empty.md", doc_type="markdown")
        chunker = TokenChunker(small_config)
        assert chunker.chunk(doc) == []

    def test_whitespace_only_returns_empty_list(self, small_config):
        doc = Document(content="   \n\t  ", source_file="ws.md", doc_type="markdown")
        chunker = TokenChunker(small_config)
        assert chunker.chunk(doc) == []

    def test_single_token_document(self, small_config):
        doc = Document(content="hello", source_file="tiny.md", doc_type="markdown")
        chunker = TokenChunker(small_config)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].token_count == 1
        assert chunks[0].chunk_index == 0

    def test_content_not_mutated(self, small_config, long_doc):
        """Chunker must not modify the original Document content."""
        original = long_doc.content
        chunker = TokenChunker(small_config)
        chunker.chunk(long_doc)
        assert long_doc.content == original


# ---------------------------------------------------------------------------
# ChunkerFactory
# ---------------------------------------------------------------------------


class TestChunkerFactory:
    def test_returns_token_chunker_instance(self):
        """Factory must return a TokenChunker for any doc_type."""
        factory = ChunkerFactory()
        chunker = factory.get_chunker("markdown")
        assert isinstance(chunker, TokenChunker)

    def test_same_doc_type_returns_same_object(self):
        """Factory must cache chunkers (identical object for repeated calls)."""
        factory = ChunkerFactory()
        c1 = factory.get_chunker("pdf")
        c2 = factory.get_chunker("pdf")
        assert c1 is c2

    def test_different_doc_types_different_objects(self):
        """Different doc_types should yield independent chunker instances."""
        factory = ChunkerFactory()
        c_pdf = factory.get_chunker("pdf")
        c_md = factory.get_chunker("markdown")
        # They are separate objects (even if config is same)
        assert c_pdf is not c_md

    def test_unknown_doc_type_falls_back_to_default(self):
        """An unrecognised doc_type should use the default chunking config."""
        from citesage.config import get_settings

        settings = get_settings()
        factory = ChunkerFactory()
        chunker = factory.get_chunker("unknown_type_xyz")
        assert isinstance(chunker, TokenChunker)
        assert chunker._config == settings.chunking.default
