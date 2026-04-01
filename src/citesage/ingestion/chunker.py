"""Token-based text chunker.

Critical rules (from ingestion CLAUDE.md):
- MUST use tiktoken length_function with RecursiveCharacterTextSplitter.
- NEVER pass chunk_size without length_function — that defaults to characters.
- Chunk ID = SHA-256(chunk_text + source_file) for idempotency.
- ChunkerFactory builds per-doc-type chunkers from config.yaml.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import ChunkingStrategyConfig, get_settings
from .models import Chunk, Document

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk_id(chunk_text: str, source_file: str) -> str:
    """Return SHA-256(chunk_text + source_file) as a hex string."""
    raw = (chunk_text + source_file).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_splitter(config: ChunkingStrategyConfig) -> RecursiveCharacterTextSplitter:
    """Construct a RecursiveCharacterTextSplitter with tiktoken length_function."""
    enc = tiktoken.get_encoding("cl100k_base")
    return RecursiveCharacterTextSplitter(
        chunk_size=config.size,
        chunk_overlap=config.overlap,
        length_function=lambda text: len(enc.encode(text)),
        separators=["\n\n", "\n", ". ", " "],
    )


# ---------------------------------------------------------------------------
# TokenChunker
# ---------------------------------------------------------------------------


class TokenChunker:
    """Split a Document into token-sized Chunks.

    All size/overlap values come from a ChunkingStrategyConfig so that
    chunking behaviour is fully driven by config.yaml.
    """

    def __init__(self, config: ChunkingStrategyConfig) -> None:
        self._config = config
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._splitter = _build_splitter(config)

    def chunk(self, document: Document) -> list[Chunk]:
        """Return a list of Chunks for *document*.

        Empty documents return an empty list.
        """
        if not document.content.strip():
            return []

        splits = self._splitter.split_text(document.content)
        timestamp = datetime.now(timezone.utc).isoformat()
        chunks: list[Chunk] = []

        for idx, text in enumerate(splits):
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_id = _make_chunk_id(text, document.source_file)
            token_count = len(self._enc.encode(text))

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content=text,
                    source_file=document.source_file,
                    page_number=document.page_number,
                    section_heading=document.section_heading,
                    chunk_index=idx,
                    doc_type=document.doc_type,
                    ingestion_timestamp=timestamp,
                    content_hash=content_hash,
                    token_count=token_count,
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# ChunkerFactory
# ---------------------------------------------------------------------------


class ChunkerFactory:
    """Vend TokenChunker instances keyed by doc_type.

    Chunkers are cached after first creation so tiktoken encoding is
    initialised only once per doc_type per process.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cache: dict[str, TokenChunker] = {}

    def get_chunker(self, doc_type: str) -> TokenChunker:
        """Return (and cache) the TokenChunker for *doc_type*."""
        if doc_type not in self._cache:
            config = self._settings.chunking.for_doc_type(doc_type)
            self._cache[doc_type] = TokenChunker(config)
        return self._cache[doc_type]
