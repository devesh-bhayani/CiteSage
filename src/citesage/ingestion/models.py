"""Domain models for the ingestion pipeline."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A loaded document page/section, prior to chunking."""

    content: str
    source_file: str
    doc_type: str  # "pdf" | "markdown" | "html" | "txt"
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk ready for indexing in ChromaDB and BM25."""

    chunk_id: str  # SHA-256(chunk_text + source_file) — idempotency key
    content: str
    source_file: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_index: int
    doc_type: str
    ingestion_timestamp: str  # ISO-8601 UTC
    content_hash: str  # SHA-256(chunk_text)
    token_count: int
