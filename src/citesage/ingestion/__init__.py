"""CiteSage ingestion package.

Public surface:
    Document, Chunk          — domain models
    load_document            — auto-dispatch file loader
    PDFLoader, MarkdownLoader, HTMLLoader, TextLoader
    TokenChunker             — token-based splitter
    ChunkerFactory           — per-doc-type chunker cache
    ChromaStore              — ChromaDB vector store
    BM25Index                — keyword index (pickle-backed)
    IngestPipeline           — end-to-end orchestrator
"""

from .chunker import ChunkerFactory, TokenChunker
from .loaders import HTMLLoader, MarkdownLoader, PDFLoader, TextLoader, load_document
from .models import Chunk, Document
from .pipeline import IngestPipeline
from .storage import BM25Index, ChromaStore

__all__ = [
    "Document",
    "Chunk",
    "load_document",
    "PDFLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "TextLoader",
    "TokenChunker",
    "ChunkerFactory",
    "ChromaStore",
    "BM25Index",
    "IngestPipeline",
]
