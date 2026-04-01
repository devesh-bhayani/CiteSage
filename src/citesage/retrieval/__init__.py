"""CiteSage retrieval package.

Phase 2: hybrid BM25 + vector retrieval with RRF fusion and cross-encoder reranking.
"""

from .reranker import Reranker
from .retriever import Retriever
from ._types import RetrievalResult, ScoredChunk

__all__ = ["Retriever", "RetrievalResult", "ScoredChunk", "Reranker"]
