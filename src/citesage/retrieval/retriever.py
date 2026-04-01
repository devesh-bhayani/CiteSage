"""Hybrid retriever: BM25 + vector → RRF fusion → cross-encoder reranking.

Pipeline (per retrieval CLAUDE.md):
    BM25 top_k=20 + Vector top_k=20
        → RRF fusion (k=60)
        → dedup by chunk_id          ← implicit in rrf_fuse()
        → take top rerank_candidates (default 15) for the cross-encoder
        → rerank → return top rerank_top_k (default 5)

Logged per query: bm25_count, vector_count, overlap_count, rerank_scores,
latency_ms.

``ScoredChunk`` and ``RetrievalResult`` are re-exported here for backwards
compatibility with code that imports them from this module.
"""

from __future__ import annotations

import time

import structlog

from ..config import get_settings
from ..ingestion.models import Chunk
from ..ingestion.storage import BM25Index, ChromaStore
from ._types import RetrievalResult, ScoredChunk
from .reranker import Reranker
from .rrf import rrf_fuse

# Re-export so existing ``from .retriever import ScoredChunk`` imports keep working.
__all__ = ["Retriever", "RetrievalResult", "ScoredChunk"]

logger = structlog.get_logger(__name__)


def _matches_where(chunk: Chunk, where: dict) -> bool:
    """Return True when *chunk* satisfies all conditions in a *where* dict.

    Supports ChromaDB-style equality conditions:
        ``{"field": "value"}``  or  ``{"field": {"$eq": "value"}}``

    Unknown operators default to True (permissive) to avoid silently dropping
    chunks when an advanced operator isn't implemented yet.
    """
    for field_name, condition in where.items():
        val = getattr(chunk, field_name, None)
        if isinstance(condition, dict):
            for op, expected in condition.items():
                if op == "$eq" and val != expected:
                    return False
                elif op == "$ne" and val == expected:
                    return False
        else:
            if val != condition:
                return False
    return True


class Retriever:
    """Hybrid retriever backed by ChromaDB (vector) and BM25 (keyword).

    The public interface is ``retrieve(query) → RetrievalResult`` so
    generation and CLI code stays decoupled from retrieval internals.

    Args:
        chroma_store: Injected ChromaStore; loads from config paths if None.
        bm25_index: Injected BM25Index; loads from config path if None.
        reranker: Injected Reranker; constructed from config if None.
    """

    def __init__(
        self,
        chroma_store: ChromaStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = chroma_store or ChromaStore()
        self._bm25 = bm25_index or BM25Index.load()
        self._reranker = reranker or Reranker()
        self._settings = get_settings()

    def retrieve_candidates(
        self,
        query: str,
        where: dict | None = None,
    ) -> tuple[list[Chunk], dict]:
        """BM25 + vector → RRF → top rerank_candidates. No cross-encoder.

        Used by the LangGraph pipeline so that reranking is a separate node.

        Returns:
            Tuple of (candidates, log_info) where candidates is the list of
            Chunk objects in RRF-fused order, and log_info contains counts for
            structured logging (bm25_count, vector_count, overlap_count).
        """
        settings = self._settings

        bm25_raw: list[tuple[Chunk, float]] = self._bm25.search(
            query, top_k=settings.retrieval.bm25_top_k
        )
        vector_raw: list[tuple[Chunk, float]] = self._store.query(
            query_text=query,
            top_k=settings.retrieval.vector_top_k,
            where=where,
        )

        bm25_list: list[tuple[str, Chunk]] = [(c.chunk_id, c) for c, _ in bm25_raw]
        vector_list: list[tuple[str, Chunk]] = [(c.chunk_id, c) for c, _ in vector_raw]

        if where:
            bm25_list = [(cid, c) for cid, c in bm25_list if _matches_where(c, where)]

        bm25_ids = {cid for cid, _ in bm25_list}
        vector_ids = {cid for cid, _ in vector_list}

        fused = rrf_fuse([bm25_list, vector_list], k=settings.retrieval.rrf_k)
        n_candidates = settings.retrieval.rerank_candidates
        candidates: list[Chunk] = [chunk for _, chunk, _ in fused[:n_candidates]]

        log_info = {
            "bm25_count": len(bm25_list),
            "vector_count": len(vector_list),
            "overlap_count": len(bm25_ids & vector_ids),
            "rrf_candidates": len(fused),
        }
        return candidates, log_info

    def retrieve(
        self,
        query: str,
        where: dict | None = None,
    ) -> RetrievalResult:
        """Run the full hybrid retrieval pipeline for *query*.

        Steps:
        1. BM25 keyword search (top bm25_top_k).
        2. Vector search in ChromaDB (top vector_top_k).
        3. Post-filter BM25 results by *where* metadata conditions.
        4. Build ranked lists and fuse with RRF.
        5. Take top rerank_candidates from RRF output.
        6. Cross-encoder reranking; filter by confidence_threshold.
        7. Return sorted by descending reranker score.

        Args:
            query: The user's question.
            where: Optional ChromaDB-style metadata filter applied to both
                vector search and BM25 results.

        Returns:
            ``RetrievalResult`` whose ``has_relevant_chunks`` is True only
            when at least one chunk survived the confidence threshold.
        """
        start = time.monotonic()

        candidates, log_info = self.retrieve_candidates(query, where=where)

        final: list[ScoredChunk] = self._reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=self._settings.retrieval.rerank_top_k,
        )

        latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "retrieval.hybrid.complete",
            query_preview=query[:80],
            **log_info,
            rerank_candidates=len(candidates),
            above_threshold=len(final),
            rerank_scores=[round(sc.score, 3) for sc in final],
            latency_ms=round(latency_ms, 1),
        )

        return RetrievalResult(query=query, scored_chunks=final)
