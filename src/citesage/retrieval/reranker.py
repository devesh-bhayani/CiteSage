"""Cross-encoder reranker for Phase 2 hybrid retrieval.

Uses ms-marco-MiniLM-L-6-v2 (configurable via models.reranker in config.yaml).
Chunks scoring below ``retrieval.confidence_threshold`` are filtered out.
"""

from __future__ import annotations

from functools import lru_cache

import structlog
from sentence_transformers import CrossEncoder

from ..config import get_settings
from ..ingestion.models import Chunk
from ._types import ScoredChunk

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=None)
def _load_cross_encoder(name: str) -> CrossEncoder:
    """Load (and cache) a CrossEncoder by name.

    The model is loaded once per process and reused across every ``Reranker``
    instance. The graph constructs a ``Reranker`` inside node functions (so
    several times per query); without this cache each construction reloaded
    ~90 MB of weights, leaking memory until the process was OOM-killed mid-eval
    (~query 20). Cache key is the config-provided model name, so swapping
    ``models.reranker`` still yields a fresh model.
    """
    return CrossEncoder(name)


class Reranker:
    """Reranks candidate chunks using a cross-encoder model.

    The cross-encoder score replaces the upstream vector/RRF scores.
    Only chunks scoring at or above ``confidence_threshold`` are returned.

    Args:
        model_name: Override the model from ``models.reranker`` in config.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        name = model_name or settings.models.reranker
        self._model = _load_cross_encoder(name)
        self._settings = settings

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int | None = None,
        skip_threshold: bool = False,
    ) -> list[ScoredChunk]:
        """Score and rank *candidates* for *query*.

        Args:
            query: The user's question.
            candidates: Chunks to rerank (typically the top-N from RRF).
            top_k: Maximum results to return; defaults to
                ``retrieval.rerank_top_k`` from config.

        Returns:
            ``ScoredChunk`` list sorted by descending cross-encoder score,
            filtered to those at or above ``retrieval.confidence_threshold``.
            Empty list when *candidates* is empty.
        """
        if not candidates:
            return []

        settings = self._settings
        k = top_k if top_k is not None else settings.retrieval.rerank_top_k
        threshold = settings.retrieval.confidence_threshold

        pairs = [(query, c.content) for c in candidates]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)

        # Pair with original Chunk objects, sort descending, apply threshold.
        ranked = sorted(
            zip(raw_scores, candidates),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        results = [
            ScoredChunk(chunk=chunk, score=float(score))
            for score, chunk in ranked[:k]
            if skip_threshold or float(score) >= threshold
        ]

        logger.info(
            "reranker.complete",
            candidates_in=len(candidates),
            above_threshold=len(results),
            top_score=results[0].score if results else 0.0,
        )

        return results
