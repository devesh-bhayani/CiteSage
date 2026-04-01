"""Shared data types for the retrieval package.

Isolated here so that rrf.py, reranker.py, and retriever.py can all import
ScoredChunk / RetrievalResult without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ingestion.models import Chunk


@dataclass
class ScoredChunk:
    """A chunk paired with its relevance score (higher = more relevant).

    Phase 1: score is cosine similarity (0–1).
    Phase 2: score is cross-encoder reranker logit (unbounded, higher = better).
    """

    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    """Everything the generation layer needs from retrieval."""

    query: str
    scored_chunks: list[ScoredChunk] = field(default_factory=list)

    @property
    def has_relevant_chunks(self) -> bool:
        """True when at least one chunk survived the confidence threshold."""
        return len(self.scored_chunks) > 0

    @property
    def top_score(self) -> float:
        """Highest relevance score, or 0.0 if empty."""
        if not self.scored_chunks:
            return 0.0
        return self.scored_chunks[0].score
