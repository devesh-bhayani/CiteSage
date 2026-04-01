"""RAGState: the single shared state dict threaded through the LangGraph pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    pass


class RAGState(TypedDict):
    """Full state for one CiteSage query through the LangGraph pipeline.

    Fields
    ------
    question
        The current user question (may be rewritten by transform_query_node).
    retrieved_chunks
        Raw RRF-fused candidates from BM25 + vector search (pre-rerank).
    reranked_chunks
        All candidates scored by the cross-encoder, sorted descending.
        Not filtered by confidence_threshold — routing reads the top score
        first, then the generate nodes apply the threshold.
    reranker_top_score
        The highest cross-encoder score in reranked_chunks, or 0.0 if empty.
        The path router compares this against ``retrieval.confidence_threshold``.
    answer
        The generated (or canned decline) answer text.
    citations
        The ScoredChunk objects that were actually cited in the answer.
    confidence
        "high" on the FAST PATH or when citations are verified.
        "low" after max retries or when >50 % of citations are unsupported.
        "" until set.
    path_taken
        "fast" | "thorough" | "declined" — set by the first generate/decline node.
    retry_count
        Number of query rewrites performed.  Guard rail: max 1 rewrite.
    error
        Non-fatal error message, if any node caught an exception.
    token_usage
        Flat accumulated token counts across all LLM calls (input_tokens,
        output_tokens).  Kept for backward compatibility.
    model_usage
        Per-model token counts and call counts, keyed by model ID::

            {"claude-sonnet-4-20250514": {"input_tokens": 1200,
                                          "output_tokens": 450,
                                          "calls": 1}}

        Used by CostTracker to compute estimated USD cost.
    """

    question: str
    retrieved_chunks: list  # list[Chunk]
    reranked_chunks: list  # list[ScoredChunk]
    reranker_top_score: float
    answer: str
    citations: list  # list[ScoredChunk]
    confidence: str
    path_taken: str
    retry_count: int
    error: str | None
    token_usage: dict
    model_usage: dict
