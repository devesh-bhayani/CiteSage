"""Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

Formula (per retrieval CLAUDE.md):
    score(doc) = sum over all lists of  1 / (k + rank_in_list)

where rank_in_list is 1-based (best = 1) and k is the RRF constant (default 60).
"""

from __future__ import annotations

from ..ingestion.models import Chunk


def rrf_fuse(
    ranked_lists: list[list[tuple[str, Chunk]]],
    k: int = 60,
) -> list[tuple[str, Chunk, float]]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Deduplication is implicit: a chunk appearing in multiple lists accumulates
    score from each list; the Chunk object is taken from its first occurrence.

    Args:
        ranked_lists: Each inner list is ``(chunk_id, Chunk)`` pairs already
            ordered from best (rank 1) to worst.  Empty inner lists are
            ignored.
        k: RRF constant — larger values reduce the impact of rank differences.

    Returns:
        Deduplicated ``(chunk_id, Chunk, rrf_score)`` triples sorted by
        descending rrf_score.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, chunk) in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            # Keep the chunk object from its first occurrence across all lists.
            if chunk_id not in chunks:
                chunks[chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(cid, chunks[cid], scores[cid]) for cid in sorted_ids]
