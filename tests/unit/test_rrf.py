"""Unit tests for RRF fusion logic and deduplication.

Covers:
1. RRF score formula: 1/(k + rank), summed across lists.
2. Deduplication: chunk appearing in multiple lists appears once in output.
3. Cross-list boost: overlapping chunk outranks a non-overlapping chunk.
4. Output ordering: descending by RRF score.
5. Empty ranked list is ignored gracefully.
6. Single ranked list behaves correctly.
7. All empty lists → empty output.
8. k parameter changes scores but not relative order for non-overlapping lists.
"""

from __future__ import annotations

import pytest

from citesage.ingestion.models import Chunk
from citesage.retrieval.rrf import rrf_fuse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, content: str = "text") -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        source_file="test.md",
        chunk_index=0,
        doc_type="markdown",
        ingestion_timestamp="2024-01-01T00:00:00",
        content_hash="abc",
        token_count=1,
    )


# ---------------------------------------------------------------------------
# RRF score formula
# ---------------------------------------------------------------------------


class TestRRFFormula:
    def test_single_list_single_item(self):
        """rank=1, k=60 → score = 1/61."""
        c = _chunk("a")
        result = rrf_fuse([[("a", c)]], k=60)
        assert len(result) == 1
        chunk_id, chunk, score = result[0]
        assert chunk_id == "a"
        assert pytest.approx(score, rel=1e-9) == 1.0 / (60 + 1)

    def test_single_list_rank_order(self):
        """Rank 1 scores higher than rank 2."""
        a, b = _chunk("a"), _chunk("b")
        result = rrf_fuse([[("a", a), ("b", b)]], k=60)
        scores = {cid: s for cid, _, s in result}
        assert scores["a"] > scores["b"]
        assert pytest.approx(scores["a"], rel=1e-9) == 1.0 / 61
        assert pytest.approx(scores["b"], rel=1e-9) == 1.0 / 62

    def test_two_lists_non_overlapping(self):
        """Items from separate lists each get 1/(k+rank) from their list."""
        a, b = _chunk("a"), _chunk("b")
        result = rrf_fuse([[("a", a)], [("b", b)]], k=60)
        scores = {cid: s for cid, _, s in result}
        # Both appear at rank 1 in their respective list → same score.
        assert pytest.approx(scores["a"], rel=1e-9) == 1.0 / 61
        assert pytest.approx(scores["b"], rel=1e-9) == 1.0 / 61

    def test_custom_k_parameter(self):
        """k=0 → score = 1/rank; k=100 → score = 1/(100+rank)."""
        c = _chunk("c")
        result_k0 = rrf_fuse([[("c", c)]], k=0)
        result_k100 = rrf_fuse([[("c", c)]], k=100)
        _, _, s0 = result_k0[0]
        _, _, s100 = result_k100[0]
        assert pytest.approx(s0, rel=1e-9) == 1.0
        assert pytest.approx(s100, rel=1e-9) == 1.0 / 101


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_chunk_in_two_lists_appears_once(self):
        """A chunk_id present in both lists must appear exactly once in output."""
        c = _chunk("shared")
        result = rrf_fuse([[("shared", c)], [("shared", c)]], k=60)
        ids = [cid for cid, _, _ in result]
        assert ids.count("shared") == 1

    def test_overlap_adds_scores(self):
        """Chunk appearing at rank 1 in both lists → score = 2 * 1/(k+1)."""
        c = _chunk("shared")
        result = rrf_fuse([[("shared", c)], [("shared", c)]], k=60)
        _, _, score = result[0]
        expected = 2.0 / (60 + 1)
        assert pytest.approx(score, rel=1e-9) == expected

    def test_overlapping_chunk_outranks_non_overlapping(self):
        """Chunk in both lists beats a chunk that appears in only one list."""
        shared = _chunk("shared")
        unique = _chunk("unique")
        # "shared" is rank 2 in list A, rank 2 in list B.
        # "unique" is rank 1 in list A only.
        # shared: 1/62 + 1/62 = 2/62 ≈ 0.0323
        # unique: 1/61 ≈ 0.0164 — wait, rank 1 = 1/61 > 2/62 = 1/31. Actually
        # 1/61 ≈ 0.01639, 2/62 ≈ 0.03226. So shared wins.
        result = rrf_fuse(
            [[("unique", unique), ("shared", shared)], [("shared", shared)]],
            k=60,
        )
        scores = {cid: s for cid, _, s in result}
        assert scores["shared"] > scores["unique"]

    def test_chunk_object_taken_from_first_occurrence(self):
        """When a chunk_id appears in multiple lists, the first-seen Chunk object is kept."""
        c1 = _chunk("x")
        c1.content = "from list A"
        c2 = _chunk("x")
        c2.content = "from list B"
        result = rrf_fuse([[("x", c1)], [("x", c2)]], k=60)
        _, kept_chunk, _ = result[0]
        assert kept_chunk.content == "from list A"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_empty_lists_returns_empty(self):
        assert rrf_fuse([], k=60) == []
        assert rrf_fuse([[]], k=60) == []
        assert rrf_fuse([[], []], k=60) == []

    def test_one_empty_one_nonempty(self):
        """Empty inner list is silently ignored."""
        c = _chunk("a")
        result = rrf_fuse([[], [("a", c)]], k=60)
        assert len(result) == 1
        assert result[0][0] == "a"

    def test_output_sorted_descending(self):
        """Output is always ordered by descending RRF score."""
        chunks = [_chunk(str(i)) for i in range(5)]
        # Give them clear rank ordering in one list.
        ranked = [(str(i), c) for i, c in enumerate(chunks)]
        result = rrf_fuse([ranked], k=60)
        scores = [s for _, _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_multiple_lists_many_items(self):
        """Larger fusion doesn't lose items or produce duplicates."""
        list_a = [(_id, _chunk(_id)) for _id in "abcde"]
        list_b = [(_id, _chunk(_id)) for _id in "cdefg"]
        result = rrf_fuse([list_a, list_b], k=60)
        ids = [cid for cid, _, _ in result]
        # All 7 unique IDs present, each exactly once.
        assert sorted(ids) == sorted("abcdefg")
        assert len(ids) == len(set(ids))
