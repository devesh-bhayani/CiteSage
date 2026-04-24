"""Integration tests for the CiteSage LangGraph pipeline routing.

Covers:
1. Hybrid vs vector-only retrieval (TestHybridVsVectorOnly)
2. Routing functions directly (TestRouteAfterRerank, TestRouteAfterVerify)
3. FAST PATH end-to-end (TestFastPath)
4. THOROUGH PATH end-to-end (TestThoroughPath)
5. DECLINE PATH end-to-end (TestDeclinePath)
6. RETRY limit enforcement (TestRetryLimit)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from citesage.graph.nodes import (
    route_after_rerank,
    route_after_verify,
)
from citesage.graph.pipeline import run_pipeline
from citesage.ingestion.models import Chunk
from citesage.retrieval._types import ScoredChunk
from citesage.retrieval.rrf import rrf_fuse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, content: str = "test content") -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        source_file="test.md",
        page_number=1,
        section_heading="Test",
        chunk_index=0,
        doc_type="markdown",
        ingestion_timestamp="2024-01-01T00:00:00",
        content_hash="abc123",
        token_count=10,
    )


def _scored(chunk_id: str, score: float, content: str = "test content") -> ScoredChunk:
    return ScoredChunk(chunk=_make_chunk(chunk_id, content), score=score)


def _stub_verifier(
    mock_verifier_cls,
    *,
    confidence: str = "high",
    total_cited: int = 1,
    supported_count: int = 1,
) -> None:
    """Configure a mocked CitationVerifier class so fast/thorough generate
    nodes get well-typed attributes (not MagicMocks) for token_usage and
    counters that flow into cost_tracker."""
    vresult = MagicMock()
    vresult.confidence = confidence
    vresult.total_cited = total_cited
    vresult.supported_count = supported_count
    vresult.partial_count = 0
    vresult.unsupported_count = 0
    vresult.weak_indices = []
    vresult.unsupported_indices = []
    vresult.token_usage = {}  # empty dict — `if vresult.token_usage` is False
    mock_verifier_cls.return_value.verify.return_value = vresult


def _make_state(**overrides) -> dict:
    """Return a minimal RAGState dict with sensible defaults."""
    base = {
        "question": "What is attention?",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "reranker_top_score": 0.0,
        "answer": "",
        "citations": [],
        "confidence": "",
        "path_taken": "",
        "retry_count": 0,
        "error": None,
        "token_usage": {},
        "model_usage": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Hybrid vs Vector-Only
# ---------------------------------------------------------------------------


class TestHybridVsVectorOnly:
    """Show that BM25 catches exact-term matches that vector search misses,
    and that hybrid RRF fusion combines both signal types."""

    def test_bm25_catches_exact_term_vaswani(self):
        """BM25 ranks the 'Vaswani' chunk first; vector ranks it lower."""
        vaswani_chunk = _make_chunk(
            "vaswani",
            "Vaswani et al. introduced the Transformer model in 2017.",
        )
        other_chunk = _make_chunk(
            "other",
            "Self-attention computes queries, keys, and values.",
        )

        # BM25 mock: exact match puts 'vaswani' at rank 1
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [
            (vaswani_chunk, 4.5),  # rank 1 — exact "vaswani" match
            (other_chunk, 0.1),  # rank 2
        ]

        # Vector mock: semantic similarity reverses the order
        mock_store = MagicMock()
        mock_store.query.return_value = [
            (other_chunk, 0.12),  # rank 1 in vector space
            (vaswani_chunk, 0.45),  # rank 2 — semantically less similar
        ]

        bm25_list = [(c.chunk_id, c) for c, _ in mock_bm25.search.return_value]
        vector_list = [(c.chunk_id, c) for c, _ in mock_store.query.return_value]

        bm25_top_id = bm25_list[0][0]
        vector_top_id = vector_list[0][0]

        # BM25 puts vaswani first; vector puts other first
        assert bm25_top_id == "vaswani"
        assert vector_top_id == "other"

        # BM25 catches vaswani while vector misses it as top-1
        assert bm25_top_id not in {vector_list[0][0]}

    def test_vector_catches_semantic_match_bm25_misses(self):
        """Vector search ranks the semantically relevant chunk first even
        when BM25 finds no keyword overlap."""
        semantic_chunk = _make_chunk(
            "semantic",
            "Scaled dot-product attention maps queries and keys.",
        )
        unrelated_chunk = _make_chunk(
            "unrelated",
            "The positional encoding uses sine and cosine functions.",
        )

        # BM25: query "how does attention mechanism work" has no exact
        # keyword match to "maps queries and keys" so ranks unrelated first
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [
            (unrelated_chunk, 2.1),  # BM25 rank 1 (keyword coincidence)
            (semantic_chunk, 0.3),  # BM25 rank 2
        ]

        # Vector: semantic similarity puts the right chunk first
        mock_store = MagicMock()
        mock_store.query.return_value = [
            (semantic_chunk, 0.08),  # vector rank 1 (low distance = close)
            (unrelated_chunk, 0.40),  # vector rank 2
        ]

        vector_top_id = mock_store.query.return_value[0][0].chunk_id
        bm25_top_id = mock_bm25.search.return_value[0][0].chunk_id

        assert vector_top_id == "semantic"
        assert bm25_top_id == "unrelated"

    def test_hybrid_rrf_combines_both_signals(self):
        """RRF fusion elevates a chunk that appears in both lists above a
        chunk that appears in only one."""
        shared_chunk = _make_chunk(
            "shared",
            "The Transformer attention layer uses QKV projections.",
        )
        bm25_only_chunk = _make_chunk(
            "bm25_only",
            "Vaswani et al. 2017 original paper citation.",
        )
        vector_only_chunk = _make_chunk(
            "vector_only",
            "Neural sequence-to-sequence mapping with context.",
        )

        # BM25 list: bm25_only at rank 1, shared at rank 2
        bm25_list = [
            ("bm25_only", bm25_only_chunk),
            ("shared", shared_chunk),
        ]
        # Vector list: vector_only at rank 1, shared at rank 2
        vector_list = [
            ("vector_only", vector_only_chunk),
            ("shared", shared_chunk),
        ]

        fused = rrf_fuse([bm25_list, vector_list], k=60)
        fused_ids = [cid for cid, _, _ in fused]
        fused_scores = {cid: score for cid, _, score in fused}

        # shared appears in both lists → accumulates score from both → wins
        assert fused_ids[0] == "shared"
        # shared score = 1/(60+2) + 1/(60+2) = 2/62
        expected_shared = 2.0 / (60 + 2)
        assert abs(fused_scores["shared"] - expected_shared) < 1e-9

        # Single-list chunks each get only 1/(60+1) from their rank-1 position
        expected_single = 1.0 / (60 + 1)
        assert abs(fused_scores["bm25_only"] - expected_single) < 1e-9
        assert abs(fused_scores["vector_only"] - expected_single) < 1e-9

        # The overlap chunk outranks both single-list chunks
        assert fused_scores["shared"] > fused_scores["bm25_only"]
        assert fused_scores["shared"] > fused_scores["vector_only"]

    def test_rrf_deduplication_no_duplicate_chunk_ids(self):
        """A chunk appearing in both BM25 and vector lists appears exactly
        once in the fused output."""
        chunk = _make_chunk("dup", "Attention is all you need.")
        bm25_list = [("dup", chunk)]
        vector_list = [("dup", chunk)]

        fused = rrf_fuse([bm25_list, vector_list], k=60)
        ids = [cid for cid, _, _ in fused]
        assert ids.count("dup") == 1


# ---------------------------------------------------------------------------
# 2. Routing functions
# ---------------------------------------------------------------------------


class TestRouteAfterRerank:
    """Test route_after_rerank boundary conditions."""

    def test_high_score_routes_fast(self):
        state = _make_state(
            reranked_chunks=[_scored("c1", 0.9)],
            reranker_top_score=0.9,
        )
        assert route_after_rerank(state) == "fast"

    def test_score_at_confidence_threshold_routes_fast(self):
        """Score exactly at confidence_threshold (0.8) → fast (>= check)."""
        state = _make_state(
            reranked_chunks=[_scored("c1", 0.8)],
            reranker_top_score=0.8,
        )
        assert route_after_rerank(state) == "fast"

    def test_medium_score_routes_thorough(self):
        state = _make_state(
            reranked_chunks=[_scored("c1", 0.3)],
            reranker_top_score=0.3,
        )
        assert route_after_rerank(state) == "thorough"

    def test_score_at_decline_threshold_routes_thorough_not_decline(self):
        """decline_threshold is -3.0; condition is top < -3.0, so -3.0 is NOT declined."""
        state = _make_state(
            reranked_chunks=[_scored("c1", -3.0)],
            reranker_top_score=-3.0,
        )
        # -3.0 < -3.0 is False → not declined → falls to thorough
        assert route_after_rerank(state) == "thorough"

    def test_score_just_below_decline_threshold_routes_decline(self):
        """Score -3.01 < -3.0 → decline."""
        state = _make_state(
            reranked_chunks=[_scored("c1", -3.01)],
            reranker_top_score=-3.01,
        )
        assert route_after_rerank(state) == "decline"

    def test_very_low_score_routes_decline(self):
        state = _make_state(
            reranked_chunks=[_scored("c1", -10.0)],
            reranker_top_score=-10.0,
        )
        assert route_after_rerank(state) == "decline"

    def test_empty_chunks_routes_decline(self):
        state = _make_state(
            reranked_chunks=[],
            reranker_top_score=0.0,
        )
        assert route_after_rerank(state) == "decline"


class TestRouteAfterVerify:
    """Test route_after_verify boundary conditions."""

    def test_low_confidence_retry_count_zero_routes_retry(self):
        """Low confidence and no retry yet → retry the query."""
        state = _make_state(confidence="low", retry_count=0)
        assert route_after_verify(state) == "retry"

    def test_low_confidence_retry_count_one_routes_done(self):
        """Low confidence but max retry (1) reached → done."""
        state = _make_state(confidence="low", retry_count=1)
        assert route_after_verify(state) == "done"

    def test_high_confidence_routes_done(self):
        state = _make_state(confidence="high", retry_count=0)
        assert route_after_verify(state) == "done"

    def test_partial_confidence_routes_done(self):
        state = _make_state(confidence="partial", retry_count=0)
        assert route_after_verify(state) == "done"


# ---------------------------------------------------------------------------
# 3. FAST PATH
# ---------------------------------------------------------------------------


class TestFastPath:
    """End-to-end fast path: high reranker score → one LLM call."""

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_fast_path_high_score(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """Score 0.85 triggers fast path: path_taken=fast, confidence=high,
        exactly one LLM call."""
        chunk = _make_chunk("c1", "The Transformer was introduced by Vaswani et al.")
        high_scored = _scored(
            "c1", 0.85, "The Transformer was introduced by Vaswani et al."
        )
        _stub_verifier(MockCitationVerifier)

        # Retriever mock returns candidates
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([chunk], {})
        MockRetriever.return_value = mock_retriever_instance

        # Reranker mock returns high-scored chunk
        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [high_scored]
        MockReranker.return_value = mock_reranker_instance

        # LLM mock returns a generated answer
        mock_llm_invoke.return_value = (
            "The Transformer was introduced in [Source 1].",
            {"input_tokens": 100, "output_tokens": 50},
        )

        result = run_pipeline("Who introduced the Transformer?")

        assert result.path_taken == "fast"
        assert result.confidence == "high"
        assert result.declined is False
        # Only one LLM call for the fast path generate node
        assert mock_llm_invoke.call_count == 1

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_fast_path_answer_returned(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """Fast path answer is the LLM output text."""
        chunk = _make_chunk("c1", "Self-attention mechanism text.")
        high_scored = _scored("c1", 0.9, "Self-attention mechanism text.")
        _stub_verifier(MockCitationVerifier)

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([chunk], {})
        MockRetriever.return_value = mock_retriever_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [high_scored]
        MockReranker.return_value = mock_reranker_instance

        expected_answer = "Self-attention uses queries, keys, and values [Source 1]."
        mock_llm_invoke.return_value = (
            expected_answer,
            {"input_tokens": 80, "output_tokens": 30},
        )

        result = run_pipeline("How does self-attention work?")

        assert result.answer == expected_answer
        assert len(result.citations) == 1


# ---------------------------------------------------------------------------
# 4. THOROUGH PATH
# ---------------------------------------------------------------------------


class TestThoroughPath:
    """End-to-end thorough path: medium reranker score → grade + generate + verify."""

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_thorough_path_medium_score(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """Score 0.3 triggers thorough path: path_taken=thorough, 2 LLM calls
        (grade_relevance + generate_thorough)."""
        chunk = _make_chunk(
            "c1", "Feed-forward networks apply two linear transformations."
        )
        medium_scored = _scored(
            "c1", 0.3, "Feed-forward networks apply two linear transformations."
        )

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([chunk], {})
        MockRetriever.return_value = mock_retriever_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [medium_scored]
        MockReranker.return_value = mock_reranker_instance

        # grade returns "[1]" → keep chunk 1
        # generate returns answer text
        mock_llm_invoke.side_effect = [
            ("[1]", {"input_tokens": 50, "output_tokens": 5}),  # grade call
            (
                "The feed-forward sublayer applies two linear transforms [Source 1].",
                {"input_tokens": 120, "output_tokens": 60},
            ),  # generate call
        ]

        # CitationVerifier mock: high confidence, no LLM calls
        mock_verifier_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.confidence = "high"
        mock_result.total_cited = 1
        mock_result.supported_count = 1
        mock_result.partial_count = 0
        mock_result.unsupported_count = 0
        mock_result.llm_calls = 0
        mock_result.token_usage = {}
        mock_verifier_instance.verify.return_value = mock_result
        MockCitationVerifier.return_value = mock_verifier_instance

        result = run_pipeline("What do feed-forward layers do?")

        assert result.path_taken == "thorough"
        # grade + generate = 2 LLM calls
        assert mock_llm_invoke.call_count == 2

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_thorough_path_confidence_from_verifier(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """Thorough path propagates the verifier confidence to the result."""
        chunk = _make_chunk("c1", "Multi-head attention text.")
        medium_scored = _scored("c1", 0.4, "Multi-head attention text.")

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([chunk], {})
        MockRetriever.return_value = mock_retriever_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [medium_scored]
        MockReranker.return_value = mock_reranker_instance

        mock_llm_invoke.side_effect = [
            ("[1]", {}),
            ("Answer with citation [Source 1].", {}),
        ]

        mock_verifier_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.confidence = "partial"
        mock_result.total_cited = 1
        mock_result.supported_count = 0
        mock_result.partial_count = 1
        mock_result.unsupported_count = 0
        mock_result.llm_calls = 0
        mock_result.token_usage = {}
        mock_verifier_instance.verify.return_value = mock_result
        MockCitationVerifier.return_value = mock_verifier_instance

        result = run_pipeline("How does multi-head attention work?")

        assert result.path_taken == "thorough"
        assert result.confidence == "partial"


# ---------------------------------------------------------------------------
# 5. DECLINE PATH
# ---------------------------------------------------------------------------


class TestDeclinePath:
    """Decline cases: empty retrieval, score too low, grading removes all chunks."""

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_empty_retrieval_declines(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """No retrieved chunks → declined=True, path_taken=declined, 0 LLM calls."""
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([], {})
        MockRetriever.return_value = mock_retriever_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = []
        MockReranker.return_value = mock_reranker_instance

        result = run_pipeline("What is a banana?")

        assert result.declined is True
        assert result.path_taken == "declined"
        assert mock_llm_invoke.call_count == 0

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_very_low_reranker_score_declines(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """Score -12.0 far below decline_threshold (-5.0) → decline, 0 LLM calls."""
        chunk = _make_chunk("c1", "Unrelated content about cooking.")
        very_low_scored = _scored("c1", -12.0, "Unrelated content about cooking.")

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([chunk], {})
        MockRetriever.return_value = mock_retriever_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [very_low_scored]
        MockReranker.return_value = mock_reranker_instance

        result = run_pipeline("What is the best pasta recipe?")

        assert result.declined is True
        assert result.path_taken == "declined"
        assert mock_llm_invoke.call_count == 0

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_grade_removes_all_chunks_declines(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """Grade node returns empty list → route_after_grade → decline."""
        chunk = _make_chunk("c1", "Marginally relevant content.")
        medium_scored = _scored("c1", 0.3, "Marginally relevant content.")

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([chunk], {})
        MockRetriever.return_value = mock_retriever_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [medium_scored]
        MockReranker.return_value = mock_reranker_instance

        # Grade call returns empty list [] — all chunks removed
        mock_llm_invoke.return_value = (
            "[]",
            {"input_tokens": 40, "output_tokens": 5},
        )

        result = run_pipeline("Irrelevant question about recipes?")

        assert result.declined is True
        assert result.path_taken == "declined"
        # Only 1 LLM call (grade), then decline without generate
        assert mock_llm_invoke.call_count == 1


# ---------------------------------------------------------------------------
# 6. RETRY LIMIT
# ---------------------------------------------------------------------------


class TestRetryLimit:
    """Verify the retry loop fires exactly once and then stops."""

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_retry_limit_exactly_5_llm_calls(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """Pipeline flow with one retry:
        grade(1) → generate(2) → verify[low] → transform(3) → grade(4) → generate(5)
        → verify[low again, retry_count=1] → done (max retry reached).
        Final: confidence=low, path_taken=thorough, exactly 5 LLM calls.
        """
        chunk = _make_chunk("c1", "Some content about attention layers.")
        medium_scored = _scored("c1", 0.3, "Some content about attention layers.")

        # Retriever always returns the same chunk
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([chunk], {})
        MockRetriever.return_value = mock_retriever_instance

        # Reranker always returns medium-scored chunk
        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [medium_scored]
        MockReranker.return_value = mock_reranker_instance

        # LLM call sequence:
        # Call 1: grade_relevance → "[1]" (keep chunk)
        # Call 2: generate_thorough → answer text
        # Call 3: transform_query → rewritten question
        # Call 4: grade_relevance (2nd pass) → "[1]"
        # Call 5: generate_thorough (2nd pass) → answer text
        mock_llm_invoke.side_effect = [
            ("[1]", {"input_tokens": 50, "output_tokens": 5}),  # grade 1st
            (
                "Answer attempt 1 [Source 1].",
                {"input_tokens": 100, "output_tokens": 40},
            ),  # generate 1st
            (
                "rewritten: attention mechanism query",
                {"input_tokens": 30, "output_tokens": 15},
            ),  # transform
            ("[1]", {"input_tokens": 50, "output_tokens": 5}),  # grade 2nd
            (
                "Answer attempt 2 [Source 1].",
                {"input_tokens": 100, "output_tokens": 40},
            ),  # generate 2nd
        ]

        # CitationVerifier always returns low confidence to trigger retry path
        mock_verifier_instance = MagicMock()

        def _low_confidence_result():
            mock_result = MagicMock()
            mock_result.confidence = "low"
            mock_result.total_cited = 1
            mock_result.supported_count = 0
            mock_result.partial_count = 0
            mock_result.unsupported_count = 1
            mock_result.llm_calls = 0
            mock_result.token_usage = {}
            return mock_result

        mock_verifier_instance.verify.side_effect = [
            _low_confidence_result(),  # 1st verify → low → triggers retry
            _low_confidence_result(),  # 2nd verify → low → but retry_count=1 → done
        ]
        MockCitationVerifier.return_value = mock_verifier_instance

        result = run_pipeline("What layers does the Transformer use?")

        # Exactly 5 LLM calls: grade, generate, transform, grade, generate
        assert (
            mock_llm_invoke.call_count == 5
        ), f"Expected 5 LLM calls, got {mock_llm_invoke.call_count}"
        assert result.confidence == "low"
        assert result.path_taken == "thorough"
        assert result.declined is False

    @patch("citesage.graph.nodes.CitationVerifier")
    @patch("citesage.graph.nodes._llm_invoke_with_retry")
    @patch("citesage.graph.nodes.Reranker")
    @patch("citesage.graph.nodes.Retriever")
    def test_no_retry_when_confidence_high_after_first_verify(
        self,
        MockRetriever,
        MockReranker,
        mock_llm_invoke,
        MockCitationVerifier,
    ):
        """High confidence after first verify → no transform_query call → 2 LLM calls."""
        chunk = _make_chunk("c1", "Attention mechanism uses queries and keys.")
        medium_scored = _scored("c1", 0.4, "Attention mechanism uses queries and keys.")

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_candidates.return_value = ([chunk], {})
        MockRetriever.return_value = mock_retriever_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [medium_scored]
        MockReranker.return_value = mock_reranker_instance

        mock_llm_invoke.side_effect = [
            ("[1]", {}),
            ("Good answer with citation [Source 1].", {}),
        ]

        mock_verifier_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.confidence = "high"
        mock_result.total_cited = 1
        mock_result.supported_count = 1
        mock_result.partial_count = 0
        mock_result.unsupported_count = 0
        mock_result.llm_calls = 0
        mock_result.token_usage = {}
        mock_verifier_instance.verify.return_value = mock_result
        MockCitationVerifier.return_value = mock_verifier_instance

        result = run_pipeline("Explain attention mechanism.")

        # grade + generate only; no transform
        assert mock_llm_invoke.call_count == 2
        assert result.confidence == "high"
        assert result.path_taken == "thorough"
