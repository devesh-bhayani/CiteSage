"""Unit tests for CitationVerifier — deterministic overlap checker.

These tests exercise only the pure-Python helpers (_token_overlap,
_extract_cited_indices, _extract_claim_context) and the confidence-decision
logic with a mocked LLM judge, so no API key is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from citesage.generation.citation_verifier import CitationVerifier
from citesage.ingestion.models import Chunk
from citesage.retrieval._types import ScoredChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source(content: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id="test-id",
        content=content,
        source_file="test.pdf",
        doc_type="pdf",
        chunk_index=0,
        ingestion_timestamp="2024-01-01T00:00:00Z",
        content_hash="abc123",
        token_count=len(content.split()),
    )
    return ScoredChunk(chunk=chunk, score=0.9)


# ---------------------------------------------------------------------------
# _token_overlap — deterministic
# ---------------------------------------------------------------------------


class TestTokenOverlap:
    def test_identical_text_returns_one(self):
        text = "The mitochondria is the powerhouse of the cell."
        assert CitationVerifier._token_overlap(text, text) == pytest.approx(1.0)

    def test_completely_unrelated_returns_zero(self):
        claim = "photosynthesis converts sunlight into glucose"
        source = "the stock market crashed during the financial crisis"
        assert CitationVerifier._token_overlap(claim, source) == pytest.approx(0.0)

    def test_partial_overlap(self):
        claim = "neural networks learn through backpropagation"
        source = "backpropagation is used to train neural networks in deep learning"
        overlap = CitationVerifier._token_overlap(claim, source)
        # "neural", "networks", "learn", "through", "backpropagation" are 4+ chars
        # "neural", "networks", "backpropagation" appear in source → 3/5 = 0.6
        assert overlap > 0.5

    def test_short_words_ignored(self):
        # All words < 4 chars → no significant tokens → returns 0.0
        claim = "a big cat ate the rat"
        source = "the cat sat on the mat"
        assert CitationVerifier._token_overlap(claim, source) == pytest.approx(0.0)

    def test_empty_claim_returns_zero(self):
        assert CitationVerifier._token_overlap(
            "", "some source content here"
        ) == pytest.approx(0.0)

    def test_case_insensitive(self):
        claim = "Reinforcement Learning maximizes cumulative Reward"
        source = "reinforcement learning maximizes cumulative reward"
        assert CitationVerifier._token_overlap(claim, source) == pytest.approx(1.0)

    def test_above_threshold_for_good_citation(self):
        claim = "According to [Source 1], transformer models use attention mechanisms."
        source = (
            "Transformer models rely on self-attention mechanisms to process sequences."
        )
        overlap = CitationVerifier._token_overlap(claim, source)
        assert overlap >= CitationVerifier.WEAK_THRESHOLD

    def test_below_threshold_for_bad_citation(self):
        claim = "According to [Source 1], photosynthesis produces oxygen."
        source = (
            "The French Revolution began in 1789 with the storming of the Bastille."
        )
        overlap = CitationVerifier._token_overlap(claim, source)
        assert overlap < CitationVerifier.WEAK_THRESHOLD


# ---------------------------------------------------------------------------
# _extract_cited_indices
# ---------------------------------------------------------------------------


class TestExtractCitedIndices:
    def test_single_citation(self):
        answer = "Neural networks are powerful [Source 1]."
        assert CitationVerifier._extract_cited_indices(answer) == {1}

    def test_multiple_citations(self):
        answer = "See [Source 1] and [Source 3] for details."
        assert CitationVerifier._extract_cited_indices(answer) == {1, 3}

    def test_duplicate_citations_deduplicated(self):
        answer = "[Source 2] is important. As shown in [Source 2], this holds."
        assert CitationVerifier._extract_cited_indices(answer) == {2}

    def test_no_citations(self):
        assert CitationVerifier._extract_cited_indices("No citations here.") == set()

    def test_whitespace_variants(self):
        answer = "See [Source  1] and [Source\t2]."
        indices = CitationVerifier._extract_cited_indices(answer)
        assert 1 in indices or 2 in indices  # tolerate spacing variants


# ---------------------------------------------------------------------------
# _extract_claim_context
# ---------------------------------------------------------------------------


class TestExtractClaimContext:
    def test_extracts_sentence_with_citation(self):
        answer = "The sky is blue. Transformers use attention [Source 1]. Nothing else."
        ctx = CitationVerifier._extract_claim_context(answer, 1)
        assert "attention" in ctx
        assert "Source 1" in ctx

    def test_falls_back_to_start_when_no_match(self):
        answer = "A" * 400
        ctx = CitationVerifier._extract_claim_context(answer, 99)
        assert len(ctx) <= 300


# ---------------------------------------------------------------------------
# verify() — full integration with mocked LLM
# ---------------------------------------------------------------------------


@pytest.fixture()
def verifier_no_llm():
    """CitationVerifier with ChatAnthropic replaced by a MagicMock."""
    mock_llm = MagicMock()
    with patch(
        "citesage.generation.citation_verifier.ChatAnthropic",
        return_value=mock_llm,
    ):
        v = CitationVerifier()
    v._llm = mock_llm
    return v, mock_llm


class TestVerify:
    def test_no_citations_returns_high(self, verifier_no_llm):
        v, _ = verifier_no_llm
        result = v.verify("No citations in this answer.", [])
        assert result.confidence == "high"

    def test_good_overlap_no_llm_call(self, verifier_no_llm):
        v, mock_llm = verifier_no_llm
        source = _make_source(
            "Transformer models use self-attention mechanisms to encode sequences."
        )
        answer = "Transformer models use attention mechanisms [Source 1]."
        result = v.verify(answer, [source])
        mock_llm.invoke.assert_not_called()
        assert result.confidence == "high"
        assert result.supported_count == 1

    def test_weak_overlap_calls_llm(self, verifier_no_llm):
        v, mock_llm = verifier_no_llm
        mock_response = MagicMock()
        mock_response.content = "YES"
        mock_llm.invoke.return_value = mock_response

        source = _make_source("The French Revolution began in 1789.")
        answer = "Photosynthesis produces oxygen [Source 1]."
        result = v.verify(answer, [source])

        mock_llm.invoke.assert_called_once()
        assert result.supported_count == 1
        assert 1 in result.weak_indices

    def test_llm_no_verdict_marks_unsupported(self, verifier_no_llm):
        v, mock_llm = verifier_no_llm
        mock_response = MagicMock()
        mock_response.content = "NO"
        mock_llm.invoke.return_value = mock_response

        source = _make_source("The French Revolution began in 1789.")
        answer = "Photosynthesis produces oxygen [Source 1]."
        result = v.verify(answer, [source])

        assert result.unsupported_count == 1
        assert result.confidence == "low"

    def test_partial_verdict_sets_partial_confidence(self, verifier_no_llm):
        v, mock_llm = verifier_no_llm
        mock_response = MagicMock()
        mock_response.content = "PARTIAL"
        mock_llm.invoke.return_value = mock_response

        source = _make_source("The French Revolution began in 1789.")
        answer = "Photosynthesis produces oxygen [Source 1]."
        result = v.verify(answer, [source])

        assert result.partial_count == 1
        assert result.confidence == "partial"

    def test_fabricated_source_index_is_unsupported(self, verifier_no_llm):
        v, _ = verifier_no_llm
        source = _make_source("Only one source exists.")
        # Answer cites [Source 5] which doesn't exist.
        answer = "Something is true [Source 5]."
        result = v.verify(answer, [source])
        assert result.unsupported_count == 1
        assert 5 in result.unsupported_indices
        assert result.confidence == "low"

    def test_majority_unsupported_returns_low(self, verifier_no_llm):
        v, mock_llm = verifier_no_llm
        mock_response = MagicMock()
        mock_response.content = "NO"
        mock_llm.invoke.return_value = mock_response

        sources = [
            _make_source("Unrelated content alpha beta gamma delta."),
            _make_source("Unrelated content epsilon zeta theta iota."),
            _make_source("Unrelated content kappa lambda mu nu xi."),
        ]
        # All three will have weak overlap → all go to LLM → all NO
        answer = (
            "Photosynthesis [Source 1] converts sunlight [Source 2] into "
            "glucose and oxygen [Source 3]."
        )
        result = v.verify(answer, sources)
        assert result.confidence == "low"

    def test_custom_weak_threshold(self):
        """A threshold of 1.0 forces every citation through the LLM judge."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "YES"
        mock_llm.invoke.return_value = mock_response

        with patch(
            "citesage.generation.citation_verifier.ChatAnthropic",
            return_value=mock_llm,
        ):
            v = CitationVerifier(weak_threshold=1.0)
        v._llm = mock_llm

        source = _make_source("Transformer models use self-attention mechanisms.")
        answer = "Transformers use attention [Source 1]."
        result = v.verify(answer, [source])
        mock_llm.invoke.assert_called_once()
        assert result.supported_count == 1
