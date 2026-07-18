"""Unit tests for the evaluation metric functions and grader-response parsers.

These functions define the project's success criteria (GAPS.md #8): a metric
bug already shipped once (citation precision measured as a boolean subset and
averaged). All tests here are pure-Python — no LLM, no retrieval stack.

Covers:
- check_citations        (evaluation/run_eval.py) — per-query citation precision
- _parse_grade_response  (evaluation/run_eval.py) — eval-grader JSON parsing
- _parse_grade_indices   (graph/nodes.py)         — relevance-grader array parsing
"""

from __future__ import annotations

import pytest

from citesage.evaluation.run_eval import _parse_grade_response, check_citations
from citesage.graph.nodes import _parse_grade_indices
from citesage.graph.pipeline import PipelineResult
from citesage.ingestion.models import Chunk
from citesage.retrieval._types import ScoredChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cited(chunk_id: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        content="some content",
        source_file="test.md",
        doc_type="markdown",
        chunk_index=0,
        ingestion_timestamp="2026-01-01T00:00:00Z",
        content_hash="abc123",
        token_count=2,
    )
    return ScoredChunk(chunk=chunk, score=0.9)


def _result(cited_ids: list[str], declined: bool = False) -> PipelineResult:
    return PipelineResult(
        answer="" if declined else "an answer",
        citations=[_cited(cid) for cid in cited_ids],
        declined=declined,
        path_taken="declined" if declined else "fast",
    )


# ---------------------------------------------------------------------------
# check_citations — per-query precision |cited ∩ expected| / |cited|
# ---------------------------------------------------------------------------


class TestCheckCitations:
    def test_declined_is_correct_and_unmeasured(self):
        ok, precision, _ = check_citations(_result([], declined=True), ["e1"])
        assert ok is True
        assert precision is None

    def test_no_expected_chunks_is_unmeasured(self):
        ok, precision, _ = check_citations(_result(["c1"]), [])
        assert ok is True
        assert precision is None

    def test_answered_without_citations_is_wrong_but_unmeasured(self):
        ok, precision, detail = check_citations(_result([]), ["e1"])
        assert ok is False
        assert precision is None
        assert "no citations" in detail

    def test_all_cited_correct(self):
        ok, precision, _ = check_citations(_result(["e1", "e2"]), ["e1", "e2", "e3"])
        assert ok is True
        assert precision == pytest.approx(1.0)

    def test_partial_precision_not_boolean(self):
        # The historic bug: 9 correct + 1 wrong scored 0.0. Must score 0.9.
        cited = [f"e{i}" for i in range(9)] + ["wrong"]
        expected = [f"e{i}" for i in range(9)]
        ok, precision, _ = check_citations(_result(cited), expected)
        assert ok is False
        assert precision == pytest.approx(0.9)

    def test_all_cited_wrong(self):
        ok, precision, _ = check_citations(_result(["w1", "w2"]), ["e1"])
        assert ok is False
        assert precision == pytest.approx(0.0)

    def test_duplicate_citations_counted_once(self):
        # citations are set-deduped: citing the same chunk twice is one cite
        ok, precision, _ = check_citations(_result(["e1", "e1"]), ["e1"])
        assert ok is True
        assert precision == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _parse_grade_response — eval grader verdict parsing
# ---------------------------------------------------------------------------


class TestParseGradeResponse:
    def test_clean_json(self):
        verdict, reason = _parse_grade_response(
            '{"verdict": "correct", "reason": "matches"}'
        )
        assert verdict == "correct"
        assert reason == "matches"

    def test_markdown_fenced_json(self):
        verdict, _ = _parse_grade_response(
            '```json\n{"verdict": "partial", "reason": "incomplete"}\n```'
        )
        assert verdict == "partial"

    def test_uppercase_verdict_normalized(self):
        verdict, _ = _parse_grade_response('{"verdict": "CORRECT", "reason": "x"}')
        assert verdict == "correct"

    def test_invalid_verdict_falls_back_to_incorrect(self):
        verdict, _ = _parse_grade_response('{"verdict": "maybe", "reason": "x"}')
        assert verdict == "incorrect"

    def test_prose_fallback_correct(self):
        verdict, _ = _parse_grade_response("The answer is correct.")
        assert verdict == "correct"

    def test_prose_fallback_partial(self):
        verdict, _ = _parse_grade_response("This is only a partial match")
        assert verdict == "partial"

    def test_garbage_falls_back_to_incorrect(self):
        verdict, _ = _parse_grade_response("no idea what this is")
        assert verdict == "incorrect"

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_input(self, text):
        verdict, _ = _parse_grade_response(text)
        assert verdict == "incorrect"


# ---------------------------------------------------------------------------
# _parse_grade_indices — relevance grader array parsing
# ---------------------------------------------------------------------------


class TestParseGradeIndices:
    def test_clean_array(self):
        assert _parse_grade_indices("[1, 3]", total=4) == [1, 3]

    def test_empty_array(self):
        assert _parse_grade_indices("[]", total=4) == []

    def test_prose_wrapped_array(self):
        assert _parse_grade_indices(
            "Relevant chunks: [2, 4] as requested", total=4
        ) == [
            2,
            4,
        ]

    def test_markdown_fenced_array(self):
        assert _parse_grade_indices("```json\n[1]\n```", total=3) == [1]

    def test_out_of_range_indices_dropped(self):
        assert _parse_grade_indices("[1, 7]", total=3) == [1]

    def test_zero_and_negative_dropped(self):
        assert _parse_grade_indices("[0, -1, 2]", total=3) == [2]

    def test_string_elements_dropped(self):
        # Documents current behavior: '["1"]' parses to [] → caller declines.
        # The real grade.yaml prompt shows an int example so production output
        # is ints; if this ever changes, loosen the parser, don't delete this.
        assert _parse_grade_indices('["1", "2"]', total=3) == []

    def test_garbage_returns_empty(self):
        assert _parse_grade_indices("I think chunk one is best", total=3) == []

    def test_empty_input_returns_empty(self):
        assert _parse_grade_indices("", total=3) == []

    def test_float_indices_coerced(self):
        assert _parse_grade_indices("[1.0, 2.0]", total=3) == [1, 2]
