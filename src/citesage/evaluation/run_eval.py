"""CiteSage Evaluation Harness — Phase 3.

Runs the full CiteSage pipeline against the golden dataset and reports:
  - Per-query answer correctness (LLM judge via Haiku)
  - Citation precision (cited chunks ⊆ expected_source_chunks)
  - Decline recall / precision for unanswerable questions
  - Per-category accuracy breakdown
  - Aggregate metrics vs Phase 3 targets
  - Latency percentiles (p50 / p95)
  - Token usage and USD cost
  - Optional RAGAS trend metrics (--ragas flag)

Usage
-----
    python -m citesage.evaluation.run_eval \\
        --dataset tests/eval/golden_dataset.json \\
        --output  tests/eval/results.json \\
        --errors  tests/eval/errors.jsonl

Phase 3 exit targets
--------------------
    accuracy         >= 85 %
    citation_precision >= 90 %
    decline_recall   >= 85 %
    p95_latency      <  5000 ms
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, quantiles
from typing import Optional

import structlog
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..utils.llm_factory import get_grader_llm
from ..graph.pipeline import PipelineResult, run_pipeline
from ..prompts import load_prompt
from ..utils.cost_tracker import CostTracker

load_dotenv()
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUDGET_CAP_USD = 2.0  # hard stop if total spend hits this

# Phase 3 targets (used for pass/fail summary)
TARGET_ACCURACY = 0.85
TARGET_CITATION_PRECISION = 0.90
TARGET_DECLINE_RECALL = 0.85
TARGET_P95_LATENCY_MS = 5000.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GoldenItem:
    """One entry from the golden dataset."""

    id: str
    question: str
    expected_answer: Optional[str]
    expected_source_chunks: list[str]
    category: str
    difficulty: str
    decline_reason: Optional[str] = None


@dataclass
class QueryResult:
    """Graded output for one golden item."""

    item: GoldenItem
    pipeline_result: PipelineResult
    latency_ms: float
    # "correct" | "partial" | "incorrect"
    # "declined_correct" (unanswerable → declined)
    # "declined_incorrect" (answerable → declined)
    grade: str
    grade_reason: str
    citation_correct: bool
    # Per-query precision: |cited ∩ expected| / |cited|. None when not measurable
    # (declined, no expected chunks, or no citations emitted).
    citation_precision: Optional[float]
    citation_details: str
    grader_cost_usd: float = 0.0
    pipeline_cost_usd: float = 0.0
    error: Optional[str] = None

    @property
    def is_correct(self) -> bool:
        """True when the answer is fully or partially correct."""
        return self.grade in ("correct", "partial", "declined_correct")

    @property
    def is_fully_correct(self) -> bool:
        return self.grade in ("correct", "declined_correct")


@dataclass
class CategoryStats:
    """Aggregated stats for one category."""

    count: int = 0
    correct: int = 0
    partial: int = 0
    incorrect: int = 0
    citation_correct: int = 0  # queries where cited ⊆ expected
    citation_checked: int = 0  # queries for which citation precision is measured
    citation_precision_sum: float = 0.0  # sum of per-query precision values
    failures: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.count == 0:
            return 0.0
        return (self.correct + self.partial * 0.5) / self.count


# ---------------------------------------------------------------------------
# Answer grader (Haiku LLM judge)
# ---------------------------------------------------------------------------


class AnswerGrader:
    """Uses Haiku to semantically grade answer correctness.

    Cumulates spend so the caller can enforce budget caps.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.models.grader
        self._llm = get_grader_llm(max_tokens=256)
        self._tracker = CostTracker(settings.pricing.as_dict())
        self.total_cost_usd: float = 0.0

    def grade(
        self, question: str, expected: str, actual: str
    ) -> tuple[str, str, float]:
        """Return (verdict, reason, call_cost_usd).

        verdict: "correct" | "partial" | "incorrect"
        """
        prompt = load_prompt("eval_grade")
        messages = [
            SystemMessage(content=prompt["system"]),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Expected answer: {expected}\n\n"
                    f"Actual answer: {actual}"
                )
            ),
        ]

        last_exc: Exception | None = None
        response = None
        usage: dict = {}
        for attempt in range(1, 4):
            try:
                response = self._llm.invoke(messages)
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = dict(response.usage_metadata)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        else:
            raise last_exc  # type: ignore[misc]

        # Accumulate cost
        model_usage = {self._model_name: {**usage, "calls": 1}}
        call_cost = self._tracker.compute(model_usage).total_cost_usd
        self.total_cost_usd += call_cost

        # Parse JSON response
        text = (response.content or "").strip()  # type: ignore[union-attr]
        verdict, reason = _parse_grade_response(text)

        return verdict, reason, call_cost


def _parse_grade_response(text: str) -> tuple[str, str]:
    """Extract verdict/reason from the LLM judge response."""
    # Strip markdown fences if the model wrapped its JSON
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        raw = json.loads(stripped)
        verdict = str(raw.get("verdict", "incorrect")).lower()
        reason = str(raw.get("reason", ""))
        if verdict not in ("correct", "partial", "incorrect"):
            verdict = "incorrect"
        return verdict, reason
    except (json.JSONDecodeError, AttributeError):
        lower = stripped.lower()
        if "correct" in lower:
            verdict = "correct"
        elif "partial" in lower:
            verdict = "partial"
        else:
            verdict = "incorrect"
        return verdict, stripped[:200]


# ---------------------------------------------------------------------------
# Citation checker
# ---------------------------------------------------------------------------


def check_citations(
    pipeline_result: PipelineResult,
    expected_chunks: list[str],
) -> tuple[bool, Optional[float], str]:
    """Return (is_all_correct, precision, details_str).

    - is_all_correct: every cited chunk is in expected_source_chunks (kept as a
      secondary, strict breakdown).
    - precision: |cited ∩ expected| / |cited| for this query. None when not
      measurable (declined, no reference chunks, or no citations emitted).
      This is the per-query value that feeds the aggregate citation_precision
      metric defined by the Phase 3 spec.
    """
    if pipeline_result.declined:
        return True, None, "declined — no citations"

    cited_ids = {sc.chunk.chunk_id for sc in pipeline_result.citations}

    if not expected_chunks:
        return True, None, "no reference chunks to check against"

    if not cited_ids:
        return False, None, "answered but emitted no citations"

    expected_set = set(expected_chunks)
    correct_cites = cited_ids & expected_set
    false_citations = cited_ids - expected_set
    precision = len(correct_cites) / len(cited_ids)

    if not false_citations:
        detail = f"all {len(cited_ids)} cited chunk(s) are correct"
        return True, precision, detail

    missing = expected_set - cited_ids
    detail = (
        f"precision={precision:.2f}; " f"false citations: {sorted(false_citations)}"
    )
    if missing:
        detail += f"; expected but missing: {sorted(missing)}"
    return False, precision, detail


# ---------------------------------------------------------------------------
# RAGAS trend metrics (optional)
# ---------------------------------------------------------------------------


def run_ragas_metrics(
    query_results: list[QueryResult],
) -> dict:
    """Compute RAGAS faithfulness + answer_relevancy as trend metrics.

    Skips unanswerable (declined) items since they have no generated answer.
    Returns a dict with metric names and scores, or {"error": ...} on failure.
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics.collections import answer_relevancy, faithfulness
    except ImportError:
        return {"error": "ragas not installed"}

    samples = []
    for qr in query_results:
        if qr.item.category == "unanswerable" or qr.pipeline_result.declined:
            continue
        if not qr.pipeline_result.answer:
            continue
        contexts = [sc.chunk.content for sc in qr.pipeline_result.citations]
        samples.append(
            SingleTurnSample(
                user_input=qr.item.question,
                response=qr.pipeline_result.answer,
                retrieved_contexts=contexts or [""],
                reference=qr.item.expected_answer or "",
            )
        )

    if not samples:
        return {"error": "no answerable samples to evaluate"}

    try:
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            raise_exceptions=False,
            show_progress=False,
        )
        scores: dict = {}
        for key, val in result.items():
            try:
                scores[key] = round(float(val), 4)
            except (TypeError, ValueError):
                scores[key] = val
        return scores
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Aggregate metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    results: list[QueryResult],
) -> dict:
    """Compute all aggregate + per-category metrics from graded results."""
    cats: dict[str, CategoryStats] = {}
    latencies: list[float] = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    # Path counters
    fast_count = 0

    # Decline counters
    total_unanswerable = 0
    correctly_declined = 0  # unanswerable → declined
    total_declined = 0  # all queries that were declined
    incorrectly_declined = 0  # answerable → declined

    for qr in results:
        cat = qr.item.category
        stats = cats.setdefault(cat, CategoryStats())
        stats.count += 1
        latencies.append(qr.latency_ms)

        # Path tracking
        if qr.pipeline_result.path_taken == "fast":
            fast_count += 1

        # Token + cost
        pu = qr.pipeline_result.token_usage
        total_input_tokens += pu.get("input_tokens", 0)
        total_output_tokens += pu.get("output_tokens", 0)
        total_cost_usd += qr.pipeline_cost_usd + qr.grader_cost_usd

        # Decline tracking
        if qr.pipeline_result.declined:
            total_declined += 1
            if cat == "unanswerable":
                correctly_declined += 1

        if cat == "unanswerable":
            total_unanswerable += 1

        # Grade counts
        if qr.grade in ("correct", "declined_correct"):
            stats.correct += 1
        elif qr.grade == "partial":
            stats.partial += 1
        else:  # incorrect / declined_incorrect
            stats.incorrect += 1
            if qr.grade == "declined_incorrect":
                incorrectly_declined += 1

        # Citation: only measure precision on answerable items where the
        # pipeline emitted citations (precision is None otherwise — e.g.,
        # declined or no reference chunks to check against).
        if (
            qr.item.category != "unanswerable"
            and qr.item.expected_source_chunks
            and qr.citation_precision is not None
        ):
            stats.citation_checked += 1
            stats.citation_precision_sum += qr.citation_precision
            if qr.citation_correct:
                stats.citation_correct += 1

    # Aggregate accuracy (over ALL queries, counting unanswerable-correct)
    total = len(results)
    total_correct = sum(s.correct for s in cats.values())
    total_partial = sum(s.partial for s in cats.values())
    accuracy = (total_correct + total_partial * 0.5) / total if total else 0.0

    # Citation precision: mean of per-query |cited ∩ expected| / |cited|
    # across answerable queries with reference chunks (Phase 3 spec definition).
    total_cit_checked = sum(s.citation_checked for s in cats.values())
    total_cit_correct = sum(s.citation_correct for s in cats.values())
    total_cit_precision_sum = sum(s.citation_precision_sum for s in cats.values())
    citation_precision = (
        total_cit_precision_sum / total_cit_checked if total_cit_checked else 1.0
    )
    # Secondary: fraction of queries where every cite was correct (strict).
    citation_all_correct_rate = (
        total_cit_correct / total_cit_checked if total_cit_checked else 1.0
    )

    # Decline recall = correctly_declined / total_unanswerable
    decline_recall = (
        correctly_declined / total_unanswerable if total_unanswerable else 1.0
    )

    # Decline precision = correctly_declined / total_declined
    # (what fraction of declines were actually unanswerable)
    decline_precision = correctly_declined / total_declined if total_declined else 1.0

    # Latency percentiles
    sorted_lat = sorted(latencies)
    p50 = median(sorted_lat) if sorted_lat else 0.0
    p95 = (
        quantiles(sorted_lat, n=20)[18]
        if len(sorted_lat) >= 2
        else (sorted_lat[0] if sorted_lat else 0.0)
    )

    # Per-category breakdown
    by_category: dict = {}
    for cat_name, stats in sorted(cats.items()):
        failures = [
            qr
            for qr in results
            if qr.item.category == cat_name and not qr.is_fully_correct
        ]
        by_category[cat_name] = {
            "count": stats.count,
            "correct": stats.correct,
            "partial": stats.partial,
            "incorrect": stats.incorrect,
            "accuracy": round(stats.accuracy, 4),
            "citation_checked": stats.citation_checked,
            "citation_all_correct": stats.citation_correct,
            "citation_precision": (
                round(stats.citation_precision_sum / stats.citation_checked, 4)
                if stats.citation_checked
                else None
            ),
            "failures": [
                {
                    "id": qr.item.id,
                    "grade": qr.grade,
                    "reason": qr.grade_reason,
                    "citations_ok": qr.citation_correct,
                    "citation_precision": (
                        round(qr.citation_precision, 4)
                        if qr.citation_precision is not None
                        else None
                    ),
                }
                for qr in failures
            ],
        }

    return {
        "total_queries": total,
        "items_completed": total,
        "aggregate": {
            "accuracy": round(accuracy, 4),
            "citation_precision": round(citation_precision, 4),
            "citation_all_correct_rate": round(citation_all_correct_rate, 4),
            "decline_recall": round(decline_recall, 4),
            "decline_precision": round(decline_precision, 4),
        },
        "targets": {
            "accuracy": TARGET_ACCURACY,
            "citation_precision": TARGET_CITATION_PRECISION,
            "decline_recall": TARGET_DECLINE_RECALL,
            "p95_latency_ms": TARGET_P95_LATENCY_MS,
        },
        "passes_targets": {
            "accuracy": accuracy >= TARGET_ACCURACY,
            "citation_precision": citation_precision >= TARGET_CITATION_PRECISION,
            "decline_recall": decline_recall >= TARGET_DECLINE_RECALL,
            "p95_latency_ms": p95 <= TARGET_P95_LATENCY_MS,
        },
        "latency": {
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
        },
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "average_per_query": (
                round((total_input_tokens + total_output_tokens) / total, 1)
                if total
                else 0
            ),
        },
        "cost": {
            "total_usd": round(total_cost_usd, 6),
            "avg_cost_per_query": round(total_cost_usd / total, 6) if total else 0.0,
        },
        "fast_path_ratio": round(fast_count / total, 4) if total else 0.0,
        "by_category": by_category,
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Orchestrates the full evaluation: pipeline → grade → aggregate."""

    def __init__(self) -> None:
        self._grader = AnswerGrader()
        settings = get_settings()
        self._cost_tracker = CostTracker(settings.pricing.as_dict())
        self._pipeline_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self._grader.total_cost_usd + self._pipeline_cost_usd

    def run(
        self,
        dataset: list[GoldenItem],
        budget_cap: float = BUDGET_CAP_USD,
        verbose: bool = True,
    ) -> list[QueryResult]:
        """Run all items in *dataset* and return graded QueryResult list."""
        results: list[QueryResult] = []

        for i, item in enumerate(dataset, start=1):
            if self.total_cost_usd >= budget_cap:
                _warn_budget(budget_cap, self.total_cost_usd, i - 1, len(dataset))
                break

            if verbose:
                print(
                    f"  [{i:3d}/{len(dataset)}] {item.id}: " f"{item.question[:55]}...",
                    end="",
                    flush=True,
                )

            t0 = time.perf_counter()
            try:
                pipeline_result = run_pipeline(item.question)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.error("eval.pipeline_error", id=item.id, error=str(exc))
                qr = QueryResult(
                    item=item,
                    pipeline_result=PipelineResult(
                        answer="",
                        declined=False,
                        citations=[],
                        confidence="",
                        path_taken="error",
                    ),
                    latency_ms=elapsed_ms,
                    grade="incorrect",
                    grade_reason=f"pipeline error: {exc}",
                    citation_correct=False,
                    citation_precision=None,
                    citation_details="pipeline error",
                    error=str(exc),
                )
                results.append(qr)
                if verbose:
                    print(" ERROR")
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Accumulate pipeline cost
            pipeline_cost = 0.0
            if pipeline_result.query_cost:
                pipeline_cost = pipeline_result.query_cost.total_cost_usd
                self._pipeline_cost_usd += pipeline_cost

            # Grade
            grade, grade_reason, grader_cost = self._grade_item(item, pipeline_result)

            # Citation check
            citation_correct, citation_precision, citation_details = check_citations(
                pipeline_result, item.expected_source_chunks
            )

            qr = QueryResult(
                item=item,
                pipeline_result=pipeline_result,
                latency_ms=elapsed_ms,
                grade=grade,
                grade_reason=grade_reason,
                citation_correct=citation_correct,
                citation_precision=citation_precision,
                citation_details=citation_details,
                grader_cost_usd=grader_cost,
                pipeline_cost_usd=pipeline_cost,
            )
            results.append(qr)

            if verbose:
                icon = _grade_icon(grade)
                print(f" {icon} ({elapsed_ms:.0f}ms)")

        if verbose:
            print(
                f"\n  Total spend: ${self.total_cost_usd:.4f} USD "
                f"(pipeline: ${self._pipeline_cost_usd:.4f}, "
                f"grader: ${self._grader.total_cost_usd:.4f})"
            )

        return results

    def _grade_item(
        self,
        item: GoldenItem,
        result: PipelineResult,
    ) -> tuple[str, str, float]:
        """Return (grade, reason, grader_cost_usd) for one item."""
        if item.category == "unanswerable":
            if result.declined:
                return (
                    "declined_correct",
                    "correctly declined unanswerable question",
                    0.0,
                )
            return (
                "declined_incorrect",
                "failed to decline — should have declined this unanswerable question",
                0.0,
            )

        # Answerable question
        if result.declined:
            return (
                "incorrect",
                "incorrectly declined an answerable question",
                0.0,
            )

        if not item.expected_answer:
            # No expected answer to compare against (shouldn't happen for answerable)
            return "partial", "no expected answer provided for comparison", 0.0

        verdict, reason, cost = self._grader.grade(
            question=item.question,
            expected=item.expected_answer,
            actual=result.answer,
        )
        return verdict, reason, cost


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _grade_icon(grade: str) -> str:
    icons = {
        "correct": "✓",
        "declined_correct": "✓",
        "partial": "~",
        "incorrect": "✗",
        "declined_incorrect": "✗",
    }
    return icons.get(grade, "?")


def _warn_budget(cap: float, spent: float, done: int, total: int) -> None:
    msg = (
        f"\n[WARNING] Budget cap ${cap:.2f} reached after "
        f"{done}/{total} queries (${spent:.4f} spent). Stopping early."
    )
    print(msg, file=sys.stderr)
    logger.warning(
        "eval.budget_exceeded",
        cap_usd=cap,
        spent_usd=round(spent, 4),
        items_done=done,
        items_total=total,
    )


def write_results(
    metrics: dict,
    ragas_scores: Optional[dict],
    output_path: Path,
    timestamp: str,
) -> None:
    """Write evaluation_results.json."""
    payload = {
        "timestamp": timestamp,
        **metrics,
    }
    if ragas_scores is not None:
        payload["ragas"] = ragas_scores

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
    print(f"\n  Results written to: {output_path}")


def write_errors(
    results: list[QueryResult],
    errors_path: Path,
) -> None:
    """Write evaluation_errors.jsonl — one line per failure."""
    failures = [qr for qr in results if not qr.is_fully_correct or qr.error]

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with open(errors_path, "w", encoding="utf-8") as fh:
        for qr in failures:
            cited_ids = (
                [sc.chunk.chunk_id for sc in qr.pipeline_result.citations]
                if not qr.pipeline_result.declined
                else []
            )
            entry = {
                "id": qr.item.id,
                "category": qr.item.category,
                "difficulty": qr.item.difficulty,
                "error_type": qr.grade,
                "question": qr.item.question,
                "expected": qr.item.expected_answer,
                "got": (
                    qr.pipeline_result.answer[:500]
                    if qr.pipeline_result.answer
                    else None
                ),
                "grade_reason": qr.grade_reason,
                "citation_correct": qr.citation_correct,
                "citation_details": qr.citation_details,
                "cited_chunks": cited_ids,
                "expected_chunks": qr.item.expected_source_chunks,
                "path_taken": qr.pipeline_result.path_taken,
                "latency_ms": round(qr.latency_ms, 1),
                "pipeline_error": qr.error,
            }
            fh.write(json.dumps(entry, ensure_ascii=True) + "\n")

    if failures:
        print(f"  Error log written to: {errors_path} ({len(failures)} failures)")
    else:
        print(f"  No failures — error log empty: {errors_path}")


def print_summary(metrics: dict) -> None:
    """Print a human-readable pass/fail summary to stdout."""
    agg = metrics["aggregate"]
    tgt = metrics["targets"]
    pss = metrics["passes_targets"]
    lat = metrics["latency"]

    print("\n" + "=" * 60)
    print("  PHASE 3 EVALUATION SUMMARY")
    print("=" * 60)
    print(
        f"  Queries evaluated : {metrics['items_completed']} / {metrics['total_queries']}"
    )
    print()
    print("  Metric                 Score    Target  Pass?")
    print("  " + "-" * 54)
    _row("Accuracy", agg["accuracy"], tgt["accuracy"], pss["accuracy"])
    _row(
        "Citation Precision",
        agg["citation_precision"],
        tgt["citation_precision"],
        pss["citation_precision"],
    )
    _row(
        "Decline Recall",
        agg["decline_recall"],
        tgt["decline_recall"],
        pss["decline_recall"],
    )
    print(
        f"  {'P95 Latency':<22} {lat['p95_ms']:>6.0f}ms  "
        f"<{tgt['p95_latency_ms']:.0f}ms  "
        f"{'PASS' if pss['p95_latency_ms'] else 'FAIL'}"
    )
    print()
    print("  Per-category accuracy:")
    for cat, stats in metrics["by_category"].items():
        acc_pct = stats["accuracy"] * 100
        print(
            f"    {cat:<20} {acc_pct:5.1f}%  ({stats['correct']+stats['partial']}/{stats['count']})"
        )
    print()
    all_pass = all(pss.values())
    if all_pass:
        print("  *** ALL TARGETS MET — Phase 3 COMPLETE ***")
    else:
        failed = [k for k, v in pss.items() if not v]
        print(f"  *** {len(failed)} target(s) not met: {', '.join(failed)} ***")
    print("=" * 60 + "\n")


def _row(name: str, score: float, target: float, passed: bool) -> None:
    print(
        f"  {name:<22} {score:>5.1%}  >= {target:.0%}   {'PASS' if passed else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[GoldenItem]:
    """Load and validate the golden dataset from *path*."""
    with open(path, encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)

    items: list[GoldenItem] = []
    for entry in raw:
        items.append(
            GoldenItem(
                id=entry["id"],
                question=entry["question"],
                expected_answer=entry.get("expected_answer"),
                expected_source_chunks=entry.get("expected_source_chunks", []),
                category=entry["category"],
                difficulty=entry["difficulty"],
                decline_reason=entry.get("decline_reason"),
            )
        )
    return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run CiteSage evaluation against the golden dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/eval/golden_dataset.json"),
        help="Path to golden_dataset.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("reports/eval_report.json"),
        help="Path to write eval_report.json",
    )
    p.add_argument(
        "--errors",
        type=Path,
        default=Path("reports/eval_errors.jsonl"),
        help="Path to write eval_errors.jsonl",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Run only this category (e.g. factual_lookup)",
    )
    p.add_argument(
        "--subset",
        "--limit",
        dest="subset",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N items (for quick smoke tests)",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=BUDGET_CAP_USD,
        help="USD budget cap; stops early if exceeded",
    )
    p.add_argument(
        "--ragas",
        action="store_true",
        default=False,
        help="Also compute RAGAS faithfulness + answer_relevancy trend metrics",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-query progress output",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns 0 on success, 1 if targets are not met."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Windows stdout may default to cp1252; force UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\nCiteSage Evaluation Harness")
    print(f"  Dataset : {args.dataset}")
    print(f"  Output  : {args.output}")
    print(f"  Errors  : {args.errors}")
    print(f"  Budget  : ${args.budget:.2f} USD\n")

    # Load dataset
    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    dataset = load_dataset(args.dataset)

    # Filter by category
    if args.category:
        dataset = [i for i in dataset if i.category == args.category]
        print(f"  Filtered to category '{args.category}': {len(dataset)} items")

    # Apply subset limit
    if args.subset:
        dataset = dataset[: args.subset]
        print(f"  Limited to first {args.subset} items")

    if not dataset:
        print("ERROR: no items to evaluate after filtering.", file=sys.stderr)
        return 1

    print(f"  Running {len(dataset)} queries...\n")

    timestamp = datetime.now(timezone.utc).isoformat()

    # Run evaluation
    runner = EvalRunner()
    results = runner.run(dataset, budget_cap=args.budget, verbose=not args.quiet)

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["total_queries"] = len(dataset)  # reflect original count

    # Optional RAGAS metrics
    ragas_scores: Optional[dict] = None
    if args.ragas:
        print("\n  Computing RAGAS trend metrics...")
        ragas_scores = run_ragas_metrics(results)
        if "error" in ragas_scores:
            print(f"  RAGAS skipped: {ragas_scores['error']}")
        else:
            print(f"  RAGAS scores: {ragas_scores}")

    # Write outputs
    write_results(metrics, ragas_scores, args.output, timestamp)
    write_errors(results, args.errors)

    # Print summary
    print_summary(metrics)

    # Return exit code: 0 if all targets met, 1 otherwise
    all_pass = all(metrics["passes_targets"].values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
