"""Unit tests for the eval budget guard: partial-run banner + cost estimate.

GAPS.md #12. A budget-capped run stops early and otherwise produces a report
that reads exactly like a full one, so the loud banner is the only thing
standing between a partial run and a bad baseline being committed. These tests
pin that behaviour (and the projection arithmetic) without spending anything.
"""

from __future__ import annotations

import json

import pytest

from citesage.evaluation.run_eval import estimate_cost, print_summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics(done: int, planned: int) -> dict:
    """Minimal metrics dict shaped like compute_metrics() output."""
    return {
        "items_completed": done,
        "total_queries": planned,
        "aggregate": {
            "accuracy": 0.7,
            "citation_precision": 0.25,
            "citation_all_correct_rate": 0.18,
            "decline_recall": 0.6,
            "decline_precision": 0.46,
        },
        "targets": {
            "accuracy": 0.85,
            "citation_precision": 0.9,
            "decline_recall": 0.85,
            "p95_latency_ms": 5000,
        },
        "passes_targets": {
            "accuracy": False,
            "citation_precision": False,
            "decline_recall": False,
            "p95_latency_ms": False,
        },
        "latency": {"p50_ms": 7574, "p95_ms": 122823},
        "by_category": {
            "factual_lookup": {
                "accuracy": 0.849,
                "correct": 28,
                "partial": 2,
                "count": 33,
            }
        },
    }


def _report(path, in_tok: int, out_tok: int, queries: int = 65):
    """Write a minimal eval report JSON and return its path."""
    path.write_text(
        json.dumps(
            {
                "items_completed": queries,
                "token_usage": {
                    "total_input_tokens": in_tok,
                    "total_output_tokens": out_tok,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Partial-run banner
# ---------------------------------------------------------------------------


def test_partial_run_prints_loud_banner(capsys):
    print_summary(_metrics(done=18, planned=65))
    out = capsys.readouterr().out
    assert "PARTIAL RUN" in out
    assert "18/65" in out
    assert "47 queries never ran" in out
    # It must actively discourage using the numbers as a baseline.
    assert "NOT" in out and "baseline" in out


def test_full_run_has_no_banner(capsys):
    print_summary(_metrics(done=65, planned=65))
    out = capsys.readouterr().out
    assert "PARTIAL RUN" not in out
    assert "65 / 65" in out


def test_banner_is_ascii_only(capsys):
    """The banner must survive a Windows cp1252 console."""
    print_summary(_metrics(done=1, planned=65))
    out = capsys.readouterr().out
    banner = [ln for ln in out.splitlines() if "PARTIAL RUN" in ln][0]
    banner.encode("ascii")  # raises UnicodeEncodeError on regression


def test_off_by_one_partial_still_warns(capsys):
    """64/65 is still partial — the guard must not round it away."""
    print_summary(_metrics(done=64, planned=65))
    assert "PARTIAL RUN" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------


def test_estimate_missing_report_fails(tmp_path, capsys):
    assert estimate_cost(tmp_path / "nope.json", 65, 2.0) == 1
    assert "No report" in capsys.readouterr().err


def test_estimate_report_without_tokens_fails(tmp_path, capsys):
    p = tmp_path / "empty.json"
    p.write_text(
        json.dumps({"items_completed": 65, "token_usage": {}}), encoding="utf-8"
    )
    assert estimate_cost(p, 65, 2.0) == 1
    assert "no token usage" in capsys.readouterr().err


def test_estimate_scales_per_query_usage(tmp_path, capsys):
    """Projection scales measured per-query usage to the requested count."""
    p = _report(tmp_path / "r.json", in_tok=6500, out_tok=13000, queries=65)
    estimate_cost(p, 130, 2.0)  # double the queries
    out = capsys.readouterr().out
    assert "100 in / 200 out tokens" in out  # per query
    assert "13,000 in / 26,000 out" in out  # projected for 130


@pytest.fixture
def priced(monkeypatch):
    """Pin generator/grader pricing so budget assertions don't depend on config.yaml.

    Uses Sonnet/Haiku list prices ($3/$15 and $0.25/$1.25 per million).
    """

    class _Price:
        def __init__(self, i: float, o: float) -> None:
            self.input_per_million = i
            self.output_per_million = o

    class _Pricing:
        def get_model(self, name: str):
            return {
                "gen-model": _Price(3.0, 15.0),
                "grader-model": _Price(0.25, 1.25),
            }.get(name)

    class _Models:
        generator = "gen-model"
        grader = "grader-model"

    class _Settings:
        pricing = _Pricing()
        models = _Models()

    monkeypatch.setattr(
        "citesage.evaluation.run_eval.get_settings", lambda *a, **k: _Settings()
    )


@pytest.mark.parametrize(
    "budget,expect_code,marker",
    [(2.0, 0, "fits"), (1.0, 1, "OVER BUDGET")],
)
def test_estimate_exit_code_tracks_budget(
    tmp_path, capsys, priced, budget, expect_code, marker
):
    """Exit code is the machine-readable gate: 0 fits, 1 over."""
    # 78,452 in / 85,999 out at $3/$15 per M = $1.53 upper bound.
    p = _report(tmp_path / "r.json", in_tok=78452, out_tok=85999)
    code = estimate_cost(p, 65, budget)
    out = capsys.readouterr().out
    assert code == expect_code
    assert marker in out
    assert "$1.53" in out


def test_estimate_reports_cheaper_grader_lower_bound(tmp_path, capsys, priced):
    """Both bounds are shown so the $2 cap isn't judged on the worst case alone."""
    p = _report(tmp_path / "r.json", in_tok=78452, out_tok=85999)
    estimate_cost(p, 65, 2.0)
    out = capsys.readouterr().out
    assert "Upper bound" in out and "Lower bound" in out
    assert "$0.13" in out  # same tokens at grader rates


def test_estimate_unpriced_generator_fails_loudly(tmp_path, capsys, monkeypatch):
    """An unpriced model must not silently project $0.00 and look affordable."""

    class _Settings:
        class pricing:  # noqa: N801
            @staticmethod
            def get_model(name: str):
                return None

        class models:  # noqa: N801
            generator = "mystery-model"
            grader = "mystery-model"

    monkeypatch.setattr(
        "citesage.evaluation.run_eval.get_settings", lambda *a, **k: _Settings()
    )
    p = _report(tmp_path / "r.json", in_tok=78452, out_tok=85999)
    assert estimate_cost(p, 65, 2.0) == 1
    assert "No pricing configured" in capsys.readouterr().out
