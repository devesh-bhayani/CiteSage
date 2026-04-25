"""CiteSage Regression Checker — Phase 3.

Compares a current evaluation report against a baseline and flags any metric
that has regressed by more than the allowed tolerance.

Usage
-----
    python -m citesage.evaluation.check_regression \\
        --baseline reports/baseline_scores.json \\
        --current  reports/eval_report.json

Exit codes
----------
    0 — all metrics within tolerance (no regression)
    1 — one or more metrics regressed beyond threshold
    2 — bad arguments / file not found
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tolerance — flag WARNING when a metric drops more than this fraction
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE = 0.05  # 5 % relative drop triggers a WARNING


# ---------------------------------------------------------------------------
# Metric definitions: (report_key_path, display_name, higher_is_better)
# ---------------------------------------------------------------------------

# Key path is dot-separated; e.g. "aggregate.accuracy" → report["aggregate"]["accuracy"]
_METRICS: list[tuple[str, str, bool]] = [
    ("aggregate.accuracy", "Accuracy", True),
    ("aggregate.citation_precision", "Citation Precision", True),
    ("aggregate.decline_recall", "Decline Recall", True),
    ("aggregate.decline_precision", "Decline Precision", True),
    ("fast_path_ratio", "Fast-path Ratio", True),  # drop = slower pipeline
    ("latency.p95_ms", "P95 Latency (ms)", False),  # lower is better
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricDiff:
    key: str
    display: str
    baseline: float
    current: float
    delta: float  # current - baseline
    relative_change: float  # (current - baseline) / baseline
    regressed: bool
    higher_is_better: bool

    @property
    def direction_symbol(self) -> str:
        if self.delta > 0:
            return "+"
        if self.delta < 0:
            return "-"
        return " "

    @property
    def status_label(self) -> str:
        if self.regressed:
            return "REGRESSED"
        # Improvement worth noting
        if self.higher_is_better and self.delta > 0:
            return "improved"
        if not self.higher_is_better and self.delta < 0:
            return "improved"
        return "ok"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nested(data: dict, key_path: str) -> Optional[float]:
    """Resolve a dot-separated key path in *data*, returning None if missing."""
    parts = key_path.split(".")
    node: object = data
    for part in parts:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    try:
        return float(node)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _compare_metric(
    key: str,
    display: str,
    higher_is_better: bool,
    baseline_val: float,
    current_val: float,
    tolerance: float,
) -> MetricDiff:
    delta = current_val - baseline_val
    relative = delta / baseline_val if baseline_val != 0 else 0.0

    # Regression = metric moved in the *wrong* direction by more than tolerance
    if higher_is_better:
        regressed = delta < 0 and abs(relative) > tolerance
    else:
        regressed = delta > 0 and abs(relative) > tolerance

    return MetricDiff(
        key=key,
        display=display,
        baseline=baseline_val,
        current=current_val,
        delta=delta,
        relative_change=relative,
        regressed=regressed,
        higher_is_better=higher_is_better,
    )


# ---------------------------------------------------------------------------
# Per-category regression
# ---------------------------------------------------------------------------


def _compare_categories(
    baseline: dict,
    current: dict,
    tolerance: float,
) -> list[MetricDiff]:
    """Compare per-category accuracy between two reports."""
    diffs: list[MetricDiff] = []
    baseline_cats: dict = baseline.get("by_category", {})
    current_cats: dict = current.get("by_category", {})

    all_cats = set(baseline_cats) | set(current_cats)
    for cat in sorted(all_cats):
        b_acc = baseline_cats.get(cat, {}).get("accuracy")
        c_acc = current_cats.get(cat, {}).get("accuracy")

        if b_acc is None or c_acc is None:
            continue

        diff = _compare_metric(
            key=f"by_category.{cat}.accuracy",
            display=f"  {cat} accuracy",
            higher_is_better=True,
            baseline_val=float(b_acc),
            current_val=float(c_acc),
            tolerance=tolerance,
        )
        diffs.append(diff)

    return diffs


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------


def compare_reports(
    baseline: dict,
    current: dict,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[list[MetricDiff], bool]:
    """Return (all_diffs, any_regression).

    Compares all tracked metrics and per-category accuracy.
    """
    diffs: list[MetricDiff] = []

    for key, display, higher_is_better in _METRICS:
        b_val = _get_nested(baseline, key)
        c_val = _get_nested(current, key)

        if b_val is None or c_val is None:
            # Metric missing from one report — skip silently
            continue

        diffs.append(
            _compare_metric(
                key=key,
                display=display,
                higher_is_better=higher_is_better,
                baseline_val=b_val,
                current_val=c_val,
                tolerance=tolerance,
            )
        )

    diffs.extend(_compare_categories(baseline, current, tolerance))

    any_regression = any(d.regressed for d in diffs)
    return diffs, any_regression


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_val(key: str, val: float) -> str:
    """Format a metric value for display."""
    if "latency" in key or "ms" in key:
        return f"{val:.1f}ms"
    return f"{val:.1%}"


def print_report(
    diffs: list[MetricDiff],
    any_regression: bool,
    baseline_path: Path,
    current_path: Path,
    tolerance: float,
) -> None:
    print("\n" + "=" * 64)
    print("  CITESAGE REGRESSION CHECK")
    print("=" * 64)
    print(f"  Baseline : {baseline_path}")
    print(f"  Current  : {current_path}")
    print(f"  Tolerance: {tolerance:.0%} relative drop\n")

    print(f"  {'Metric':<30} {'Baseline':>10} {'Current':>10}  {'Change':>8}  Status")
    print("  " + "-" * 60)

    for d in diffs:
        b_str = _fmt_val(d.key, d.baseline)
        c_str = _fmt_val(d.key, d.current)
        chg = _fmt_val(d.key, abs(d.delta))
        sign = "+" if d.delta >= 0 else "-"
        status = d.status_label
        flag = " <-- WARNING" if d.regressed else ""
        print(
            f"  {d.display:<30} {b_str:>10} {c_str:>10}  "
            f"{sign}{chg:>7}  {status}{flag}"
        )

    print()
    if any_regression:
        regressions = [d for d in diffs if d.regressed]
        print(f"  *** REGRESSION DETECTED: {len(regressions)} metric(s) degraded ***")
    else:
        print("  *** No regressions — all metrics within tolerance ***")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two CiteSage eval reports and flag regressions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("reports/baseline_scores.json"),
        help="Path to the baseline eval report",
    )
    p.add_argument(
        "--current",
        type=Path,
        default=Path("reports/eval_report.json"),
        help="Path to the current eval report to compare",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Relative drop threshold that triggers a WARNING (e.g. 0.05 = 5%%)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns 0 (pass) or 1 (regression detected) or 2 (error)."""
    # Windows stdout may default to cp1252; force UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args(argv)

    for label, path in [("baseline", args.baseline), ("current", args.current)]:
        if not path.exists():
            print(f"ERROR: {label} report not found: {path}", file=sys.stderr)
            return 2

    baseline = _load_report(args.baseline)
    current = _load_report(args.current)

    diffs, any_regression = compare_reports(baseline, current, args.tolerance)
    print_report(diffs, any_regression, args.baseline, args.current, args.tolerance)

    return 1 if any_regression else 0


if __name__ == "__main__":
    sys.exit(main())
