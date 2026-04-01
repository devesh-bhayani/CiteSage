"""Unit tests for CostTracker.

All tests are pure-Python — no API key, no LLM calls.
"""

from __future__ import annotations

import pytest

from citesage.utils.cost_tracker import CostTracker, QueryCost, _model_label


# Pricing fixture matching the values in config.yaml
PRICING = {
    "claude-sonnet-4-20250514": {
        "input_per_million": 3.0,
        "output_per_million": 15.0,
    },
    "claude-haiku-4-5-20251001": {
        "input_per_million": 0.25,
        "output_per_million": 1.25,
    },
}


def _tracker() -> CostTracker:
    return CostTracker(PRICING)


# ---------------------------------------------------------------------------
# CostTracker.compute — basic arithmetic
# ---------------------------------------------------------------------------


class TestCostTrackerCompute:
    def test_empty_model_usage_returns_zero(self):
        result = _tracker().compute({})
        assert result.total_cost_usd == pytest.approx(0.0)
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
        assert result.model_breakdown == []

    def test_single_sonnet_call(self):
        model_usage = {
            "claude-sonnet-4-20250514": {
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "calls": 1,
            }
        }
        result = _tracker().compute(model_usage)
        # 1M input × $3 + 1M output × $15 = $18
        assert result.total_cost_usd == pytest.approx(18.0)
        assert result.total_input_tokens == 1_000_000
        assert result.total_output_tokens == 1_000_000
        assert len(result.model_breakdown) == 1
        assert result.model_breakdown[0].model == "claude-sonnet-4-20250514"
        assert result.model_breakdown[0].cost_usd == pytest.approx(18.0)
        assert result.model_breakdown[0].calls == 1

    def test_single_haiku_call(self):
        model_usage = {
            "claude-haiku-4-5-20251001": {
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "calls": 2,
            }
        }
        result = _tracker().compute(model_usage)
        # 1M input × $0.25 + 1M output × $1.25 = $1.50
        assert result.total_cost_usd == pytest.approx(1.50)
        assert result.model_breakdown[0].calls == 2

    def test_mixed_models_sum_correctly(self):
        model_usage = {
            "claude-sonnet-4-20250514": {
                "input_tokens": 500,
                "output_tokens": 200,
                "calls": 1,
            },
            "claude-haiku-4-5-20251001": {
                "input_tokens": 800,
                "output_tokens": 100,
                "calls": 2,
            },
        }
        result = _tracker().compute(model_usage)
        # Sonnet: (500×3 + 200×15) / 1e6 = (1500 + 3000) / 1e6 = 0.0045
        # Haiku:  (800×0.25 + 100×1.25) / 1e6 = (200 + 125) / 1e6 = 0.000325
        expected = (500 * 3 + 200 * 15 + 800 * 0.25 + 100 * 1.25) / 1_000_000
        assert result.total_cost_usd == pytest.approx(expected)
        assert result.total_input_tokens == 1300
        assert result.total_output_tokens == 300

    def test_unknown_model_costs_zero(self):
        model_usage = {
            "claude-unknown-model": {
                "input_tokens": 500,
                "output_tokens": 200,
                "calls": 1,
            }
        }
        result = _tracker().compute(model_usage)
        assert result.total_cost_usd == pytest.approx(0.0)
        assert len(result.model_breakdown) == 1
        assert result.model_breakdown[0].cost_usd == pytest.approx(0.0)

    def test_breakdown_sorted_by_cost_descending(self):
        model_usage = {
            "claude-haiku-4-5-20251001": {
                "input_tokens": 100,
                "output_tokens": 50,
                "calls": 1,
            },
            "claude-sonnet-4-20250514": {
                "input_tokens": 100,
                "output_tokens": 50,
                "calls": 1,
            },
        }
        result = _tracker().compute(model_usage)
        # Sonnet is always more expensive → must come first
        assert result.model_breakdown[0].model == "claude-sonnet-4-20250514"
        assert result.model_breakdown[1].model == "claude-haiku-4-5-20251001"

    def test_zero_tokens(self):
        model_usage = {
            "claude-sonnet-4-20250514": {
                "input_tokens": 0,
                "output_tokens": 0,
                "calls": 1,
            }
        }
        result = _tracker().compute(model_usage)
        assert result.total_cost_usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# QueryCost.format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def _cost(self, model_usage: dict) -> QueryCost:
        return _tracker().compute(model_usage)

    def test_single_sonnet_call_summary(self):
        model_usage = {
            "claude-sonnet-4-20250514": {
                "input_tokens": 1200,
                "output_tokens": 450,
                "calls": 1,
            }
        }
        summary = self._cost(model_usage).format_summary()
        assert "Query cost:" in summary
        assert "$" in summary
        assert "1 Sonnet call" in summary

    def test_plural_haiku_calls(self):
        model_usage = {
            "claude-haiku-4-5-20251001": {
                "input_tokens": 800,
                "output_tokens": 120,
                "calls": 3,
            }
        }
        summary = self._cost(model_usage).format_summary()
        assert "3 Haiku calls" in summary

    def test_mixed_models_summary(self):
        model_usage = {
            "claude-sonnet-4-20250514": {
                "input_tokens": 1200,
                "output_tokens": 450,
                "calls": 1,
            },
            "claude-haiku-4-5-20251001": {
                "input_tokens": 800,
                "output_tokens": 120,
                "calls": 2,
            },
        }
        summary = self._cost(model_usage).format_summary()
        assert "1 Sonnet call" in summary
        assert "2 Haiku calls" in summary

    def test_empty_usage_shows_zero_calls(self):
        summary = self._cost({}).format_summary()
        assert "0 LLM calls" in summary

    def test_cost_formatted_with_dollar_sign_and_four_decimals(self):
        model_usage = {
            "claude-sonnet-4-20250514": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "calls": 1,
            }
        }
        summary = self._cost(model_usage).format_summary()
        # e.g. "$0.0108"
        import re

        assert re.search(r"\$\d+\.\d{4}", summary), f"No 4-decimal cost in: {summary}"


# ---------------------------------------------------------------------------
# QueryCost.format_breakdown
# ---------------------------------------------------------------------------


class TestFormatBreakdown:
    def test_breakdown_contains_model_id(self):
        model_usage = {
            "claude-sonnet-4-20250514": {
                "input_tokens": 500,
                "output_tokens": 200,
                "calls": 1,
            }
        }
        breakdown = _tracker().compute(model_usage).format_breakdown()
        assert "claude-sonnet-4-20250514" in breakdown
        assert "Total:" in breakdown

    def test_empty_breakdown(self):
        assert _tracker().compute({}).format_breakdown() == "  No LLM calls."


# ---------------------------------------------------------------------------
# _model_label helper
# ---------------------------------------------------------------------------


class TestModelLabel:
    def test_sonnet_label(self):
        assert _model_label("claude-sonnet-4-20250514") == "Sonnet"

    def test_haiku_label(self):
        assert _model_label("claude-haiku-4-5-20251001") == "Haiku"

    def test_opus_label(self):
        assert _model_label("claude-opus-4-20250514") == "Opus"

    def test_unknown_returns_model_id(self):
        assert _model_label("some-unknown-model") == "some-unknown-model"


# ---------------------------------------------------------------------------
# CostTracker with Pydantic PricingModelConfig objects
# ---------------------------------------------------------------------------


class TestPricingModelConfigObjects:
    """CostTracker should also accept Pydantic model objects, not just dicts."""

    def test_with_pydantic_pricing_objects(self):
        from citesage.config import PricingModelConfig

        pricing_objects = {
            "claude-sonnet-4-20250514": PricingModelConfig(
                input_per_million=3.0, output_per_million=15.0
            )
        }
        tracker = CostTracker(pricing_objects)
        model_usage = {
            "claude-sonnet-4-20250514": {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "calls": 1,
            }
        }
        result = tracker.compute(model_usage)
        # 1M input × $3.0 / 1M = $3.0
        assert result.total_cost_usd == pytest.approx(3.0)
