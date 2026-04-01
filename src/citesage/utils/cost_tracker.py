"""Per-query LLM cost tracker for CiteSage.

Usage
-----
``model_usage`` is the dict stored in RAGState under the ``model_usage`` key.
Each entry is keyed by the full model ID and contains::

    {
        "input_tokens":  int,
        "output_tokens": int,
        "calls":         int,
    }

Call :meth:`CostTracker.compute` at the end of a query to get a
:class:`QueryCost` with per-model breakdown and a total USD estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ModelCost:
    """Cost contribution from one model during a single query."""

    model: str
    input_tokens: int
    output_tokens: int
    calls: int
    cost_usd: float


@dataclass
class QueryCost:
    """Aggregated cost for one full pipeline run."""

    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    model_breakdown: list[ModelCost] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Formatting helpers                                                    #
    # ------------------------------------------------------------------ #

    def format_summary(self) -> str:
        """Return a one-line human-readable cost summary.

        Example::

            Query cost: $0.0043 (1 Sonnet call, 2 Haiku calls)
        """
        call_parts: list[str] = []
        for mc in self.model_breakdown:
            label = _model_label(mc.model)
            noun = "call" if mc.calls == 1 else "calls"
            call_parts.append(f"{mc.calls} {label} {noun}")
        calls_str = ", ".join(call_parts) if call_parts else "0 LLM calls"
        return f"Query cost: ${self.total_cost_usd:.4f} ({calls_str})"

    def format_breakdown(self) -> str:
        """Return a multi-line breakdown table (for verbose output)."""
        if not self.model_breakdown:
            return "  No LLM calls."
        lines: list[str] = []
        for mc in self.model_breakdown:
            lines.append(
                f"  {mc.model}: {mc.calls} call(s), "
                f"{mc.input_tokens} in / {mc.output_tokens} out tokens "
                f"= ${mc.cost_usd:.4f}"
            )
        lines.append(
            f"  Total: {self.total_input_tokens} in / "
            f"{self.total_output_tokens} out tokens = ${self.total_cost_usd:.4f}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Stateless utility for computing query cost from model_usage state.

    Parameters
    ----------
    pricing:
        Mapping of model ID → ``{"input_per_million": float, "output_per_million": float}``.
        Typically comes from ``get_settings().pricing``.
    """

    def __init__(self, pricing: dict) -> None:
        self._pricing = pricing  # raw dict from config

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def compute(self, model_usage: dict) -> QueryCost:
        """Compute :class:`QueryCost` from *model_usage* state dict.

        Parameters
        ----------
        model_usage:
            Dict stored in ``RAGState["model_usage"]``, e.g.::

                {
                    "claude-sonnet-4-20250514": {
                        "input_tokens": 1200,
                        "output_tokens": 450,
                        "calls": 1,
                    },
                    ...
                }
        """
        breakdown: list[ModelCost] = []
        total_cost = 0.0
        total_in = 0
        total_out = 0

        for model_id, usage in model_usage.items():
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            calls = usage.get("calls", 0)
            cost = self._model_cost(model_id, in_tok, out_tok)

            breakdown.append(
                ModelCost(
                    model=model_id,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    calls=calls,
                    cost_usd=cost,
                )
            )
            total_cost += cost
            total_in += in_tok
            total_out += out_tok

        # Sort by cost descending so the most expensive model comes first.
        breakdown.sort(key=lambda mc: mc.cost_usd, reverse=True)

        return QueryCost(
            total_cost_usd=total_cost,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            model_breakdown=breakdown,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                             #
    # ------------------------------------------------------------------ #

    def _model_cost(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        pricing = self._pricing.get(model_id)
        if not pricing:
            return 0.0
        if isinstance(pricing, dict):
            in_rate = pricing.get("input_per_million", 0.0)
            out_rate = pricing.get("output_per_million", 0.0)
        else:
            # Pydantic model
            in_rate = pricing.input_per_million
            out_rate = pricing.output_per_million
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_label(model_id: str) -> str:
    """Map a full model ID to a short display name."""
    lower = model_id.lower()
    if "opus" in lower:
        return "Opus"
    if "sonnet" in lower:
        return "Sonnet"
    if "haiku" in lower:
        return "Haiku"
    return model_id
