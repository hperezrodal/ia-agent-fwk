"""Token cost estimation for LLM calls.

Provides a ``CostEstimator`` class with a default pricing table for common
models.  Ollama (self-hosted) always reports zero cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ia_agent_fwk.llm.models import TokenUsage

from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ModelPricing:
    """Per-token pricing in USD."""

    input_per_token: float
    output_per_token: float


# Default pricing (USD per token).  Updated as of 2026-03.
_DEFAULT_PRICING: dict[str, _ModelPricing] = {
    # OpenAI
    "gpt-4o": _ModelPricing(input_per_token=2.50 / 1_000_000, output_per_token=10.00 / 1_000_000),
    "gpt-4o-mini": _ModelPricing(input_per_token=0.15 / 1_000_000, output_per_token=0.60 / 1_000_000),
    "gpt-4": _ModelPricing(input_per_token=30.00 / 1_000_000, output_per_token=60.00 / 1_000_000),
    "gpt-4-turbo": _ModelPricing(input_per_token=10.00 / 1_000_000, output_per_token=30.00 / 1_000_000),
    "gpt-3.5-turbo": _ModelPricing(input_per_token=0.50 / 1_000_000, output_per_token=1.50 / 1_000_000),
    # Anthropic
    "claude-sonnet-4-20250514": _ModelPricing(input_per_token=3.00 / 1_000_000, output_per_token=15.00 / 1_000_000),
    "claude-3-5-sonnet-20241022": _ModelPricing(input_per_token=3.00 / 1_000_000, output_per_token=15.00 / 1_000_000),
    "claude-3-opus-20240229": _ModelPricing(input_per_token=15.00 / 1_000_000, output_per_token=75.00 / 1_000_000),
    "claude-3-haiku-20240307": _ModelPricing(input_per_token=0.25 / 1_000_000, output_per_token=1.25 / 1_000_000),
}

# Providers whose models are always zero-cost (self-hosted).
_ZERO_COST_PROVIDERS: frozenset[str] = frozenset({"ollama"})


@dataclass
class CostEstimator:
    """Estimate the USD cost of an LLM call from token usage data.

    Parameters
    ----------
    pricing:
        Custom pricing overrides.  Keys are model names; values are
        ``_ModelPricing`` instances with ``input_per_token`` and
        ``output_per_token``.

    """

    pricing: dict[str, _ModelPricing] = field(default_factory=lambda: dict(_DEFAULT_PRICING))

    def estimate(
        self,
        usage: TokenUsage,
        model: str,
        provider: str = "",
    ) -> float:
        """Return the estimated cost in USD.

        Parameters
        ----------
        usage:
            Token counts for the call.
        model:
            Model name (e.g. ``"gpt-4o"``).
        provider:
            Provider name.  Zero-cost providers (e.g. ``"ollama"``) always
            return ``0.0``.

        """
        if provider in _ZERO_COST_PROVIDERS:
            return 0.0

        pricing = self.pricing.get(model)
        if pricing is None:
            logger.warning("No pricing data for model '%s'; returning 0.0", model)
            return 0.0

        cost = usage.prompt_tokens * pricing.input_per_token + usage.completion_tokens * pricing.output_per_token
        collector = get_metrics_collector()
        collector.observe(
            "llm_cost_dollars",
            cost,
            labels={"provider": provider, "model": model},
        )
        return cost
