"""Tests for the CostEstimator."""

from __future__ import annotations

import pytest

from ia_agent_fwk.llm.cost import CostEstimator, _ModelPricing
from ia_agent_fwk.llm.models import TokenUsage


@pytest.fixture
def estimator():
    return CostEstimator()


class TestCostEstimator:
    def test_known_model_pricing(self, estimator):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        cost = estimator.estimate(usage, model="gpt-4o")
        # gpt-4o: input $2.50/M, output $10.00/M
        expected = 1000 * 2.50 / 1_000_000 + 500 * 10.00 / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_unknown_model_returns_zero(self, estimator):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        cost = estimator.estimate(usage, model="unknown-model-xyz")
        assert cost == 0.0

    def test_ollama_provider_zero_cost(self, estimator):
        usage = TokenUsage(prompt_tokens=10000, completion_tokens=5000)
        cost = estimator.estimate(usage, model="llama3.1", provider="ollama")
        assert cost == 0.0

    def test_zero_tokens(self, estimator):
        usage = TokenUsage()
        cost = estimator.estimate(usage, model="gpt-4o")
        assert cost == 0.0

    def test_custom_pricing(self):
        custom = {"my-model": _ModelPricing(input_per_token=0.001, output_per_token=0.002)}
        estimator = CostEstimator(pricing=custom)
        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        cost = estimator.estimate(usage, model="my-model")
        assert cost == pytest.approx(100 * 0.001 + 200 * 0.002, rel=1e-6)

    def test_anthropic_model_pricing(self, estimator):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        cost = estimator.estimate(usage, model="claude-sonnet-4-20250514")
        assert cost > 0.0

    def test_gpt35_turbo_pricing(self, estimator):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        cost = estimator.estimate(usage, model="gpt-3.5-turbo")
        assert cost > 0.0
