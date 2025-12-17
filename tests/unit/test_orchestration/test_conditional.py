"""Tests for conditional routing workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ia_agent_fwk.orchestration.conditional import ConditionalWorkflow
from ia_agent_fwk.orchestration.exceptions import OrchestrationError
from ia_agent_fwk.orchestration.models import WorkflowStep

from .conftest import MockAgent, OrcMockLLMProvider

if TYPE_CHECKING:
    from ia_agent_fwk.agents.config import AgentConfig


@pytest.mark.unit
class TestConditionalWorkflow:
    @pytest.fixture
    def routes(self):
        return {
            "billing": WorkflowStep(name="billing_step", agent_name="billing_agent"),
            "technical": WorkflowStep(name="technical_step", agent_name="tech_agent"),
            "general": WorkflowStep(name="general_step", agent_name="general_agent"),
        }

    @pytest.fixture
    def default_step(self):
        return WorkflowStep(name="fallback_step", agent_name="fallback_agent")

    @pytest.fixture
    def cond_factory(self):
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=provider,
                output=f"handled by {config.name}",
            )

        return factory

    async def test_conditional_router_matches_first_route(self, routes, cond_factory):
        def router(input_text: str) -> str:
            if "bill" in input_text.lower():
                return "billing"
            if "tech" in input_text.lower():
                return "technical"
            return "general"

        workflow = ConditionalWorkflow(
            router=router,
            routes=routes,
            agent_factory=cond_factory,
        )
        result = await workflow.execute("I have a billing question")

        assert result.error is None
        assert result.metadata is not None
        assert result.metadata["route_key"] == "billing"
        assert len(result.step_results) == 1
        assert result.step_results[0].agent_name == "billing_agent"

    async def test_conditional_router_technical_route(self, routes, cond_factory):
        def router(input_text: str) -> str:  # noqa: ARG001
            return "technical"

        workflow = ConditionalWorkflow(
            router=router,
            routes=routes,
            agent_factory=cond_factory,
        )
        result = await workflow.execute("How do I configure X?")

        assert result.metadata is not None
        assert result.metadata["route_key"] == "technical"
        assert result.step_results[0].agent_name == "tech_agent"

    async def test_conditional_router_fallback(self, routes, default_step, cond_factory):
        def router(input_text: str) -> str:  # noqa: ARG001
            return "unknown_route"

        workflow = ConditionalWorkflow(
            router=router,
            routes=routes,
            agent_factory=cond_factory,
            default=default_step,
        )
        result = await workflow.execute("Something unexpected")

        assert result.error is None
        assert result.step_results[0].agent_name == "fallback_agent"

    async def test_conditional_router_no_match_no_fallback(self, routes, cond_factory):
        def router(input_text: str) -> str:  # noqa: ARG001
            return "nonexistent"

        workflow = ConditionalWorkflow(
            router=router,
            routes=routes,
            agent_factory=cond_factory,
        )

        with pytest.raises(OrchestrationError, match="No route found"):
            await workflow.execute("Something random")

    async def test_conditional_router_callable_condition(self, cond_factory):
        """Test with a callable that does complex routing logic."""
        routes = {
            "positive": WorkflowStep(name="positive_step", agent_name="positive_agent"),
            "negative": WorkflowStep(name="negative_step", agent_name="negative_agent"),
        }

        def sentiment_router(text: str) -> str:
            positive_words = {"great", "awesome", "good", "love"}
            words = set(text.lower().split())
            if words & positive_words:
                return "positive"
            return "negative"

        workflow = ConditionalWorkflow(
            router=sentiment_router,
            routes=routes,
            agent_factory=cond_factory,
        )

        result = await workflow.execute("This is a great product!")
        assert result.metadata is not None
        assert result.metadata["route_key"] == "positive"

        # Create a new workflow instance (agent is single-use)
        workflow2 = ConditionalWorkflow(
            router=sentiment_router,
            routes=routes,
            agent_factory=cond_factory,
        )
        result2 = await workflow2.execute("This is terrible")
        assert result2.metadata is not None
        assert result2.metadata["route_key"] == "negative"

    async def test_conditional_orchestrator_type(self, routes, cond_factory):
        workflow = ConditionalWorkflow(
            router=lambda x: "billing",
            routes=routes,
            agent_factory=cond_factory,
        )
        assert workflow.orchestrator_type == "conditional"
