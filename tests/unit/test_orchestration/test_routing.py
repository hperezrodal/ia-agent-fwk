"""Tests for conditional routing (alias file for test_conditional.py).

All routing tests are in test_conditional.py. This file exists
to satisfy the task requirement for a test_routing.py file.
"""

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
class TestConditionalRouting:
    async def test_conditional_router_matches_first_route(self):
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(config=config, provider=provider, output=f"handled by {config.name}")

        routes = {
            "route_a": WorkflowStep(name="step_a", agent_name="agent_a"),
            "route_b": WorkflowStep(name="step_b", agent_name="agent_b"),
        }

        workflow = ConditionalWorkflow(
            router=lambda x: "route_a",
            routes=routes,
            agent_factory=factory,
        )
        result = await workflow.execute("test input")

        assert result.metadata is not None
        assert result.metadata["route_key"] == "route_a"

    async def test_conditional_router_fallback(self):
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(config=config, provider=provider, output="fallback")

        routes = {"known": WorkflowStep(name="known_step", agent_name="known_agent")}
        default = WorkflowStep(name="default_step", agent_name="default_agent")

        workflow = ConditionalWorkflow(
            router=lambda x: "unknown",
            routes=routes,
            agent_factory=factory,
            default=default,
        )
        result = await workflow.execute("test")
        assert result.step_results[0].agent_name == "default_agent"

    async def test_conditional_router_no_match_no_fallback(self):
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(config=config, provider=provider)

        routes = {"only": WorkflowStep(name="only_step", agent_name="only_agent")}

        workflow = ConditionalWorkflow(
            router=lambda x: "missing",
            routes=routes,
            agent_factory=factory,
        )
        with pytest.raises(OrchestrationError):
            await workflow.execute("test")

    async def test_conditional_router_callable_condition(self):
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(config=config, provider=provider, output="result")

        routes = {
            "short": WorkflowStep(name="short_step", agent_name="short_agent"),
            "long": WorkflowStep(name="long_step", agent_name="long_agent"),
        }

        def length_router(text: str) -> str:
            return "long" if len(text) > 20 else "short"

        workflow = ConditionalWorkflow(
            router=length_router,
            routes=routes,
            agent_factory=factory,
        )

        result = await workflow.execute("hi")
        assert result.metadata is not None
        assert result.metadata["route_key"] == "short"
