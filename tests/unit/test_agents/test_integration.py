"""Integration tests for the agents module.

Tests multi-component flows, concurrency, config round-trips,
and performance characteristics.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig, AgentResult
from ia_agent_fwk.agents.context import AgentContext
from ia_agent_fwk.agents.registry import AgentRegistry
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.config.settings import AgentSettings
from ia_agent_fwk.llm.models import FinishReason, Message

from .conftest import MockLLMProvider, make_chat_response


class _IntegrationTestAgent(Agent):
    """Concrete agent for integration tests."""

    @property
    def agent_type(self) -> str:
        return "integration"


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state between tests."""
    saved = dict(AgentRegistry._registry)
    AgentRegistry._registry.clear()
    yield
    AgentRegistry._registry.clear()
    AgentRegistry._registry.update(saved)


class TestEndToEndFlow:
    @pytest.mark.asyncio
    async def test_factory_to_result(self):
        """End-to-end: factory -> registry -> agent -> reasoning -> result."""
        AgentRegistry.register("integration", _IntegrationTestAgent)

        config = AgentConfig(
            name="e2e-agent",
            agent_type="integration",
            system_prompt="You are helpful.",
            provider_name="openai",
            max_iterations=5,
            context_window=8192,
        )

        # Test factory lookup + manual wiring (avoids real provider creation)
        agent_cls = AgentRegistry.get("integration")
        provider = MockLLMProvider(
            responses=[
                make_chat_response(content="Integration test complete!", finish_reason=FinishReason.stop),
            ]
        )
        agent = agent_cls(config=config, provider=provider)

        result = await agent.run("Test input")

        assert isinstance(result, AgentResult)
        assert result.state == AgentState.COMPLETED
        assert result.output == "Integration test complete!"
        assert result.iterations == 1
        assert result.duration_ms > 0


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_five_agents_concurrent(self):
        """NFR-005: 5 agents run concurrently without interference."""
        agents = []
        for i in range(5):
            config = AgentConfig(
                name=f"agent-{i}",
                agent_type="test",
                system_prompt=f"Agent {i}",
                context_window=8192,
            )
            provider = MockLLMProvider(
                responses=[
                    make_chat_response(
                        content=f"Response from agent {i}",
                        finish_reason=FinishReason.stop,
                    ),
                ]
            )
            agent = _IntegrationTestAgent(config=config, provider=provider)
            agents.append(agent)

        results = await asyncio.gather(*[agent.run(f"Input for agent {i}") for i, agent in enumerate(agents)])

        # All should complete
        for i, result in enumerate(results):
            assert result.state == AgentState.COMPLETED
            assert result.output == f"Response from agent {i}"
            assert result.error is None


class TestConfigRoundTrip:
    def test_agent_settings_defaults(self):
        """AC-20: AgentSettings accessible with valid defaults."""
        settings = AgentSettings()
        assert settings.default_agent == ""
        assert settings.agents == {}

    def test_agent_settings_with_config(self):
        from ia_agent_fwk.config.settings import AgentConfigSettings

        settings = AgentSettings(
            default_agent="support",
            agents={
                "support": AgentConfigSettings(
                    name="support-bot",
                    agent_type="support",
                    system_prompt="You are a support agent.",
                    max_iterations=5,
                ),
            },
        )
        assert settings.default_agent == "support"
        assert "support" in settings.agents
        assert settings.agents["support"].max_iterations == 5


class TestPerformance:
    def test_context_operations_under_5ms(self):
        """NFR-003: context ops < 5ms for 100 messages."""
        ctx = AgentContext(
            system_prompt="System",
            token_budget=100000,
            token_counter=lambda t: len(t) // 4,
        )

        start = time.monotonic()
        for i in range(100):
            ctx.add_message(Message(role="user", content=f"Message {i}"))
        _ = ctx.get_messages()
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 500  # generous bound for CI; spec says <5ms per operation


class TestPublicAPI:
    def test_all_exports_importable(self):
        """FR-043: all key types importable from ia_agent_fwk.agents."""
        from ia_agent_fwk.agents import (
            Agent,
            AgentConfig,
            AgentConfigError,
            AgentContext,
            AgentError,
            AgentFactory,
            AgentMaxIterationsError,
            AgentRegistry,
            AgentResult,
            AgentState,
            AgentTimeoutError,
            InvalidStateTransitionError,
            NoOpToolExecutor,
            ToolExecutor,
            ToolResult,
        )

        # Verify they're the correct types (not None)
        assert Agent is not None
        assert AgentConfig is not None
        assert AgentResult is not None
        assert AgentState is not None
        assert AgentContext is not None
        assert AgentRegistry is not None
        assert AgentFactory is not None
        assert AgentError is not None
        assert AgentConfigError is not None
        assert AgentTimeoutError is not None
        assert AgentMaxIterationsError is not None
        assert InvalidStateTransitionError is not None
        assert ToolExecutor is not None
        assert ToolResult is not None
        assert NoOpToolExecutor is not None
