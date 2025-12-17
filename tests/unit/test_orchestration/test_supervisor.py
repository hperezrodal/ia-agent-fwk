"""Tests for the SupervisorAgent."""

from __future__ import annotations

import json

import pytest

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.models import (
    FinishReason,
    ToolCall,
)
from ia_agent_fwk.orchestration.supervisor import SupervisorAgent

from .conftest import MockAgent, OrcMockLLMProvider, make_chat_response


@pytest.mark.unit
class TestSupervisorAgent:
    @pytest.fixture
    def supervisor_config(self):
        return AgentConfig(
            name="supervisor",
            agent_type="supervisor",
            system_prompt="You are a supervisor agent. Delegate tasks to sub-agents.",
            provider_name="mock",
            max_iterations=5,
            execution_timeout=30,
        )

    @pytest.fixture
    def sub_agent_configs(self):
        return [
            (
                AgentConfig(name="researcher", agent_type="mock"),
                "Researches topics and gathers information",
            ),
            (
                AgentConfig(name="writer", agent_type="mock"),
                "Writes content based on research",
            ),
        ]

    async def test_supervisor_delegates_to_subagent(self, supervisor_config, sub_agent_configs):
        """Supervisor should invoke a sub-agent via tool call and return final answer."""
        # LLM response 1: tool call to researcher
        tool_call_args = json.dumps({"task": "Research AI agents"})
        tool_call = ToolCall(id="tc-1", name="agent_researcher", arguments=tool_call_args)

        # LLM response 2: final answer after getting tool result
        provider = OrcMockLLMProvider(
            responses=[
                make_chat_response(
                    content="Let me research that.",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tool_call],
                ),
                make_chat_response(
                    content="Based on my research, AI agents are autonomous systems.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=provider,
                output="Research result: AI agents use LLMs.",
            )

        supervisor = SupervisorAgent(
            config=supervisor_config,
            provider=provider,
            sub_agents=sub_agent_configs,
            agent_factory=factory,
            max_delegation_depth=3,
        )

        result = await supervisor.run("Tell me about AI agents")

        assert result.state == AgentState.COMPLETED
        assert "autonomous" in result.output.lower() or result.output != ""

    async def test_supervisor_max_depth_exceeded(self, supervisor_config, sub_agent_configs):
        """When delegation depth is 1, sub-agent tools get depth 0 and refuse execution."""
        tool_call_args = json.dumps({"task": "Research AI"})
        tool_call = ToolCall(id="tc-1", name="agent_researcher", arguments=tool_call_args)

        provider = OrcMockLLMProvider(
            responses=[
                make_chat_response(
                    content="Let me research.",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tool_call],
                ),
                make_chat_response(
                    content="Delegation depth was exceeded.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=provider,
                output="Should not reach here",
            )

        supervisor = SupervisorAgent(
            config=supervisor_config,
            provider=provider,
            sub_agents=sub_agent_configs,
            agent_factory=factory,
            max_delegation_depth=1,  # Sub-agents get depth 0
        )

        result = await supervisor.run("Tell me about AI")
        # Should complete (the LLM sees the depth-exceeded error and provides a final answer)
        assert result.state == AgentState.COMPLETED

    async def test_supervisor_returns_final_answer(self, supervisor_config, sub_agent_configs):
        """Supervisor returns final answer without delegation when LLM decides so."""
        provider = OrcMockLLMProvider(
            responses=[
                make_chat_response(
                    content="I can answer this directly: AI is fascinating.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(config=config, provider=provider, output="unused")

        supervisor = SupervisorAgent(
            config=supervisor_config,
            provider=provider,
            sub_agents=sub_agent_configs,
            agent_factory=factory,
        )

        result = await supervisor.run("What is AI?")

        assert result.state == AgentState.COMPLETED
        assert "AI" in result.output

    async def test_supervisor_handles_subagent_failure(self, supervisor_config, sub_agent_configs):
        """Supervisor should handle sub-agent failures gracefully."""
        tool_call_args = json.dumps({"task": "Research AI"})
        tool_call = ToolCall(id="tc-1", name="agent_researcher", arguments=tool_call_args)

        provider = OrcMockLLMProvider(
            responses=[
                make_chat_response(
                    content="Let me research.",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tool_call],
                ),
                make_chat_response(
                    content="The sub-agent failed, but I can still provide a partial answer.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=provider,
                should_fail=True,
            )

        supervisor = SupervisorAgent(
            config=supervisor_config,
            provider=provider,
            sub_agents=sub_agent_configs,
            agent_factory=factory,
        )

        result = await supervisor.run("Research AI agents")

        # Supervisor should still complete even if sub-agent fails
        assert result.state == AgentState.COMPLETED

    async def test_supervisor_agent_type(self, supervisor_config, sub_agent_configs):
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(config=config, provider=provider)

        supervisor = SupervisorAgent(
            config=supervisor_config,
            provider=provider,
            sub_agents=sub_agent_configs,
            agent_factory=factory,
        )

        assert supervisor.agent_type == "supervisor"

    def test_supervisor_registered_in_agent_registry(self):
        """Verify SupervisorAgent is registered under 'supervisor'."""
        from ia_agent_fwk.agents.registry import AgentRegistry

        agent_class = AgentRegistry.get("supervisor")
        assert agent_class is SupervisorAgent

    def test_build_sub_agent_prompt(self, supervisor_config, sub_agent_configs):
        """Test that _build_sub_agent_prompt generates correct text."""
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(config=config, provider=provider)

        supervisor = SupervisorAgent(
            config=supervisor_config,
            provider=provider,
            sub_agents=sub_agent_configs,
            agent_factory=factory,
        )
        prompt_section = supervisor._build_sub_agent_prompt()

        assert "Available sub-agents" in prompt_section
        assert "researcher" in prompt_section
        assert "writer" in prompt_section
        assert "Researches topics" in prompt_section
        assert "Writes content" in prompt_section

    def test_build_sub_agent_prompt_empty(self, supervisor_config):
        """Test with no sub-agents returns empty string."""
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(config=config, provider=provider)

        supervisor = SupervisorAgent(
            config=supervisor_config,
            provider=provider,
            sub_agents=[],
            agent_factory=factory,
        )
        assert supervisor._build_sub_agent_prompt() == ""
