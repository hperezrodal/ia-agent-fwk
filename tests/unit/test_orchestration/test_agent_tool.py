"""Tests for the AgentTool adapter."""

from __future__ import annotations

import pytest

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.orchestration.agent_tool import AgentTool, AgentToolInput, AgentToolOutput
from ia_agent_fwk.tools.base import ToolContext

from .conftest import MockAgent, OrcMockLLMProvider


@pytest.mark.unit
class TestAgentTool:
    @pytest.fixture
    def agent_config(self):
        return AgentConfig(
            name="researcher",
            agent_type="mock",
            system_prompt="You are a researcher.",
        )

    @pytest.fixture
    def tool_factory(self):
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=provider,
                output=f"researched: {config.name}",
            )

        return factory

    @pytest.fixture
    def tool_context(self):
        return ToolContext(execution_id="test-exec-1", agent_id="supervisor")

    def test_agent_tool_wraps_agent(self, agent_config, tool_factory):
        tool = AgentTool(
            agent_config=agent_config,
            agent_factory=tool_factory,
        )
        assert tool.name == "agent_researcher"
        assert "researcher" in tool.description

    def test_agent_tool_custom_name(self, agent_config, tool_factory):
        tool = AgentTool(
            agent_config=agent_config,
            agent_factory=tool_factory,
            tool_name="custom_tool",
            tool_description="A custom tool description",
        )
        assert tool.name == "custom_tool"
        assert tool.description == "A custom tool description"

    def test_agent_tool_schema(self, agent_config, tool_factory):
        tool = AgentTool(
            agent_config=agent_config,
            agent_factory=tool_factory,
        )
        assert tool.input_schema is AgentToolInput
        assert tool.output_schema is AgentToolOutput

        # Verify input schema has expected fields
        schema = tool.input_schema.model_json_schema()
        assert "task" in schema["properties"]
        assert "context" in schema["properties"]

    async def test_agent_tool_execute(self, agent_config, tool_factory, tool_context):
        tool = AgentTool(
            agent_config=agent_config,
            agent_factory=tool_factory,
        )

        input_data = AgentToolInput(task="Find information about AI")
        result = await tool.execute(input_data, tool_context)

        assert isinstance(result, AgentToolOutput)
        assert result.output == "researched: researcher"
        assert result.state == AgentState.COMPLETED.value
        assert result.usage["prompt_tokens"] == 10

    async def test_agent_tool_handles_failure(self, agent_config, tool_context):
        provider = OrcMockLLMProvider()

        def failing_factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=provider,
                should_fail=True,
            )

        tool = AgentTool(
            agent_config=agent_config,
            agent_factory=failing_factory,
        )

        input_data = AgentToolInput(task="This will fail")
        result = await tool.execute(input_data, tool_context)

        assert isinstance(result, AgentToolOutput)
        assert result.state == AgentState.FAILED.value

    async def test_agent_tool_depth_exceeded(self, agent_config, tool_factory, tool_context):
        tool = AgentTool(
            agent_config=agent_config,
            agent_factory=tool_factory,
            delegation_depth=0,
        )

        input_data = AgentToolInput(task="Should not execute")
        result = await tool.execute(input_data, tool_context)

        assert isinstance(result, AgentToolOutput)
        assert result.state == AgentState.FAILED.value
        assert "Delegation depth exceeded" in result.output

    def test_agent_tool_delegation_depth_property(self, agent_config, tool_factory):
        tool = AgentTool(
            agent_config=agent_config,
            agent_factory=tool_factory,
            delegation_depth=5,
        )
        assert tool.delegation_depth == 5

    async def test_agent_tool_with_context(self, agent_config, tool_context):
        provider = OrcMockLLMProvider()
        received_inputs = []

        def factory(config: AgentConfig) -> MockAgent:
            agent = MockAgent(config=config, provider=provider, output="result")
            original_run = agent.run

            async def tracked_run(input_text, conversation_history=None):
                received_inputs.append(input_text)
                return await original_run(input_text, conversation_history)

            agent.run = tracked_run  # type: ignore[method-assign]
            return agent

        tool = AgentTool(agent_config=agent_config, agent_factory=factory)
        input_data = AgentToolInput(task="main task", context="extra context")
        await tool.execute(input_data, tool_context)

        assert len(received_inputs) == 1
        assert "main task" in received_inputs[0]
        assert "extra context" in received_inputs[0]
