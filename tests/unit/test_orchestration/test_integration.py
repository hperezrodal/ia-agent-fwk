"""End-to-end integration tests for multi-agent orchestration."""

from __future__ import annotations

import json

import pytest

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.models import FinishReason, ToolCall
from ia_agent_fwk.orchestration.conditional import ConditionalWorkflow
from ia_agent_fwk.orchestration.models import FailurePolicy, WorkflowStep
from ia_agent_fwk.orchestration.parallel import ParallelWorkflow
from ia_agent_fwk.orchestration.sequential import SequentialWorkflow
from ia_agent_fwk.orchestration.supervisor import SupervisorAgent

from .conftest import MockAgent, OrcMockLLMProvider, make_chat_response


@pytest.mark.unit
class TestIntegration:
    async def test_sequential_three_step_pipeline(self):
        """End-to-end: researcher -> analyzer -> summarizer pipeline."""
        provider = OrcMockLLMProvider()
        step_outputs = {
            "researcher": "Raw research data about quantum computing.",
            "analyzer": "Analysis shows quantum computing is in early stages.",
            "summarizer": "Summary: Quantum computing is an emerging technology.",
        }

        def factory(config: AgentConfig) -> MockAgent:
            output = step_outputs.get(config.name, "unknown agent output")
            return MockAgent(config=config, provider=provider, output=output)

        steps = [
            WorkflowStep(name="research", agent_name="researcher"),
            WorkflowStep(name="analyze", agent_name="analyzer"),
            WorkflowStep(name="summarize", agent_name="summarizer"),
        ]
        workflow = SequentialWorkflow(steps=steps, agent_factory=factory)
        result = await workflow.execute("Tell me about quantum computing")

        assert result.error is None
        assert len(result.step_results) == 3
        assert "Summary" in result.output
        assert result.usage.total_tokens > 0
        assert result.duration_ms > 0

    async def test_parallel_fan_out_fan_in(self):
        """End-to-end: parallel research on three topics, fan-in results."""
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=provider,
                output=f"Results for {config.name}",
            )

        steps = [
            WorkflowStep(name="topic_1", agent_name="researcher_1"),
            WorkflowStep(name="topic_2", agent_name="researcher_2"),
            WorkflowStep(name="topic_3", agent_name="researcher_3"),
        ]
        workflow = ParallelWorkflow(
            steps=steps,
            agent_factory=factory,
            failure_policy=FailurePolicy.collect_errors,
        )
        result = await workflow.execute("Research these topics")

        assert result.error is None
        assert len(result.step_results) == 3
        # All results should be in the output
        assert "researcher_1" in result.output
        assert "researcher_2" in result.output
        assert "researcher_3" in result.output
        # Aggregated usage: 3 * (10 + 20) = 90 total
        assert result.usage.total_tokens == 90

    async def test_supervisor_delegates_to_two_specialists(self):
        """End-to-end: supervisor delegates to researcher, then to writer."""
        # First LLM call: delegate to researcher
        tool_call_1 = ToolCall(
            id="tc-1",
            name="agent_researcher",
            arguments=json.dumps({"task": "Research AI agents"}),
        )
        # Second LLM call: delegate to writer
        tool_call_2 = ToolCall(
            id="tc-2",
            name="agent_writer",
            arguments=json.dumps({"task": "Write about AI agents"}),
        )
        # Third LLM call: final answer
        provider = OrcMockLLMProvider(
            responses=[
                make_chat_response(
                    content="Let me research first.",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tool_call_1],
                ),
                make_chat_response(
                    content="Now let me write the article.",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tool_call_2],
                ),
                make_chat_response(
                    content="Here is the complete article about AI agents.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )

        def factory(config: AgentConfig) -> MockAgent:
            outputs = {
                "researcher": "Research: AI agents are autonomous programs.",
                "writer": "Article: AI agents revolutionize software.",
            }
            return MockAgent(
                config=config,
                provider=provider,
                output=outputs.get(config.name, "default output"),
            )

        supervisor_config = AgentConfig(
            name="supervisor",
            agent_type="supervisor",
            system_prompt="You coordinate research and writing tasks.",
            max_iterations=5,
            execution_timeout=30,
        )

        sub_agents = [
            (AgentConfig(name="researcher", agent_type="mock"), "Researches topics"),
            (AgentConfig(name="writer", agent_type="mock"), "Writes content"),
        ]

        supervisor = SupervisorAgent(
            config=supervisor_config,
            provider=provider,
            sub_agents=sub_agents,
            agent_factory=factory,
            max_delegation_depth=3,
        )

        result = await supervisor.run("Create an article about AI agents")

        assert result.state == AgentState.COMPLETED
        assert result.output != ""
        assert result.usage.total_tokens > 0

    async def test_conditional_triage_system(self):
        """End-to-end: triage system routes to different agents."""
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            responses = {
                "billing_agent": "Your bill is $50. Payment is due on the 15th.",
                "tech_agent": "Have you tried restarting the device?",
                "general_agent": "How can I help you today?",
            }
            return MockAgent(
                config=config,
                provider=provider,
                output=responses.get(config.name, "Unknown department"),
            )

        routes = {
            "billing": WorkflowStep(name="billing", agent_name="billing_agent"),
            "technical": WorkflowStep(name="technical", agent_name="tech_agent"),
        }
        default = WorkflowStep(name="general", agent_name="general_agent")

        def router(text: str) -> str:
            lower = text.lower()
            if "bill" in lower or "payment" in lower:
                return "billing"
            if "broken" in lower or "error" in lower:
                return "technical"
            return "unknown"

        workflow = ConditionalWorkflow(
            router=router,
            routes=routes,
            agent_factory=factory,
            default=default,
        )

        # Billing route
        result1 = await workflow.execute("What's my bill?")
        assert "bill" in result1.output.lower() or "$50" in result1.output

        # Need a new workflow for the second execution
        workflow2 = ConditionalWorkflow(
            router=router,
            routes=routes,
            agent_factory=factory,
            default=default,
        )

        # Technical route
        result2 = await workflow2.execute("My device is broken")
        assert result2.metadata is not None
        assert result2.metadata["route_key"] == "technical"

    async def test_sequential_with_partial_failure(self):
        """E2E: sequential workflow that fails on step 2 of 3, returns partial results."""
        provider = OrcMockLLMProvider()

        def factory(config: AgentConfig) -> MockAgent:
            should_fail = config.name == "failing_agent"
            return MockAgent(
                config=config,
                provider=provider,
                output=f"output from {config.name}",
                should_fail=should_fail,
            )

        steps = [
            WorkflowStep(name="step_1", agent_name="agent_1"),
            WorkflowStep(name="step_2", agent_name="failing_agent"),
            WorkflowStep(name="step_3", agent_name="agent_3"),
        ]
        workflow = SequentialWorkflow(steps=steps, agent_factory=factory)
        result = await workflow.execute("input")

        assert result.error is not None
        assert "step_2" in result.error
        assert len(result.step_results) == 2  # step_1 succeeded, step_2 failed, step_3 never ran
        assert result.step_results[0].error is None
        assert result.step_results[1].error is not None
