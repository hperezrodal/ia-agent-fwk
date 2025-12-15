"""Tests for sequential workflow orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ia_agent_fwk.orchestration.exceptions import WorkflowTimeoutError
from ia_agent_fwk.orchestration.models import RetryPolicy, WorkflowStep
from ia_agent_fwk.orchestration.sequential import SequentialWorkflow

from .conftest import MockAgent

if TYPE_CHECKING:
    from ia_agent_fwk.agents.config import AgentConfig


@pytest.mark.unit
class TestSequentialWorkflow:
    async def test_sequential_executes_in_order(self, mock_agent_factory, simple_workflow_steps):
        workflow = SequentialWorkflow(
            steps=simple_workflow_steps,
            agent_factory=mock_agent_factory,
        )
        result = await workflow.execute("start input")

        assert result.error is None
        assert len(result.step_results) == 3
        assert result.step_results[0].step_name == "step_1"
        assert result.step_results[1].step_name == "step_2"
        assert result.step_results[2].step_name == "step_3"

    async def test_sequential_passes_output_to_next_step(self, mock_provider):
        """Verify that step N's output becomes step N+1's input."""
        call_inputs = []

        def factory(config: AgentConfig) -> MockAgent:
            agent = MockAgent(
                config=config,
                provider=mock_provider,
                output=f"output_from_{config.name}",
            )
            original_run = agent.run

            async def tracked_run(input_text, conversation_history=None):
                call_inputs.append(input_text)
                return await original_run(input_text, conversation_history)

            agent.run = tracked_run  # type: ignore[method-assign]
            return agent

        steps = [
            WorkflowStep(name="step_1", agent_name="agent_a"),
            WorkflowStep(name="step_2", agent_name="agent_b"),
        ]
        workflow = SequentialWorkflow(steps=steps, agent_factory=factory)
        result = await workflow.execute("initial input")

        assert call_inputs[0] == "initial input"
        assert call_inputs[1] == "output_from_agent_a"
        assert result.output == "output_from_agent_b"

    async def test_sequential_stops_on_failure(self, mock_provider):
        call_count = [0]

        def factory(config: AgentConfig) -> MockAgent:
            call_count[0] += 1
            should_fail = config.name == "step_2_agent"
            return MockAgent(
                config=config,
                provider=mock_provider,
                output=f"output_{call_count[0]}",
                should_fail=should_fail,
            )

        steps = [
            WorkflowStep(name="step_1", agent_name="step_1_agent"),
            WorkflowStep(name="step_2", agent_name="step_2_agent"),
            WorkflowStep(name="step_3", agent_name="step_3_agent"),
        ]
        workflow = SequentialWorkflow(steps=steps, agent_factory=factory)
        result = await workflow.execute("input")

        assert result.error is not None
        assert "step_2" in result.error
        assert len(result.step_results) == 2  # Only step_1 and step_2
        assert result.step_results[0].error is None  # step_1 succeeded
        assert result.step_results[1].error is not None  # step_2 failed

    async def test_sequential_empty_steps(self, mock_agent_factory):
        workflow = SequentialWorkflow(steps=[], agent_factory=mock_agent_factory)
        result = await workflow.execute("input")

        assert result.output == ""
        assert len(result.step_results) == 0
        assert result.error is None

    async def test_sequential_single_step(self, mock_agent_factory):
        steps = [WorkflowStep(name="only_step", agent_name="mock")]
        workflow = SequentialWorkflow(steps=steps, agent_factory=mock_agent_factory)
        result = await workflow.execute("input")

        assert len(result.step_results) == 1
        assert result.error is None
        assert result.output != ""

    async def test_sequential_timeout(self, mock_provider):
        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=mock_provider,
                output="slow output",
                delay=5.0,
            )

        steps = [WorkflowStep(name="slow_step", agent_name="slow")]
        workflow = SequentialWorkflow(steps=steps, agent_factory=factory, timeout=0.1)

        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await workflow.execute("input")
        assert exc_info.value.timeout == 0.1

    async def test_sequential_aggregates_usage(self, mock_agent_factory, simple_workflow_steps):
        workflow = SequentialWorkflow(
            steps=simple_workflow_steps,
            agent_factory=mock_agent_factory,
        )
        result = await workflow.execute("input")

        # Each mock agent returns 10 prompt + 20 completion tokens
        assert result.usage.prompt_tokens == 30  # 3 steps * 10
        assert result.usage.completion_tokens == 60  # 3 steps * 20
        assert result.usage.total_tokens == 90

    async def test_sequential_orchestrator_type(self, mock_agent_factory):
        workflow = SequentialWorkflow(steps=[], agent_factory=mock_agent_factory)
        assert workflow.orchestrator_type == "sequential"

    async def test_sequential_with_retry(self, mock_provider):
        """Test retry logic on step failure."""
        attempt_count = [0]

        def factory(config: AgentConfig) -> MockAgent:
            attempt_count[0] += 1
            # Fail on first two attempts, succeed on third
            should_fail = attempt_count[0] <= 2
            return MockAgent(
                config=config,
                provider=mock_provider,
                output="success after retry",
                should_fail=should_fail,
            )

        steps = [
            WorkflowStep(
                name="retryable_step",
                agent_name="retryable",
                retry_policy=RetryPolicy(max_retries=2, backoff_base=0.01, backoff_max=0.02),
            ),
        ]
        workflow = SequentialWorkflow(steps=steps, agent_factory=factory)
        result = await workflow.execute("input")

        assert result.error is None
        assert result.output == "success after retry"
        assert attempt_count[0] == 3  # 1 initial + 2 retries

    async def test_sequential_input_transform(self, mock_provider):
        """Test that input_transforms are applied before each step."""
        call_inputs: list[str] = []

        def factory(config: AgentConfig) -> MockAgent:
            agent = MockAgent(
                config=config,
                provider=mock_provider,
                output=f"output_from_{config.name}",
            )
            original_run = agent.run

            async def tracked_run(input_text, conversation_history=None):
                call_inputs.append(input_text)
                return await original_run(input_text, conversation_history)

            agent.run = tracked_run  # type: ignore[method-assign]
            return agent

        steps = [
            WorkflowStep(name="step_1", agent_name="agent_a"),
            WorkflowStep(name="step_2", agent_name="agent_b"),
        ]
        workflow = SequentialWorkflow(
            steps=steps,
            agent_factory=factory,
            input_transforms={
                "step_1": lambda text: f"Analyze: {text}",
            },
        )
        result = await workflow.execute("raw input")

        assert call_inputs[0] == "Analyze: raw input"
        # step_2 has no transform, gets step_1's output unchanged
        assert call_inputs[1] == "output_from_agent_a"
        assert result.error is None

    async def test_sequential_with_fallback(self, mock_provider):
        """Test that a failing step falls back to the fallback agent."""

        def factory(config: AgentConfig) -> MockAgent:
            should_fail = config.name == "primary"
            return MockAgent(
                config=config,
                provider=mock_provider,
                output=f"output_from_{config.name}",
                should_fail=should_fail,
            )

        steps = [
            WorkflowStep(
                name="step_with_fallback",
                agent_name="primary",
                fallback_agent_name="backup",
            ),
        ]
        workflow = SequentialWorkflow(steps=steps, agent_factory=factory)
        result = await workflow.execute("input")

        assert result.error is None
        assert len(result.step_results) == 1
        assert result.step_results[0].agent_name == "backup"
        assert result.output == "output_from_backup"
