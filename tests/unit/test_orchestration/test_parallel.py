"""Tests for parallel workflow orchestrator."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from ia_agent_fwk.orchestration.exceptions import WorkflowTimeoutError
from ia_agent_fwk.orchestration.models import FailurePolicy, StepResult, WorkflowStep
from ia_agent_fwk.orchestration.parallel import ParallelWorkflow

from .conftest import MockAgent

if TYPE_CHECKING:
    from ia_agent_fwk.agents.config import AgentConfig


@pytest.mark.unit
class TestParallelWorkflow:
    async def test_parallel_executes_concurrently(self, mock_provider):
        """Steps should run concurrently, not sequentially."""

        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=mock_provider,
                output=f"output_{config.name}",
                delay=0.1,
            )

        steps = [
            WorkflowStep(name="step_1", agent_name="agent_a"),
            WorkflowStep(name="step_2", agent_name="agent_b"),
            WorkflowStep(name="step_3", agent_name="agent_c"),
        ]
        workflow = ParallelWorkflow(steps=steps, agent_factory=factory)

        start = time.monotonic()
        result = await workflow.execute("input")
        elapsed = time.monotonic() - start

        assert result.error is None
        assert len(result.step_results) == 3
        # If sequential, would take ~0.3s; parallel should be ~0.1s
        assert elapsed < 0.25

    async def test_parallel_collects_all_results(self, mock_provider):
        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=mock_provider,
                output=f"result_{config.name}",
            )

        steps = [
            WorkflowStep(name="step_a", agent_name="agent_a"),
            WorkflowStep(name="step_b", agent_name="agent_b"),
        ]
        workflow = ParallelWorkflow(steps=steps, agent_factory=factory)
        result = await workflow.execute("input")

        assert len(result.step_results) == 2
        outputs = {sr.step_name: sr.output for sr in result.step_results}
        assert "result_agent_a" in outputs["step_a"]
        assert "result_agent_b" in outputs["step_b"]

    async def test_parallel_handles_partial_failures(self, mock_provider):
        """With collect_errors, all steps run even if some fail."""
        call_count = [0]

        def factory(config: AgentConfig) -> MockAgent:
            call_count[0] += 1
            should_fail = config.name == "bad_agent"
            return MockAgent(
                config=config,
                provider=mock_provider,
                output="good result",
                should_fail=should_fail,
            )

        steps = [
            WorkflowStep(name="good_step", agent_name="good_agent"),
            WorkflowStep(name="bad_step", agent_name="bad_agent"),
            WorkflowStep(name="another_good", agent_name="good_agent_2"),
        ]
        workflow = ParallelWorkflow(
            steps=steps,
            agent_factory=factory,
            failure_policy=FailurePolicy.collect_errors,
        )
        result = await workflow.execute("input")

        assert len(result.step_results) == 3
        assert result.error is not None  # There is at least one error
        # Good steps succeeded
        assert result.step_results[0].error is None
        assert result.step_results[2].error is None
        # Bad step failed
        assert result.step_results[1].error is not None

    async def test_parallel_empty_steps(self, mock_agent_factory):
        workflow = ParallelWorkflow(steps=[], agent_factory=mock_agent_factory)
        result = await workflow.execute("input")

        assert result.output == ""
        assert len(result.step_results) == 0
        assert result.error is None

    async def test_parallel_timeout(self, mock_provider):
        def factory(config: AgentConfig) -> MockAgent:
            return MockAgent(
                config=config,
                provider=mock_provider,
                output="slow",
                delay=5.0,
            )

        steps = [WorkflowStep(name="slow_step", agent_name="slow")]
        workflow = ParallelWorkflow(steps=steps, agent_factory=factory, timeout=0.1)

        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await workflow.execute("input")
        assert exc_info.value.timeout == 0.1

    async def test_parallel_aggregates_usage(self, mock_agent_factory, simple_workflow_steps):
        workflow = ParallelWorkflow(
            steps=simple_workflow_steps,
            agent_factory=mock_agent_factory,
        )
        result = await workflow.execute("input")

        # Each mock agent returns 10 prompt + 20 completion tokens
        assert result.usage.prompt_tokens == 30
        assert result.usage.completion_tokens == 60

    async def test_parallel_orchestrator_type(self, mock_agent_factory):
        workflow = ParallelWorkflow(steps=[], agent_factory=mock_agent_factory)
        assert workflow.orchestrator_type == "parallel"

    async def test_parallel_custom_aggregator(self, mock_agent_factory, simple_workflow_steps):
        def custom_aggregator(step_results: list[StepResult]) -> str:
            return " | ".join(sr.output for sr in step_results if sr.output)

        workflow = ParallelWorkflow(
            steps=simple_workflow_steps,
            agent_factory=mock_agent_factory,
            aggregator=custom_aggregator,
        )
        result = await workflow.execute("input")

        assert " | " in result.output

    async def test_parallel_max_concurrency(self, mock_provider):
        """Verify semaphore limits concurrency."""
        concurrent_count = [0]
        max_concurrent = [0]

        def factory(config: AgentConfig) -> MockAgent:
            agent = MockAgent(
                config=config,
                provider=mock_provider,
                output="result",
                delay=0.05,
            )
            original_run = agent.run

            async def tracked_run(input_text, conversation_history=None):
                concurrent_count[0] += 1
                max_concurrent[0] = max(max_concurrent[0], concurrent_count[0])
                try:
                    return await original_run(input_text, conversation_history)
                finally:
                    concurrent_count[0] -= 1

            agent.run = tracked_run  # type: ignore[method-assign]
            return agent

        steps = [WorkflowStep(name=f"step_{i}", agent_name=f"agent_{i}") for i in range(5)]
        workflow = ParallelWorkflow(
            steps=steps,
            agent_factory=factory,
            max_concurrency=2,
        )
        await workflow.execute("input")

        assert max_concurrent[0] <= 2

    async def test_parallel_input_transform(self, mock_provider):
        """Test that input_transforms are applied per step."""
        call_inputs: dict[str, str] = {}

        def factory(config: AgentConfig) -> MockAgent:
            agent = MockAgent(
                config=config,
                provider=mock_provider,
                output=f"output_from_{config.name}",
            )
            original_run = agent.run
            name = config.name

            async def tracked_run(input_text, conversation_history=None):
                call_inputs[name] = input_text
                return await original_run(input_text, conversation_history)

            agent.run = tracked_run  # type: ignore[method-assign]
            return agent

        steps = [
            WorkflowStep(name="step_a", agent_name="agent_a"),
            WorkflowStep(name="step_b", agent_name="agent_b"),
        ]
        workflow = ParallelWorkflow(
            steps=steps,
            agent_factory=factory,
            input_transforms={
                "step_a": lambda text: f"PREFIX: {text}",
            },
        )
        result = await workflow.execute("raw input")

        assert call_inputs["agent_a"] == "PREFIX: raw input"
        assert call_inputs["agent_b"] == "raw input"  # no transform
        assert result.error is None
