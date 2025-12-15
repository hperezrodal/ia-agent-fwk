"""Sequential workflow orchestrator.

Executes workflow steps in order, chaining each step's output as input
to the next step.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer
from ia_agent_fwk.orchestration.base import OrchestratorBase
from ia_agent_fwk.orchestration.exceptions import WorkflowTimeoutError
from ia_agent_fwk.orchestration.models import AgentFactoryFn, StepResult, WorkflowResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from ia_agent_fwk.orchestration.models import WorkflowStep

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class SequentialWorkflow(OrchestratorBase):
    """Execute workflow steps sequentially, chaining outputs to inputs.

    Parameters
    ----------
    steps:
        Ordered list of workflow steps to execute.
    agent_factory:
        Callable that creates an ``Agent`` from an ``AgentConfig``.
    timeout:
        Optional workflow-level timeout in seconds.

    """

    def __init__(
        self,
        steps: list[WorkflowStep],
        agent_factory: AgentFactoryFn,
        timeout: float | None = None,
        input_transforms: dict[str, Callable[[str], str]] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, input_transforms=input_transforms)
        self._steps = steps
        self._agent_factory = agent_factory

    @property
    def orchestrator_type(self) -> str:
        """Return the orchestrator type identifier."""
        return "sequential"

    async def execute(self, input_text: str) -> WorkflowResult:
        """Execute all steps in sequence.

        Parameters
        ----------
        input_text:
            The initial input text for the first step.

        Returns
        -------
        WorkflowResult
            The aggregated result.

        """
        if self._timeout is not None:
            try:
                return await asyncio.wait_for(
                    self._execute_steps(input_text),
                    timeout=self._timeout,
                )
            except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                collector = get_metrics_collector()
                collector.increment("workflow_executions_total", labels={"type": "sequential", "status": "timeout"})
                logger.error(  # noqa: TRY400
                    "Sequential workflow timed out after %.1fs",
                    self._timeout,
                    extra={
                        "orchestration_data": {
                            "event": "workflow_timeout",
                            "workflow_type": "sequential",
                            "timeout_seconds": self._timeout,
                        }
                    },
                )
                raise WorkflowTimeoutError(self._timeout)  # noqa: B904
        return await self._execute_steps(input_text)

    async def _execute_steps(self, input_text: str) -> WorkflowResult:
        """Run the internal step execution loop."""
        collector = get_metrics_collector()
        workflow_start = time.monotonic()
        step_results: list[StepResult] = []
        current_input = input_text

        for step in self._steps:
            step_result = await self._execute_single_step(step, current_input)
            step_results.append(step_result)

            if step_result.error is not None:
                # Step failed -- stop execution and return partial results
                collector.increment(
                    "workflow_step_failures_total", labels={"type": "sequential", "step_name": step.name}
                )
                logger.warning(
                    "Sequential workflow step '%s' failed: %s",
                    step.name,
                    step_result.error,
                    extra={
                        "orchestration_data": {
                            "event": "step_failed",
                            "workflow_type": "sequential",
                            "step_name": step.name,
                            "error": step_result.error,
                            "completed_steps": len(step_results),
                            "total_steps": len(self._steps),
                        }
                    },
                )
                duration_ms = (time.monotonic() - workflow_start) * 1000
                aggregated_usage = self._aggregate_usage(step_results)
                collector.increment("workflow_executions_total", labels={"type": "sequential", "status": "failure"})
                collector.observe("workflow_duration_seconds", duration_ms / 1000, labels={"type": "sequential"})
                collector.observe("workflow_steps_completed", len(step_results), labels={"type": "sequential"})
                return WorkflowResult(
                    output="",
                    step_results=step_results,
                    usage=aggregated_usage,
                    duration_ms=duration_ms,
                    error=f"Step '{step.name}' failed: {step_result.error}",
                )

            current_input = step_result.output
            collector.observe(
                "workflow_step_duration_seconds",
                step_result.duration_ms / 1000,
                labels={"type": "sequential", "step_name": step.name},
            )
            logger.info(
                "Sequential workflow step '%s' completed in %.1fms",
                step.name,
                step_result.duration_ms,
                extra={
                    "orchestration_data": {
                        "event": "step_completed",
                        "workflow_type": "sequential",
                        "step_name": step.name,
                        "agent_name": step.agent_name,
                        "duration_ms": round(step_result.duration_ms, 1),
                        "step_index": len(step_results),
                        "total_steps": len(self._steps),
                    }
                },
            )

        duration_ms = (time.monotonic() - workflow_start) * 1000
        aggregated_usage = self._aggregate_usage(step_results)
        final_output = step_results[-1].output if step_results else ""

        collector.increment("workflow_executions_total", labels={"type": "sequential", "status": "success"})
        collector.observe("workflow_duration_seconds", duration_ms / 1000, labels={"type": "sequential"})
        collector.observe("workflow_steps_completed", len(step_results), labels={"type": "sequential"})
        collector.observe("workflow_tokens_total", aggregated_usage.total_tokens, labels={"type": "sequential"})
        logger.info(
            "Sequential workflow completed: steps=%d, tokens=%d (%.1fms)",
            len(step_results),
            aggregated_usage.total_tokens,
            duration_ms,
            extra={
                "orchestration_data": {
                    "event": "workflow_completed",
                    "workflow_type": "sequential",
                    "steps_completed": len(step_results),
                    "prompt_tokens": aggregated_usage.prompt_tokens,
                    "completion_tokens": aggregated_usage.completion_tokens,
                    "total_tokens": aggregated_usage.total_tokens,
                    "duration_ms": round(duration_ms, 1),
                }
            },
        )

        return WorkflowResult(
            output=final_output,
            step_results=step_results,
            usage=aggregated_usage,
            duration_ms=duration_ms,
        )

    async def _execute_single_step(
        self,
        step: WorkflowStep,
        input_text: str,
    ) -> StepResult:
        """Execute a single step with optional retry and fallback."""
        from ia_agent_fwk.agents.config import AgentConfig  # noqa: PLC0415

        collector = get_metrics_collector()

        # Build agent config from step
        agent_config = self._build_agent_config(step)

        # Retry logic
        max_retries = step.retry_policy.max_retries if step.retry_policy else 0
        last_error: str | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                backoff = min(
                    step.retry_policy.backoff_base**attempt,  # type: ignore[union-attr]
                    step.retry_policy.backoff_max,  # type: ignore[union-attr]
                )
                collector.increment(
                    "workflow_step_retries_total", labels={"type": "sequential", "step_name": step.name}
                )
                logger.warning(
                    "Retrying step '%s' (attempt %d/%d) after %.1fs backoff",
                    step.name,
                    attempt + 1,
                    max_retries + 1,
                    backoff,
                    extra={
                        "orchestration_data": {
                            "event": "step_retry",
                            "workflow_type": "sequential",
                            "step_name": step.name,
                            "attempt": attempt + 1,
                            "max_attempts": max_retries + 1,
                            "backoff_seconds": round(backoff, 1),
                        }
                    },
                )
                await asyncio.sleep(backoff)

            step_start = time.monotonic()
            agent = self._agent_factory(agent_config)
            transformed_input = self._apply_input_transform(step, input_text)
            result = await agent.run(transformed_input)
            duration_ms = (time.monotonic() - step_start) * 1000

            if result.state != AgentState.FAILED:
                return self._create_step_result(step, result, duration_ms)

            last_error = result.error

        # All retries exhausted -- try fallback
        if step.fallback_agent_name is not None:
            collector.increment("workflow_step_fallbacks_total", labels={"type": "sequential", "step_name": step.name})
            logger.warning(
                "Step '%s' failed after %d attempts. Trying fallback agent '%s'.",
                step.name,
                max_retries + 1,
                step.fallback_agent_name,
                extra={
                    "orchestration_data": {
                        "event": "step_fallback",
                        "workflow_type": "sequential",
                        "step_name": step.name,
                        "fallback_agent": step.fallback_agent_name,
                        "attempts_exhausted": max_retries + 1,
                    }
                },
            )
            fallback_config = AgentConfig(
                name=step.fallback_agent_name,
                agent_type=step.fallback_agent_name,
            )
            if step.agent_config and "system_prompt" in step.agent_config:
                fallback_config = AgentConfig(
                    name=step.fallback_agent_name,
                    agent_type=step.fallback_agent_name,
                    system_prompt=step.agent_config["system_prompt"],
                )

            step_start = time.monotonic()
            fallback_agent = self._agent_factory(fallback_config)
            fallback_result = await fallback_agent.run(input_text)
            duration_ms = (time.monotonic() - step_start) * 1000

            return StepResult(
                step_name=step.name,
                agent_name=step.fallback_agent_name,
                output=fallback_result.output,
                usage=fallback_result.usage,
                duration_ms=duration_ms,
                state=fallback_result.state.value,
                error=fallback_result.error,
            )

        # No fallback -- return the error from last attempt
        from ia_agent_fwk.llm.models import TokenUsage  # noqa: PLC0415

        return StepResult(
            step_name=step.name,
            agent_name=step.agent_name,
            output="",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
            duration_ms=0.0,
            state=AgentState.FAILED.value,
            error=last_error,
        )
