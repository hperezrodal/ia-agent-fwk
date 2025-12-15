"""Parallel workflow orchestrator.

Executes workflow steps concurrently via ``asyncio.gather``, with
configurable failure policy and optional concurrency limits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.models import TokenUsage
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer
from ia_agent_fwk.orchestration.base import OrchestratorBase
from ia_agent_fwk.orchestration.exceptions import WorkflowTimeoutError
from ia_agent_fwk.orchestration.models import (
    AgentFactoryFn,
    FailurePolicy,
    StepResult,
    WorkflowResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ia_agent_fwk.orchestration.models import WorkflowStep

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class ParallelWorkflow(OrchestratorBase):
    """Execute workflow steps concurrently.

    Parameters
    ----------
    steps:
        List of workflow steps to execute in parallel.
    agent_factory:
        Callable that creates an ``Agent`` from an ``AgentConfig``.
    failure_policy:
        How to handle step failures. ``fail_fast`` cancels remaining
        tasks on the first failure. ``collect_errors`` waits for all.
    max_concurrency:
        Maximum number of concurrent agent executions. ``None`` means
        unlimited.
    aggregator:
        Optional callable to combine step results into a single output.
        If ``None``, results are concatenated with step labels.
    timeout:
        Optional workflow-level timeout in seconds.

    """

    def __init__(  # noqa: PLR0913
        self,
        steps: list[WorkflowStep],
        agent_factory: AgentFactoryFn,
        failure_policy: FailurePolicy = FailurePolicy.fail_fast,
        max_concurrency: int | None = None,
        aggregator: Callable[[list[StepResult]], str] | None = None,
        timeout: float | None = None,
        input_transforms: dict[str, Callable[[str], str]] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, input_transforms=input_transforms)
        self._steps = steps
        self._agent_factory = agent_factory
        self._failure_policy = failure_policy
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        self._aggregator = aggregator

    @property
    def orchestrator_type(self) -> str:
        """Return the orchestrator type identifier."""
        return "parallel"

    async def execute(self, input_text: str) -> WorkflowResult:
        """Execute all steps in parallel.

        Parameters
        ----------
        input_text:
            The input text for all steps (fan-out).

        Returns
        -------
        WorkflowResult
            The aggregated result.

        """
        if self._timeout is not None:
            try:
                return await asyncio.wait_for(
                    self._execute_parallel(input_text),
                    timeout=self._timeout,
                )
            except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                collector = get_metrics_collector()
                collector.increment("workflow_executions_total", labels={"type": "parallel", "status": "timeout"})
                logger.error(  # noqa: TRY400
                    "Parallel workflow timed out after %.1fs",
                    self._timeout,
                    extra={
                        "orchestration_data": {
                            "event": "workflow_timeout",
                            "workflow_type": "parallel",
                            "timeout_seconds": self._timeout,
                        }
                    },
                )
                raise WorkflowTimeoutError(self._timeout)  # noqa: B904
        return await self._execute_parallel(input_text)

    async def _execute_parallel(self, input_text: str) -> WorkflowResult:
        """Run all steps in parallel and collect results."""
        collector = get_metrics_collector()
        workflow_start = time.monotonic()

        if not self._steps:
            return WorkflowResult(
                output="",
                step_results=[],
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
                duration_ms=0.0,
            )

        collector.observe("workflow_parallel_fan_out_size", len(self._steps))
        tasks = [self._execute_single_step(step, input_text) for step in self._steps]

        if self._failure_policy == FailurePolicy.collect_errors:
            raw_results: list[StepResult | BaseException] = await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )
            step_results: list[StepResult] = []
            for i, raw in enumerate(raw_results):
                if isinstance(raw, BaseException):
                    collector.increment(
                        "workflow_step_failures_total", labels={"type": "parallel", "step_name": self._steps[i].name}
                    )
                    step_results.append(
                        StepResult(
                            step_name=self._steps[i].name,
                            agent_name=self._steps[i].agent_name,
                            output="",
                            usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
                            duration_ms=0.0,
                            state=AgentState.FAILED.value,
                            error=str(raw),
                        )
                    )
                else:
                    step_results.append(raw)
        else:
            # fail_fast: any exception propagates and cancels others
            try:
                step_results = list(await asyncio.gather(*tasks))
            except Exception as exc:  # noqa: BLE001
                duration_ms = (time.monotonic() - workflow_start) * 1000
                collector.increment("workflow_executions_total", labels={"type": "parallel", "status": "failure"})
                collector.observe("workflow_duration_seconds", duration_ms / 1000, labels={"type": "parallel"})
                logger.error(  # noqa: TRY400
                    "Parallel workflow fail_fast: %s (%.1fms)",
                    exc,
                    duration_ms,
                    extra={
                        "orchestration_data": {
                            "event": "workflow_fail_fast",
                            "workflow_type": "parallel",
                            "failure_policy": "fail_fast",
                            "error": str(exc),
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
                return WorkflowResult(
                    output="",
                    step_results=[],
                    usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
                    duration_ms=duration_ms,
                    error=str(exc),
                )

        duration_ms = (time.monotonic() - workflow_start) * 1000
        aggregated_usage = self._aggregate_usage(step_results)

        # Check for errors in results
        errors = [sr for sr in step_results if sr.error is not None]
        error_msg: str | None = None
        if errors:
            error_msg = "; ".join(f"Step '{sr.step_name}': {sr.error}" for sr in errors)

        status = "failure" if errors else "success"
        collector.increment("workflow_executions_total", labels={"type": "parallel", "status": status})
        collector.observe("workflow_duration_seconds", duration_ms / 1000, labels={"type": "parallel"})
        collector.observe("workflow_steps_completed", len(step_results), labels={"type": "parallel"})
        collector.observe("workflow_tokens_total", aggregated_usage.total_tokens, labels={"type": "parallel"})
        if errors:
            collector.increment(
                "workflow_parallel_partial_failures_total", labels={"failure_policy": self._failure_policy.value}
            )

        logger.info(
            "Parallel workflow completed: steps=%d, errors=%d, tokens=%d (%.1fms)",
            len(step_results),
            len(errors),
            aggregated_usage.total_tokens,
            duration_ms,
            extra={
                "orchestration_data": {
                    "event": "workflow_completed",
                    "workflow_type": "parallel",
                    "fan_out_size": len(self._steps),
                    "steps_completed": len(step_results),
                    "steps_failed": len(errors),
                    "failure_policy": self._failure_policy.value,
                    "prompt_tokens": aggregated_usage.prompt_tokens,
                    "completion_tokens": aggregated_usage.completion_tokens,
                    "total_tokens": aggregated_usage.total_tokens,
                    "duration_ms": round(duration_ms, 1),
                }
            },
        )

        # Aggregate output
        if self._aggregator is not None:
            output = self._aggregator(step_results)
        else:
            output = self._default_aggregator(step_results)

        return WorkflowResult(
            output=output,
            step_results=step_results,
            usage=aggregated_usage,
            duration_ms=duration_ms,
            error=error_msg,
        )

    async def _execute_single_step(
        self,
        step: WorkflowStep,
        input_text: str,
    ) -> StepResult:
        """Execute a single step, respecting the concurrency semaphore."""
        if self._semaphore is not None:
            async with self._semaphore:
                return await self._run_step(step, input_text)
        return await self._run_step(step, input_text)

    async def _run_step(
        self,
        step: WorkflowStep,
        input_text: str,
    ) -> StepResult:
        """Execute the agent for a single step."""
        collector = get_metrics_collector()
        agent_config = self._build_agent_config(step)
        step_start = time.monotonic()

        agent = self._agent_factory(agent_config)
        transformed_input = self._apply_input_transform(step, input_text)
        result = await agent.run(transformed_input)
        duration_ms = (time.monotonic() - step_start) * 1000

        collector.observe(
            "workflow_step_duration_seconds", duration_ms / 1000, labels={"type": "parallel", "step_name": step.name}
        )
        if result.state == AgentState.FAILED:
            collector.increment("workflow_step_failures_total", labels={"type": "parallel", "step_name": step.name})

        logger.info(
            "Parallel workflow step '%s' completed in %.1fms (state=%s)",
            step.name,
            duration_ms,
            result.state.value,
            extra={
                "orchestration_data": {
                    "event": "step_completed",
                    "workflow_type": "parallel",
                    "step_name": step.name,
                    "agent_name": step.agent_name,
                    "state": result.state.value,
                    "duration_ms": round(duration_ms, 1),
                }
            },
        )

        return self._create_step_result(step, result, duration_ms)

    @staticmethod
    def _default_aggregator(step_results: list[StepResult]) -> str:
        """Concatenate step outputs with labels."""
        parts = [f"[{sr.step_name}]: {sr.output}" for sr in step_results if sr.output]
        return "\n\n".join(parts)
