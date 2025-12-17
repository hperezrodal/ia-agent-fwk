"""Conditional routing workflow orchestrator.

Routes input to different agents based on runtime conditions evaluated
by a callable router function.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer
from ia_agent_fwk.orchestration.base import OrchestratorBase
from ia_agent_fwk.orchestration.exceptions import OrchestrationError, WorkflowTimeoutError
from ia_agent_fwk.orchestration.models import AgentFactoryFn, WorkflowResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from ia_agent_fwk.orchestration.models import WorkflowStep

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class ConditionalWorkflow(OrchestratorBase):
    """Route input to an agent based on a routing function.

    Parameters
    ----------
    router:
        Callable that receives input text and returns a route key.
    routes:
        Mapping of route keys to workflow steps.
    agent_factory:
        Callable that creates an ``Agent`` from an ``AgentConfig``.
    default:
        Optional default step if no route key matches.
    timeout:
        Optional workflow-level timeout in seconds.

    """

    def __init__(  # noqa: PLR0913
        self,
        router: Callable[[str], str],
        routes: dict[str, WorkflowStep],
        agent_factory: AgentFactoryFn,
        default: WorkflowStep | None = None,
        timeout: float | None = None,
        input_transforms: dict[str, Callable[[str], str]] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, input_transforms=input_transforms)
        self._router = router
        self._routes = routes
        self._agent_factory = agent_factory
        self._default = default

    @property
    def orchestrator_type(self) -> str:
        """Return the orchestrator type identifier."""
        return "conditional"

    async def execute(self, input_text: str) -> WorkflowResult:
        """Execute the workflow by routing to the appropriate agent.

        Parameters
        ----------
        input_text:
            The input text to route and process.

        Returns
        -------
        WorkflowResult
            The result from the selected agent.

        """
        if self._timeout is not None:
            try:
                return await asyncio.wait_for(
                    self._execute_routed(input_text),
                    timeout=self._timeout,
                )
            except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                collector = get_metrics_collector()
                collector.increment("workflow_executions_total", labels={"type": "conditional", "status": "timeout"})
                logger.error(  # noqa: TRY400
                    "Conditional workflow timed out after %.1fs",
                    self._timeout,
                    extra={
                        "orchestration_data": {
                            "event": "workflow_timeout",
                            "workflow_type": "conditional",
                            "timeout_seconds": self._timeout,
                        }
                    },
                )
                raise WorkflowTimeoutError(self._timeout)  # noqa: B904
        return await self._execute_routed(input_text)

    async def _execute_routed(self, input_text: str) -> WorkflowResult:
        """Route input and execute the selected agent."""
        collector = get_metrics_collector()
        workflow_start = time.monotonic()

        route_key = self._router(input_text)
        collector.increment("workflow_conditional_routes_total", labels={"route_key": route_key})
        logger.info(
            "Conditional router selected route: '%s'",
            route_key,
            extra={
                "orchestration_data": {
                    "event": "route_selected",
                    "workflow_type": "conditional",
                    "route_key": route_key,
                    "available_routes": list(self._routes.keys()),
                }
            },
        )

        step = self._routes.get(route_key)
        used_default = False
        if step is None:
            step = self._default
            used_default = True
        if step is None:
            collector.increment("workflow_conditional_no_route_total")
            collector.increment("workflow_executions_total", labels={"type": "conditional", "status": "failure"})
            logger.error(
                "No route found for key '%s' and no default configured",
                route_key,
                extra={
                    "orchestration_data": {
                        "event": "no_route_found",
                        "workflow_type": "conditional",
                        "route_key": route_key,
                        "available_routes": list(self._routes.keys()),
                    }
                },
            )
            msg = (
                f"No route found for key '{route_key}' and no default route configured. "
                f"Available routes: {list(self._routes.keys())}"
            )
            raise OrchestrationError(msg)

        if used_default:
            collector.increment("workflow_conditional_default_route_total")

        agent_config = self._build_agent_config(step)
        step_start = time.monotonic()
        agent = self._agent_factory(agent_config)
        transformed_input = self._apply_input_transform(step, input_text)
        result = await agent.run(transformed_input)
        duration_ms = (time.monotonic() - step_start) * 1000

        step_result = self._create_step_result(step, result, duration_ms)
        total_duration_ms = (time.monotonic() - workflow_start) * 1000

        status = "failure" if step_result.error else "success"
        collector.increment("workflow_executions_total", labels={"type": "conditional", "status": status})
        collector.observe("workflow_duration_seconds", total_duration_ms / 1000, labels={"type": "conditional"})
        collector.observe(
            "workflow_step_duration_seconds", duration_ms / 1000, labels={"type": "conditional", "step_name": step.name}
        )
        collector.observe("workflow_tokens_total", step_result.usage.total_tokens, labels={"type": "conditional"})

        logger.info(
            "Conditional workflow completed: route=%s, agent=%s, tokens=%d (%.1fms)",
            route_key,
            step.agent_name,
            step_result.usage.total_tokens,
            total_duration_ms,
            extra={
                "orchestration_data": {
                    "event": "workflow_completed",
                    "workflow_type": "conditional",
                    "route_key": route_key,
                    "used_default": used_default,
                    "agent_name": step.agent_name,
                    "prompt_tokens": step_result.usage.prompt_tokens,
                    "completion_tokens": step_result.usage.completion_tokens,
                    "total_tokens": step_result.usage.total_tokens,
                    "duration_ms": round(total_duration_ms, 1),
                }
            },
        )

        return WorkflowResult(
            output=step_result.output,
            step_results=[step_result],
            usage=step_result.usage,
            duration_ms=total_duration_ms,
            error=step_result.error,
            metadata={"route_key": route_key},
        )
