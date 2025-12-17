"""Workflow builder from declarative definitions.

Converts a ``WorkflowDefinition`` into an executable orchestrator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.orchestration.exceptions import WorkflowDefinitionError
from ia_agent_fwk.orchestration.models import (
    AgentFactoryFn,
    FailurePolicy,
    RetryPolicy,
    WorkflowStep,
    WorkflowType,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ia_agent_fwk.orchestration.base import OrchestratorBase
    from ia_agent_fwk.orchestration.models import WorkflowDefinition

logger = logging.getLogger(__name__)


def build_workflow(
    definition: WorkflowDefinition,
    agent_factory: AgentFactoryFn,
    router: Callable[[str], str] | None = None,
) -> OrchestratorBase:
    """Build an executable workflow from a declarative definition.

    Parameters
    ----------
    definition:
        The workflow definition to build.
    agent_factory:
        Callable that creates an ``Agent`` from an ``AgentConfig``.
    router:
        Optional routing function required for conditional workflows.

    Returns
    -------
    OrchestratorBase
        An executable workflow orchestrator.

    Raises
    ------
    WorkflowDefinitionError
        If the definition is invalid.

    """
    collector = get_metrics_collector()

    if not definition.steps:
        collector.increment("workflow_build_errors_total")
        msg = f"Workflow '{definition.name}' has no steps defined."
        raise WorkflowDefinitionError(msg)

    # Parse steps
    steps = _parse_steps(definition)

    # Extract timeout from config
    timeout: float | None = definition.config.get("timeout")

    if definition.workflow_type == WorkflowType.sequential:
        from ia_agent_fwk.orchestration.sequential import SequentialWorkflow  # noqa: PLC0415

        orchestrator = SequentialWorkflow(
            steps=steps,
            agent_factory=agent_factory,
            timeout=timeout,
        )
        collector.increment("workflow_builds_total", labels={"type": "sequential"})
        logger.info(
            "Built sequential workflow '%s' with %d steps",
            definition.name,
            len(steps),
            extra={
                "orchestration_data": {
                    "event": "workflow_built",
                    "workflow_type": "sequential",
                    "workflow_name": definition.name,
                    "step_count": len(steps),
                }
            },
        )
        return orchestrator

    if definition.workflow_type == WorkflowType.parallel:
        from ia_agent_fwk.orchestration.parallel import ParallelWorkflow  # noqa: PLC0415

        failure_policy_str = definition.config.get("failure_policy", "fail_fast")
        failure_policy = FailurePolicy(failure_policy_str)
        max_concurrency = definition.config.get("max_concurrency")

        orchestrator = ParallelWorkflow(
            steps=steps,
            agent_factory=agent_factory,
            failure_policy=failure_policy,
            max_concurrency=max_concurrency,
            timeout=timeout,
        )
        collector.increment("workflow_builds_total", labels={"type": "parallel"})
        logger.info(
            "Built parallel workflow '%s' with %d steps",
            definition.name,
            len(steps),
            extra={
                "orchestration_data": {
                    "event": "workflow_built",
                    "workflow_type": "parallel",
                    "workflow_name": definition.name,
                    "step_count": len(steps),
                }
            },
        )
        return orchestrator

    if definition.workflow_type == WorkflowType.conditional:
        from ia_agent_fwk.orchestration.conditional import ConditionalWorkflow  # noqa: PLC0415

        if router is None:
            collector.increment("workflow_build_errors_total")
            msg = f"Workflow '{definition.name}' is conditional but no router function was provided."
            raise WorkflowDefinitionError(msg)

        routes: dict[str, WorkflowStep] = {}
        default_step: WorkflowStep | None = None
        for step in steps:
            route_key = step.agent_config.get("route_key", step.name) if step.agent_config else step.name
            if route_key == "__default__":
                default_step = step
            else:
                routes[route_key] = step

        orchestrator = ConditionalWorkflow(
            router=router,
            routes=routes,
            agent_factory=agent_factory,
            default=default_step,
            timeout=timeout,
        )
        collector.increment("workflow_builds_total", labels={"type": "conditional"})
        logger.info(
            "Built conditional workflow '%s' with %d routes",
            definition.name,
            len(routes),
            extra={
                "orchestration_data": {
                    "event": "workflow_built",
                    "workflow_type": "conditional",
                    "workflow_name": definition.name,
                    "route_count": len(routes),
                    "has_default": default_step is not None,
                }
            },
        )
        return orchestrator

    collector.increment("workflow_build_errors_total")
    msg = f"Unsupported workflow type: {definition.workflow_type}"
    raise WorkflowDefinitionError(msg)


def _parse_steps(definition: WorkflowDefinition) -> list[WorkflowStep]:
    """Parse step dicts into ``WorkflowStep`` models."""
    steps: list[WorkflowStep] = []
    for i, raw_step in enumerate(definition.steps):
        step_dict = {**raw_step}
        if "name" not in step_dict:
            step_dict["name"] = f"step_{i + 1}"
        if "agent_name" not in step_dict:
            msg = f"Step {i + 1} in workflow '{definition.name}' is missing 'agent_name'."
            raise WorkflowDefinitionError(msg)

        # Parse retry policy if present
        retry_data = step_dict.get("retry_policy")
        retry_policy: RetryPolicy | None = None
        if retry_data is not None:
            retry_policy = RetryPolicy(**retry_data) if isinstance(retry_data, dict) else retry_data

        steps.append(
            WorkflowStep(
                name=step_dict["name"],
                agent_name=step_dict["agent_name"],
                agent_config=step_dict.get("agent_config"),
                retry_policy=retry_policy,
                fallback_agent_name=step_dict.get("fallback_agent_name"),
                input_transform=step_dict.get("input_transform"),
            )
        )
    return steps
