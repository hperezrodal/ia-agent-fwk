"""Pydantic v2 models for workflow orchestration.

Defines workflow steps, results, definitions, enums, and retry policies
used by all orchestration patterns.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.llm.models import TokenUsage  # noqa: TC001

if TYPE_CHECKING:
    from ia_agent_fwk.agents.base import Agent
    from ia_agent_fwk.agents.config import AgentConfig

AgentFactoryFn = Callable[["AgentConfig"], "Agent"]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FailurePolicy(str, Enum):
    """Failure handling policy for parallel workflows."""

    fail_fast = "fail_fast"
    collect_errors = "collect_errors"


class WorkflowType(str, Enum):
    """Discriminator for workflow types."""

    sequential = "sequential"
    parallel = "parallel"
    conditional = "conditional"
    supervisor = "supervisor"


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


class RetryPolicy(BaseModel):
    """Retry configuration for a workflow step."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(default=0, ge=0)
    backoff_base: float = Field(default=2.0, gt=0)
    backoff_max: float = Field(default=60.0, gt=0)


# ---------------------------------------------------------------------------
# Workflow step
# ---------------------------------------------------------------------------


class WorkflowStep(BaseModel):
    """Definition of a single step in a workflow.

    Attributes
    ----------
    name:
        Human-readable name for the step.
    agent_name:
        Agent name used to create the agent via factory.
    agent_config:
        Optional override agent configuration (dict form for flexibility).
    retry_policy:
        Optional retry policy for the step.
    fallback_agent_name:
        Optional fallback agent name if the primary agent fails.
    input_transform:
        Optional description for declarative input transformation.

    """

    name: str
    agent_name: str
    agent_config: dict[str, Any] | None = None
    retry_policy: RetryPolicy | None = None
    fallback_agent_name: str | None = None
    input_transform: str | None = None


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """Result of a single workflow step execution."""

    model_config = ConfigDict(frozen=True)

    step_name: str
    agent_name: str
    output: str
    usage: TokenUsage
    duration_ms: float
    state: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Workflow result
# ---------------------------------------------------------------------------


class WorkflowResult(BaseModel):
    """Aggregated result of a workflow execution."""

    model_config = ConfigDict(frozen=True)

    output: str
    step_results: list[StepResult]
    usage: TokenUsage
    duration_ms: float
    error: str | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Workflow definition (declarative)
# ---------------------------------------------------------------------------


class WorkflowDefinition(BaseModel):
    """Declarative workflow specification.

    Can be loaded from YAML/dict via ``model_validate()``.
    """

    name: str
    description: str = ""
    workflow_type: WorkflowType
    steps: list[dict[str, Any]]
    config: dict[str, Any] = Field(default_factory=dict)
