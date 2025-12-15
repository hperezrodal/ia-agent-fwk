"""Multi-Agent Orchestration module.

Provides supervisor agents, workflow engine, agent delegation,
agent-as-tool adapter, and conditional routing.
"""

from __future__ import annotations

from ia_agent_fwk.orchestration.agent_tool import AgentTool, AgentToolInput, AgentToolOutput
from ia_agent_fwk.orchestration.base import OrchestratorBase
from ia_agent_fwk.orchestration.builder import build_workflow
from ia_agent_fwk.orchestration.conditional import ConditionalWorkflow
from ia_agent_fwk.orchestration.exceptions import (
    DelegationDepthExceededError,
    OrchestrationError,
    WorkflowDefinitionError,
    WorkflowStepError,
    WorkflowTimeoutError,
)
from ia_agent_fwk.orchestration.models import (
    FailurePolicy,
    RetryPolicy,
    StepResult,
    WorkflowDefinition,
    WorkflowResult,
    WorkflowStep,
    WorkflowType,
)
from ia_agent_fwk.orchestration.parallel import ParallelWorkflow
from ia_agent_fwk.orchestration.sequential import SequentialWorkflow
from ia_agent_fwk.orchestration.supervisor import SupervisorAgent

__all__ = [
    "AgentTool",
    "AgentToolInput",
    "AgentToolOutput",
    "ConditionalWorkflow",
    "DelegationDepthExceededError",
    "FailurePolicy",
    "OrchestrationError",
    "OrchestratorBase",
    "ParallelWorkflow",
    "RetryPolicy",
    "SequentialWorkflow",
    "StepResult",
    "SupervisorAgent",
    "WorkflowDefinition",
    "WorkflowDefinitionError",
    "WorkflowResult",
    "WorkflowStep",
    "WorkflowStepError",
    "WorkflowTimeoutError",
    "WorkflowType",
    "build_workflow",
]
