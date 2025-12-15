"""Orchestration exception hierarchy.

All orchestration-specific exceptions inherit from ``OrchestrationError``.
"""

from __future__ import annotations


class OrchestrationError(Exception):
    """Base exception for all orchestration errors."""


class WorkflowStepError(OrchestrationError):
    """Raised when a workflow step fails.

    Attributes
    ----------
    step_name:
        Name of the step that failed.
    agent_name:
        Name of the agent that failed.
    cause:
        The original exception, if any.

    """

    def __init__(
        self,
        message: str,
        *,
        step_name: str,
        agent_name: str,
        cause: Exception | None = None,
    ) -> None:
        self.step_name = step_name
        self.agent_name = agent_name
        self.cause = cause
        super().__init__(message)


class DelegationDepthExceededError(OrchestrationError):
    """Raised when recursive delegation depth is exhausted.

    Attributes
    ----------
    max_depth:
        The maximum delegation depth that was exceeded.

    """

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        super().__init__(
            f"Maximum delegation depth ({max_depth}) exceeded. Cannot delegate further to prevent infinite recursion."
        )


class WorkflowDefinitionError(OrchestrationError):
    """Raised for invalid workflow definitions."""


class WorkflowTimeoutError(OrchestrationError):
    """Raised when a workflow exceeds its configured timeout.

    Attributes
    ----------
    timeout:
        The timeout value in seconds that was exceeded.

    """

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"Workflow timed out after {timeout}s")
