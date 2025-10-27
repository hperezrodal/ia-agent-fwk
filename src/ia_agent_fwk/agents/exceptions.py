"""Agent exception hierarchy.

All agent-specific exceptions inherit from ``AgentError``.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for all agent errors."""


class AgentConfigError(AgentError):
    """Raised for agent configuration errors."""


class AgentTimeoutError(AgentError):
    """Raised when agent execution exceeds the configured timeout."""


class AgentMaxIterationsError(AgentError):
    """Raised when the reasoning loop exceeds the maximum iteration limit."""


class InvalidStateTransitionError(AgentError):
    """Raised when an invalid agent state transition is attempted.

    Attributes
    ----------
    from_state:
        The state the agent was in when the transition was attempted.
    to_state:
        The target state that was rejected.

    """

    def __init__(self, from_state: str, to_state: str) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Invalid state transition: {from_state} -> {to_state}")

    def __str__(self) -> str:
        return f"Invalid state transition: {self.from_state} -> {self.to_state}"
