"""Agent state machine with transition validation.

Defines the ``AgentState`` enum and validates transitions between states.
Terminal states (COMPLETED, FAILED) accept no outgoing transitions.
"""

from __future__ import annotations

from enum import Enum

from ia_agent_fwk.agents.exceptions import InvalidStateTransitionError

# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class AgentState(str, Enum):
    """Finite state machine states for agent lifecycle."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    WAITING_FOR_INPUT = "WAITING_FOR_INPUT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.RUNNING},
    AgentState.RUNNING: {
        AgentState.WAITING_FOR_INPUT,
        AgentState.COMPLETED,
        AgentState.FAILED,
    },
    AgentState.WAITING_FOR_INPUT: {
        AgentState.RUNNING,
        AgentState.FAILED,
    },
    AgentState.COMPLETED: set(),
    AgentState.FAILED: set(),
}


# ---------------------------------------------------------------------------
# Transition validation
# ---------------------------------------------------------------------------


def validate_transition(from_state: AgentState, to_state: AgentState) -> None:
    """Validate and enforce a state transition.

    Raises
    ------
    InvalidStateTransitionError
        If the transition is not in the valid transition table.

    """
    valid_targets = _VALID_TRANSITIONS.get(from_state, set())
    if to_state not in valid_targets:
        raise InvalidStateTransitionError(from_state.value, to_state.value)
