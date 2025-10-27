"""Tests for the agent state machine."""

from __future__ import annotations

import pytest

from ia_agent_fwk.agents.exceptions import InvalidStateTransitionError
from ia_agent_fwk.agents.state import AgentState, validate_transition


class TestAgentState:
    def test_has_five_states(self):
        states = list(AgentState)
        assert len(states) == 5

    def test_state_values(self):
        assert AgentState.IDLE == "IDLE"
        assert AgentState.RUNNING == "RUNNING"
        assert AgentState.WAITING_FOR_INPUT == "WAITING_FOR_INPUT"
        assert AgentState.COMPLETED == "COMPLETED"
        assert AgentState.FAILED == "FAILED"


class TestValidTransitions:
    """Test all 6 valid transitions succeed."""

    @pytest.mark.parametrize(
        ("from_state", "to_state"),
        [
            (AgentState.IDLE, AgentState.RUNNING),
            (AgentState.RUNNING, AgentState.WAITING_FOR_INPUT),
            (AgentState.RUNNING, AgentState.COMPLETED),
            (AgentState.RUNNING, AgentState.FAILED),
            (AgentState.WAITING_FOR_INPUT, AgentState.RUNNING),
            (AgentState.WAITING_FOR_INPUT, AgentState.FAILED),
        ],
    )
    def test_valid_transition_succeeds(self, from_state, to_state):
        # Should not raise
        validate_transition(from_state, to_state)


class TestInvalidTransitions:
    """Test all invalid transitions raise InvalidStateTransitionError."""

    @pytest.mark.parametrize(
        ("from_state", "to_state"),
        [
            # IDLE: only RUNNING is valid
            (AgentState.IDLE, AgentState.IDLE),
            (AgentState.IDLE, AgentState.WAITING_FOR_INPUT),
            (AgentState.IDLE, AgentState.COMPLETED),
            (AgentState.IDLE, AgentState.FAILED),
            # RUNNING: only WAITING_FOR_INPUT, COMPLETED, FAILED are valid
            (AgentState.RUNNING, AgentState.IDLE),
            (AgentState.RUNNING, AgentState.RUNNING),
            # WAITING_FOR_INPUT: only RUNNING, FAILED are valid
            (AgentState.WAITING_FOR_INPUT, AgentState.IDLE),
            (AgentState.WAITING_FOR_INPUT, AgentState.WAITING_FOR_INPUT),
            (AgentState.WAITING_FOR_INPUT, AgentState.COMPLETED),
            # COMPLETED: terminal — all transitions invalid
            (AgentState.COMPLETED, AgentState.IDLE),
            (AgentState.COMPLETED, AgentState.RUNNING),
            (AgentState.COMPLETED, AgentState.WAITING_FOR_INPUT),
            (AgentState.COMPLETED, AgentState.COMPLETED),
            (AgentState.COMPLETED, AgentState.FAILED),
            # FAILED: terminal — all transitions invalid
            (AgentState.FAILED, AgentState.IDLE),
            (AgentState.FAILED, AgentState.RUNNING),
            (AgentState.FAILED, AgentState.WAITING_FOR_INPUT),
            (AgentState.FAILED, AgentState.COMPLETED),
            (AgentState.FAILED, AgentState.FAILED),
        ],
    )
    def test_invalid_transition_raises(self, from_state, to_state):
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            validate_transition(from_state, to_state)
        assert from_state.value in str(exc_info.value)
        assert to_state.value in str(exc_info.value)
