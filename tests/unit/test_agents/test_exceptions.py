"""Tests for the agent exception hierarchy."""

from __future__ import annotations

from ia_agent_fwk.agents.exceptions import (
    AgentConfigError,
    AgentError,
    AgentMaxIterationsError,
    AgentTimeoutError,
    InvalidStateTransitionError,
)


class TestAgentError:
    def test_base_exception(self):
        err = AgentError("base error")
        assert str(err) == "base error"
        assert isinstance(err, Exception)

    def test_config_error_inherits(self):
        err = AgentConfigError("config error")
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)

    def test_timeout_error_inherits(self):
        err = AgentTimeoutError("timed out")
        assert isinstance(err, AgentError)

    def test_max_iterations_error_inherits(self):
        err = AgentMaxIterationsError("max iterations reached")
        assert isinstance(err, AgentError)


class TestInvalidStateTransitionError:
    def test_has_from_and_to_state(self):
        err = InvalidStateTransitionError("IDLE", "COMPLETED")
        assert err.from_state == "IDLE"
        assert err.to_state == "COMPLETED"

    def test_str_contains_both_states(self):
        err = InvalidStateTransitionError("IDLE", "COMPLETED")
        msg = str(err)
        assert "IDLE" in msg
        assert "COMPLETED" in msg

    def test_inherits_from_agent_error(self):
        err = InvalidStateTransitionError("RUNNING", "IDLE")
        assert isinstance(err, AgentError)

    def test_is_catchable_as_exception(self):
        err = InvalidStateTransitionError("FAILED", "RUNNING")
        assert isinstance(err, Exception)
