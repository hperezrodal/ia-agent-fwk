"""Tests for orchestration exception hierarchy."""

from __future__ import annotations

import pytest

from ia_agent_fwk.orchestration.exceptions import (
    DelegationDepthExceededError,
    OrchestrationError,
    WorkflowDefinitionError,
    WorkflowStepError,
    WorkflowTimeoutError,
)


@pytest.mark.unit
class TestOrchestrationExceptions:
    def test_orchestration_error_is_exception(self):
        err = OrchestrationError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"

    def test_workflow_step_error_carries_context(self):
        cause = ValueError("bad value")
        err = WorkflowStepError(
            "Step failed",
            step_name="step_1",
            agent_name="researcher",
            cause=cause,
        )
        assert isinstance(err, OrchestrationError)
        assert err.step_name == "step_1"
        assert err.agent_name == "researcher"
        assert err.cause is cause
        assert str(err) == "Step failed"

    def test_workflow_step_error_no_cause(self):
        err = WorkflowStepError(
            "Step failed",
            step_name="step_2",
            agent_name="analyzer",
        )
        assert err.cause is None

    def test_delegation_depth_exceeded_error(self):
        err = DelegationDepthExceededError(max_depth=5)
        assert isinstance(err, OrchestrationError)
        assert err.max_depth == 5
        assert "5" in str(err)
        assert "exceeded" in str(err).lower()

    def test_workflow_definition_error(self):
        err = WorkflowDefinitionError("Invalid workflow")
        assert isinstance(err, OrchestrationError)
        assert str(err) == "Invalid workflow"

    def test_workflow_timeout_error(self):
        err = WorkflowTimeoutError(timeout=300.0)
        assert isinstance(err, OrchestrationError)
        assert err.timeout == 300.0
        assert "300.0" in str(err)

    def test_exception_hierarchy(self):
        """All orchestration exceptions inherit from OrchestrationError."""
        assert issubclass(WorkflowStepError, OrchestrationError)
        assert issubclass(DelegationDepthExceededError, OrchestrationError)
        assert issubclass(WorkflowDefinitionError, OrchestrationError)
        assert issubclass(WorkflowTimeoutError, OrchestrationError)
        assert issubclass(OrchestrationError, Exception)
