"""Tests for orchestration Pydantic models."""

from __future__ import annotations

import pytest

from ia_agent_fwk.llm.models import TokenUsage
from ia_agent_fwk.orchestration.models import (
    FailurePolicy,
    RetryPolicy,
    StepResult,
    WorkflowDefinition,
    WorkflowResult,
    WorkflowStep,
    WorkflowType,
)


@pytest.mark.unit
class TestFailurePolicy:
    def test_fail_fast_value(self):
        assert FailurePolicy.fail_fast.value == "fail_fast"

    def test_collect_errors_value(self):
        assert FailurePolicy.collect_errors.value == "collect_errors"


@pytest.mark.unit
class TestWorkflowType:
    def test_all_types(self):
        assert WorkflowType.sequential.value == "sequential"
        assert WorkflowType.parallel.value == "parallel"
        assert WorkflowType.conditional.value == "conditional"
        assert WorkflowType.supervisor.value == "supervisor"


@pytest.mark.unit
class TestRetryPolicy:
    def test_defaults(self):
        policy = RetryPolicy()
        assert policy.max_retries == 0
        assert policy.backoff_base == 2.0
        assert policy.backoff_max == 60.0

    def test_custom_values(self):
        policy = RetryPolicy(max_retries=3, backoff_base=1.5, backoff_max=30.0)
        assert policy.max_retries == 3
        assert policy.backoff_base == 1.5
        assert policy.backoff_max == 30.0

    def test_frozen(self):
        policy = RetryPolicy()
        with pytest.raises(Exception):  # noqa: B017, PT011
            policy.max_retries = 5  # type: ignore[misc]


@pytest.mark.unit
class TestWorkflowStep:
    def test_minimal_step(self):
        step = WorkflowStep(name="step_1", agent_name="researcher")
        assert step.name == "step_1"
        assert step.agent_name == "researcher"
        assert step.agent_config is None
        assert step.retry_policy is None
        assert step.fallback_agent_name is None
        assert step.input_transform is None

    def test_full_step(self):
        step = WorkflowStep(
            name="step_1",
            agent_name="researcher",
            agent_config={"system_prompt": "Research this topic."},
            retry_policy=RetryPolicy(max_retries=2),
            fallback_agent_name="backup_researcher",
            input_transform="prefix with 'Analyze: '",
        )
        assert step.retry_policy is not None
        assert step.retry_policy.max_retries == 2
        assert step.fallback_agent_name == "backup_researcher"


@pytest.mark.unit
class TestStepResult:
    def test_step_result_creation(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
        sr = StepResult(
            step_name="step_1",
            agent_name="researcher",
            output="research results",
            usage=usage,
            duration_ms=150.0,
            state="COMPLETED",
        )
        assert sr.step_name == "step_1"
        assert sr.output == "research results"
        assert sr.usage.total_tokens == 30
        assert sr.error is None

    def test_step_result_with_error(self):
        usage = TokenUsage(prompt_tokens=5, completion_tokens=0)
        sr = StepResult(
            step_name="step_2",
            agent_name="analyzer",
            output="",
            usage=usage,
            duration_ms=50.0,
            state="FAILED",
            error="LLM timeout",
        )
        assert sr.state == "FAILED"
        assert sr.error == "LLM timeout"

    def test_step_result_frozen(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        sr = StepResult(
            step_name="s",
            agent_name="a",
            output="",
            usage=usage,
            duration_ms=0.0,
            state="COMPLETED",
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            sr.output = "new"  # type: ignore[misc]


@pytest.mark.unit
class TestWorkflowResult:
    def test_workflow_result_success(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        result = WorkflowResult(
            output="final answer",
            step_results=[],
            usage=usage,
            duration_ms=500.0,
        )
        assert result.output == "final answer"
        assert result.error is None
        assert result.metadata is None

    def test_workflow_result_with_error(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        result = WorkflowResult(
            output="",
            step_results=[],
            usage=usage,
            duration_ms=100.0,
            error="Workflow failed",
            metadata={"route_key": "billing"},
        )
        assert result.error == "Workflow failed"
        assert result.metadata == {"route_key": "billing"}

    def test_workflow_result_frozen(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        result = WorkflowResult(
            output="",
            step_results=[],
            usage=usage,
            duration_ms=0.0,
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            result.output = "new"  # type: ignore[misc]


@pytest.mark.unit
class TestWorkflowDefinition:
    def test_minimal_definition(self):
        defn = WorkflowDefinition(
            name="my_workflow",
            workflow_type=WorkflowType.sequential,
            steps=[{"name": "s1", "agent_name": "researcher"}],
        )
        assert defn.name == "my_workflow"
        assert defn.description == ""
        assert defn.workflow_type == WorkflowType.sequential
        assert len(defn.steps) == 1

    def test_definition_from_dict(self):
        data = {
            "name": "test",
            "workflow_type": "parallel",
            "steps": [
                {"name": "a", "agent_name": "agent_a"},
                {"name": "b", "agent_name": "agent_b"},
            ],
            "config": {"failure_policy": "collect_errors"},
        }
        defn = WorkflowDefinition.model_validate(data)
        assert defn.workflow_type == WorkflowType.parallel
        assert len(defn.steps) == 2
        assert defn.config["failure_policy"] == "collect_errors"

    def test_definition_serialization(self):
        defn = WorkflowDefinition(
            name="test",
            workflow_type=WorkflowType.conditional,
            steps=[{"name": "s1", "agent_name": "router"}],
        )
        data = defn.model_dump()
        assert data["name"] == "test"
        assert data["workflow_type"] == "conditional"
        roundtrip = WorkflowDefinition.model_validate(data)
        assert roundtrip == defn
