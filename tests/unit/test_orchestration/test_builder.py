"""Tests for the workflow builder."""

from __future__ import annotations

import pytest

from ia_agent_fwk.orchestration.builder import build_workflow
from ia_agent_fwk.orchestration.conditional import ConditionalWorkflow
from ia_agent_fwk.orchestration.exceptions import WorkflowDefinitionError
from ia_agent_fwk.orchestration.models import WorkflowDefinition, WorkflowType
from ia_agent_fwk.orchestration.parallel import ParallelWorkflow
from ia_agent_fwk.orchestration.sequential import SequentialWorkflow


@pytest.mark.unit
class TestBuildWorkflow:
    def test_build_sequential_workflow(self, mock_agent_factory):
        defn = WorkflowDefinition(
            name="test_seq",
            workflow_type=WorkflowType.sequential,
            steps=[
                {"name": "step_1", "agent_name": "agent_a"},
                {"name": "step_2", "agent_name": "agent_b"},
            ],
        )
        workflow = build_workflow(defn, mock_agent_factory)

        assert isinstance(workflow, SequentialWorkflow)
        assert workflow.orchestrator_type == "sequential"

    def test_build_parallel_workflow(self, mock_agent_factory):
        defn = WorkflowDefinition(
            name="test_par",
            workflow_type=WorkflowType.parallel,
            steps=[
                {"name": "step_1", "agent_name": "agent_a"},
                {"name": "step_2", "agent_name": "agent_b"},
            ],
            config={"failure_policy": "collect_errors"},
        )
        workflow = build_workflow(defn, mock_agent_factory)

        assert isinstance(workflow, ParallelWorkflow)
        assert workflow.orchestrator_type == "parallel"

    def test_build_conditional_workflow(self, mock_agent_factory):
        defn = WorkflowDefinition(
            name="test_cond",
            workflow_type=WorkflowType.conditional,
            steps=[
                {"name": "billing", "agent_name": "billing_agent"},
                {"name": "tech", "agent_name": "tech_agent"},
            ],
        )
        router = lambda x: "billing"  # noqa: E731
        workflow = build_workflow(defn, mock_agent_factory, router=router)

        assert isinstance(workflow, ConditionalWorkflow)
        assert workflow.orchestrator_type == "conditional"

    def test_build_conditional_without_router_raises(self, mock_agent_factory):
        defn = WorkflowDefinition(
            name="test_cond",
            workflow_type=WorkflowType.conditional,
            steps=[{"name": "s1", "agent_name": "a1"}],
        )
        with pytest.raises(WorkflowDefinitionError, match="no router"):
            build_workflow(defn, mock_agent_factory)

    def test_build_empty_steps_raises(self, mock_agent_factory):
        defn = WorkflowDefinition(
            name="empty",
            workflow_type=WorkflowType.sequential,
            steps=[],
        )
        with pytest.raises(WorkflowDefinitionError, match="no steps"):
            build_workflow(defn, mock_agent_factory)

    def test_build_missing_agent_name_raises(self, mock_agent_factory):
        defn = WorkflowDefinition(
            name="bad",
            workflow_type=WorkflowType.sequential,
            steps=[{"name": "step_1"}],  # Missing agent_name
        )
        with pytest.raises(WorkflowDefinitionError, match="agent_name"):
            build_workflow(defn, mock_agent_factory)

    def test_build_with_timeout(self, mock_agent_factory):
        defn = WorkflowDefinition(
            name="test",
            workflow_type=WorkflowType.sequential,
            steps=[{"name": "s1", "agent_name": "a1"}],
            config={"timeout": 60.0},
        )
        workflow = build_workflow(defn, mock_agent_factory)
        assert isinstance(workflow, SequentialWorkflow)
        assert workflow._timeout == 60.0

    def test_build_auto_names_steps(self, mock_agent_factory):
        """Steps without names get auto-generated names."""
        defn = WorkflowDefinition(
            name="test",
            workflow_type=WorkflowType.sequential,
            steps=[
                {"agent_name": "agent_a"},
                {"agent_name": "agent_b"},
            ],
        )
        workflow = build_workflow(defn, mock_agent_factory)
        assert isinstance(workflow, SequentialWorkflow)
