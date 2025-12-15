"""Abstract base class for workflow orchestrators.

``OrchestratorBase`` defines the common interface and helper methods
shared by sequential, parallel, and conditional workflow orchestrators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.llm.models import TokenUsage
from ia_agent_fwk.orchestration.models import StepResult, WorkflowResult

if TYPE_CHECKING:
    from ia_agent_fwk.agents.config import AgentResult
    from ia_agent_fwk.orchestration.models import WorkflowStep

InputTransformFn = Callable[[str], str]


class OrchestratorBase(ABC):
    """Abstract base class for all workflow orchestrators.

    Parameters
    ----------
    timeout:
        Optional workflow-level timeout in seconds. If ``None``, no
        timeout is applied.
    input_transforms:
        Optional mapping of step name → callable to transform the
        step's input text before execution.

    """

    def __init__(
        self,
        timeout: float | None = None,
        input_transforms: dict[str, InputTransformFn] | None = None,
    ) -> None:
        self._timeout = timeout
        self._input_transforms: dict[str, InputTransformFn] = input_transforms or {}

    @property
    @abstractmethod
    def orchestrator_type(self) -> str:
        """Return the orchestrator type identifier."""
        ...

    @abstractmethod
    async def execute(self, input_text: str) -> WorkflowResult:
        """Execute the workflow with the given input.

        Parameters
        ----------
        input_text:
            The initial input text for the workflow.

        Returns
        -------
        WorkflowResult
            The aggregated result of the workflow execution.

        """
        ...

    @staticmethod
    def _create_step_result(
        step: WorkflowStep,
        agent_result: AgentResult,
        duration_ms: float,
    ) -> StepResult:
        """Map an ``AgentResult`` to a ``StepResult``."""
        return StepResult(
            step_name=step.name,
            agent_name=step.agent_name,
            output=agent_result.output,
            usage=agent_result.usage,
            duration_ms=duration_ms,
            state=agent_result.state.value,
            error=agent_result.error,
        )

    @staticmethod
    def _aggregate_usage(step_results: list[StepResult]) -> TokenUsage:
        """Sum token usage across all step results."""
        total_prompt = 0
        total_completion = 0
        for sr in step_results:
            total_prompt += sr.usage.prompt_tokens
            total_completion += sr.usage.completion_tokens
        return TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
        )

    def _apply_input_transform(self, step: WorkflowStep, input_text: str) -> str:
        """Apply the input transform for a step, if configured.

        Looks up a transform by step name in ``_input_transforms``.
        """
        transform = self._input_transforms.get(step.name)
        if transform is not None:
            return transform(input_text)
        return input_text

    @staticmethod
    def _build_agent_config(step: WorkflowStep) -> Any:
        """Build an ``AgentConfig`` from a ``WorkflowStep``."""
        from ia_agent_fwk.agents.config import AgentConfig  # noqa: PLC0415

        if step.agent_config:
            config_data: dict[str, Any] = {
                "name": step.agent_name,
                "agent_type": step.agent_name,
                **step.agent_config,
            }
            return AgentConfig(**config_data)
        return AgentConfig(
            name=step.agent_name,
            agent_type=step.agent_name,
        )
