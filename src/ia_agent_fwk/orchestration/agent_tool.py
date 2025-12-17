"""Agent-as-Tool adapter.

Wraps an agent (via factory callable) as a ``Tool`` so it can be
invoked through the standard tool-calling mechanism by other agents.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer
from ia_agent_fwk.tools.base import Tool

if TYPE_CHECKING:
    from ia_agent_fwk.agents.config import AgentConfig
    from ia_agent_fwk.orchestration.models import AgentFactoryFn
    from ia_agent_fwk.tools.base import ToolContext

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

# ---------------------------------------------------------------------------
# Input / Output schemas
# ---------------------------------------------------------------------------


class AgentToolInput(BaseModel):
    """Input schema for the agent-as-tool adapter."""

    task: str = Field(..., description="The task description for the agent.")
    context: str | None = Field(default=None, description="Optional additional context.")


class AgentToolOutput(BaseModel):
    """Output schema for the agent-as-tool adapter."""

    model_config = ConfigDict(frozen=True)

    output: str
    iterations: int
    state: str
    usage: dict[str, int]


class AgentTool(Tool):
    """Wrap an agent as a ``Tool`` for use by other agents.

    Parameters
    ----------
    agent_config:
        Configuration for the agent to wrap.
    agent_factory:
        Callable that creates an ``Agent`` from an ``AgentConfig``.
    tool_name:
        Custom tool name. Defaults to ``agent_<agent_config.name>``.
    tool_description:
        Custom tool description.
    delegation_depth:
        Remaining delegation depth. When 0, execution is refused.

    """

    def __init__(
        self,
        agent_config: AgentConfig,
        agent_factory: AgentFactoryFn,
        tool_name: str | None = None,
        tool_description: str | None = None,
        delegation_depth: int = 3,
    ) -> None:
        self._agent_config = agent_config
        self._agent_factory = agent_factory
        self._tool_name = tool_name or f"agent_{agent_config.name}"
        self._tool_description = tool_description or f"Invoke the {agent_config.name} agent"
        self._delegation_depth = delegation_depth

    @property
    def name(self) -> str:
        """Return the tool name."""
        return self._tool_name

    @property
    def description(self) -> str:
        """Return the tool description."""
        return self._tool_description

    @property
    def input_schema(self) -> type[BaseModel]:
        """Return the input schema class."""
        return AgentToolInput

    @property
    def output_schema(self) -> type[BaseModel]:
        """Return the output schema class."""
        return AgentToolOutput

    @property
    def delegation_depth(self) -> int:
        """Return the remaining delegation depth."""
        return self._delegation_depth

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Execute the wrapped agent.

        Parameters
        ----------
        validated_input:
            Validated ``AgentToolInput`` instance.
        context:
            Tool execution context.

        Returns
        -------
        AgentToolOutput
            The mapped agent result.

        """
        assert isinstance(validated_input, AgentToolInput)  # noqa: S101

        collector = get_metrics_collector()

        if self._delegation_depth <= 0:
            collector.increment(
                "agent_tool_executions_total",
                labels={"agent_name": self._agent_config.name, "status": "depth_exceeded"},
            )
            logger.warning(
                "Delegation depth exceeded for agent '%s'",
                self._agent_config.name,
                extra={
                    "orchestration_data": {
                        "event": "delegation_depth_exceeded",
                        "agent_name": self._agent_config.name,
                        "tool_name": self._tool_name,
                    }
                },
            )
            return AgentToolOutput(
                output=f"Delegation depth exceeded. Cannot invoke agent '{self._agent_config.name}'.",
                iterations=0,
                state=AgentState.FAILED.value,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        collector.increment("supervisor_delegations_total", labels={"sub_agent": self._agent_config.name})
        start = time.monotonic()
        agent = self._agent_factory(self._agent_config)
        task = validated_input.task
        if validated_input.context:
            task = f"{task}\n\nContext: {validated_input.context}"

        result = await agent.run(task)
        duration_ms = (time.monotonic() - start) * 1000

        status = "success" if result.state != AgentState.FAILED else "failure"
        collector.increment(
            "agent_tool_executions_total", labels={"agent_name": self._agent_config.name, "status": status}
        )
        collector.observe(
            "agent_tool_duration_seconds", duration_ms / 1000, labels={"agent_name": self._agent_config.name}
        )

        logger.info(
            "AgentTool delegation: agent=%s, status=%s, iterations=%d (%.1fms)",
            self._agent_config.name,
            status,
            result.iterations,
            duration_ms,
            extra={
                "orchestration_data": {
                    "event": "agent_tool_executed",
                    "agent_name": self._agent_config.name,
                    "tool_name": self._tool_name,
                    "status": status,
                    "iterations": result.iterations,
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                    "duration_ms": round(duration_ms, 1),
                    "delegation_depth": self._delegation_depth,
                }
            },
        )

        return AgentToolOutput(
            output=result.output,
            iterations=result.iterations,
            state=result.state.value,
            usage={
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        )
