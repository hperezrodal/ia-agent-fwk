"""Supervisor agent with LLM-driven dynamic delegation.

``SupervisorAgent`` extends the ``Agent`` ABC and uses sub-agents
(exposed as tools via ``AgentTool``) for dynamic task delegation.
The standard reasoning loop handles delegation through tool calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer
from ia_agent_fwk.orchestration.agent_tool import AgentTool

if TYPE_CHECKING:
    from collections.abc import Callable

    from ia_agent_fwk.agents.config import AgentConfig
    from ia_agent_fwk.agents.protocols import ToolExecutor
    from ia_agent_fwk.llm.base import LLMProvider

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class SupervisorAgent(Agent):
    """LLM-driven supervisor that delegates tasks to sub-agents.

    The supervisor receives a list of sub-agent configurations and
    descriptions. Each sub-agent is wrapped as an ``AgentTool`` and
    registered in a dedicated ``ToolRegistry``. The standard reasoning
    loop then invokes sub-agents through tool calls.

    Parameters
    ----------
    config:
        Agent configuration for the supervisor itself.
    provider:
        LLM provider for the supervisor's reasoning.
    sub_agents:
        List of ``(AgentConfig, description)`` tuples for sub-agents.
    agent_factory:
        Callable that creates an ``Agent`` from an ``AgentConfig``.
    max_delegation_depth:
        Maximum recursive delegation depth.
    tool_executor:
        Optional pre-configured tool executor. If provided, sub-agent
        tools are still added to a new registry wrapping this executor.

    """

    def __init__(  # noqa: PLR0913
        self,
        config: AgentConfig,
        provider: LLMProvider,
        sub_agents: list[tuple[AgentConfig, str]],
        agent_factory: Callable[[AgentConfig], Agent],
        max_delegation_depth: int = 3,
        tool_executor: ToolExecutor | None = None,  # noqa: ARG002
    ) -> None:
        # Build the tool infrastructure for sub-agents
        from ia_agent_fwk.tools.executor import DefaultToolExecutor  # noqa: PLC0415
        from ia_agent_fwk.tools.permissions import (  # noqa: PLC0415
            PermissionMode,
            ToolPermissionManager,
        )
        from ia_agent_fwk.tools.registry import ToolRegistry  # noqa: PLC0415

        registry = ToolRegistry()
        self._sub_agent_descriptions: list[tuple[str, str]] = []

        collector = get_metrics_collector()
        for sub_config, description in sub_agents:
            agent_tool = AgentTool(
                agent_config=sub_config,
                agent_factory=agent_factory,
                tool_description=description,
                delegation_depth=max_delegation_depth - 1,
            )
            registry.register(agent_tool)
            self._sub_agent_descriptions.append((agent_tool.name, description))

        collector.increment("supervisor_instances_created_total")
        collector.observe("supervisor_sub_agents_count", len(sub_agents))
        logger.info(
            "SupervisorAgent created: name=%s, sub_agents=%d, max_depth=%d",
            config.name,
            len(sub_agents),
            max_delegation_depth,
            extra={
                "orchestration_data": {
                    "event": "supervisor_created",
                    "supervisor_name": config.name,
                    "sub_agent_count": len(sub_agents),
                    "max_delegation_depth": max_delegation_depth,
                    "sub_agents": [name for name, _ in self._sub_agent_descriptions],
                }
            },
        )

        permission_manager = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
        )

        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=permission_manager,
            agent_id=config.name,
            default_timeout=float(config.execution_timeout),
        )

        super().__init__(config=config, provider=provider, tool_executor=executor)
        self._max_delegation_depth = max_delegation_depth

    @property
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        return "supervisor"

    def _build_sub_agent_prompt(self) -> str:
        """Generate a system prompt section describing available sub-agents."""
        if not self._sub_agent_descriptions:
            return ""

        lines = ["\n\nAvailable sub-agents (invoke via tool calls):"]
        for tool_name, description in self._sub_agent_descriptions:
            lines.append(f"- {tool_name}: {description}")

        lines.append(
            "\nTo delegate a task to a sub-agent, call the corresponding tool "
            "with a 'task' parameter describing what the sub-agent should do. "
            "You may invoke multiple sub-agents or the same sub-agent multiple times. "
            "When you have gathered all needed information, provide a final answer directly."
        )
        return "\n".join(lines)

    async def on_start(self) -> None:
        """Inject sub-agent descriptions into the context system prompt."""
        if self._context is not None:
            sub_agent_section = self._build_sub_agent_prompt()
            if sub_agent_section:
                # Append to existing system prompt
                current_prompt = self._context.system_prompt
                if sub_agent_section not in current_prompt:
                    # Recreate context with augmented prompt
                    augmented_prompt = current_prompt + sub_agent_section
                    self._context._system_prompt = augmented_prompt  # noqa: SLF001


# Register with AgentRegistry so the type is discoverable.
# Note: SupervisorAgent requires non-standard init parameters (sub_agents,
# agent_factory) and cannot be created via AgentFactory.create() directly.
# It must be instantiated programmatically.
from ia_agent_fwk.agents.registry import AgentRegistry  # noqa: E402

AgentRegistry.register("supervisor", SupervisorAgent)
