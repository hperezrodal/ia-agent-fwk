"""Agent factory for configuration-driven instantiation.

``AgentFactory.create()`` looks up the agent class from ``AgentRegistry``,
creates an ``LLMProvider`` via ``LLMProviderFactory.create()``, and
constructs the agent with a ``DefaultToolExecutor``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ia_agent_fwk.agents.exceptions import AgentConfigError
from ia_agent_fwk.agents.registry import AgentRegistry
from ia_agent_fwk.llm.exceptions import LLMConfigError
from ia_agent_fwk.llm.factory import LLMProviderFactory

if TYPE_CHECKING:
    from ia_agent_fwk.agents.base import Agent
    from ia_agent_fwk.agents.config import AgentConfig
    from ia_agent_fwk.config.settings import LLMSettings, ToolSandboxingSettings
    from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.tools.config import ToolsConfig
    from ia_agent_fwk.tools.registry import ToolRegistry


class AgentFactory:
    """Configuration-driven agent factory."""

    @classmethod
    def create(
        cls,
        config: AgentConfig,
        llm_settings: LLMSettings,
        *,
        tools_config: ToolsConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        sandboxing_config: ToolSandboxingSettings | None = None,
        memory_backend: MemoryBackend | None = None,
        conversation_backend: ConversationMemoryBackend | None = None,
    ) -> Agent:
        """Create an agent from configuration.

        Parameters
        ----------
        config:
            Agent configuration.
        llm_settings:
            LLM settings for provider instantiation.
        tools_config:
            Tool system configuration. If ``None``, uses defaults.
        tool_registry:
            Pre-configured tool registry. If ``None``, creates one with
            built-in tools (if enabled).
        sandboxing_config:
            Tool sandboxing settings for domain/path allowlists.

        Returns
        -------
        Agent
            A fully configured agent instance.

        Raises
        ------
        AgentConfigError
            If the agent type is unknown or LLM provider creation fails.

        """
        # Look up agent class
        agent_cls = AgentRegistry.get(config.agent_type)

        # Create LLM provider
        try:
            provider = LLMProviderFactory.create(
                llm_settings,
                provider_name=config.provider_name,
            )
        except LLMConfigError as exc:
            msg = f"Failed to create LLM provider '{config.provider_name}' for agent '{config.name}': {exc}"
            raise AgentConfigError(msg) from exc

        # Create tool executor (deferred imports to avoid circular dependency
        # between agents/__init__.py -> factory -> tools/executor -> agents/protocols)
        from ia_agent_fwk.tools.builtin import register_builtin_tools  # noqa: PLC0415
        from ia_agent_fwk.tools.config import ToolsConfig as _ToolsConfig  # noqa: PLC0415
        from ia_agent_fwk.tools.executor import DefaultToolExecutor  # noqa: PLC0415
        from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager  # noqa: PLC0415
        from ia_agent_fwk.tools.registry import ToolRegistry as _ToolRegistry  # noqa: PLC0415

        tc = tools_config or _ToolsConfig()

        if tool_registry is None:
            tool_registry = _ToolRegistry()
            if tc.builtin_tools_enabled:
                register_builtin_tools(tool_registry, sandboxing_config=sandboxing_config)

        permission_manager = ToolPermissionManager(
            default_mode=PermissionMode(tc.default_permission_mode),
        )

        tool_executor = DefaultToolExecutor(
            registry=tool_registry,
            permission_manager=permission_manager,
            agent_id=config.name,
            default_timeout=tc.default_timeout,
        )

        return agent_cls(
            config=config,
            provider=provider,
            tool_executor=tool_executor,
            memory_backend=memory_backend,
            conversation_backend=conversation_backend,
        )
