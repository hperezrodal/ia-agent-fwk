"""Tool registry for managing tool instances.

``ToolRegistry`` provides registration, lookup, listing, removal, and
OpenAI-format schema export for tools.
"""

from __future__ import annotations

import builtins
import logging
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.tools.exceptions import ToolNotFoundError, ToolValidationError

if TYPE_CHECKING:
    from ia_agent_fwk.tools.base import Tool
    from ia_agent_fwk.tools.permissions import ToolPermissionManager

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tool instances.

    Stores tool instances by name and provides discovery, lookup,
    and OpenAI-format schema export.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        """Register a tool instance.

        Parameters
        ----------
        tool:
            The tool instance to register.
        replace:
            If ``True``, overwrite an existing tool with the same name.

        Raises
        ------
        ToolValidationError
            If a tool with the same name is already registered and
            ``replace`` is ``False``.

        """
        if tool.name in self._tools and not replace:
            msg = f"Tool '{tool.name}' is already registered. Use replace=True to overwrite."
            raise ToolValidationError(msg)
        self._tools[tool.name] = tool
        collector = get_metrics_collector()
        collector.increment("tool_registry_registrations_total")
        collector.observe("tool_registry_size", len(self._tools))

    def get(self, name: str) -> Tool:
        """Look up a tool by name.

        Parameters
        ----------
        name:
            The tool name.

        Returns
        -------
        Tool
            The registered tool instance.

        Raises
        ------
        ToolNotFoundError
            If no tool with the given name is registered.

        """
        collector = get_metrics_collector()
        tool = self._tools.get(name)
        if tool is None:
            collector.increment("tool_registry_lookups_total", labels={"status": "miss"})
            msg = f"Tool '{name}' not found in registry."
            raise ToolNotFoundError(msg)
        collector.increment("tool_registry_lookups_total", labels={"status": "hit"})
        return tool

    def list(self, *, tags: builtins.list[str] | None = None) -> builtins.list[Tool]:
        """Return all registered tools, optionally filtered by tags.

        Parameters
        ----------
        tags:
            If provided, only return tools that have at least one
            matching tag.

        Returns
        -------
        list[Tool]
            The matching tool instances.

        """
        tools = builtins.list(self._tools.values())
        if tags is not None:
            tag_set = set(tags)
            tools = [t for t in tools if tag_set.intersection(t.tags)]
        return tools

    def has(self, name: str) -> bool:
        """Check whether a tool is registered.

        Parameters
        ----------
        name:
            The tool name.

        Returns
        -------
        bool
            ``True`` if the tool exists, ``False`` otherwise.

        """
        return name in self._tools

    def remove(self, name: str) -> None:
        """Remove a tool by name.

        Parameters
        ----------
        name:
            The tool name.

        Raises
        ------
        ToolNotFoundError
            If no tool with the given name is registered.

        """
        if name not in self._tools:
            msg = f"Tool '{name}' not found in registry."
            raise ToolNotFoundError(msg)
        del self._tools[name]
        collector = get_metrics_collector()
        collector.increment("tool_registry_removals_total")
        collector.observe("tool_registry_size", len(self._tools))

    def openai_schemas(
        self,
        *,
        agent_id: str | None = None,
        permission_manager: ToolPermissionManager | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """Export registered tools in OpenAI function-calling format.

        Parameters
        ----------
        agent_id:
            If provided together with ``permission_manager``, only
            include tools the agent is permitted to use.
        permission_manager:
            The permission manager for filtering.

        Returns
        -------
        list[dict[str, Any]]
            A list of dicts in OpenAI ``tools`` parameter format.

        """
        schemas: builtins.list[dict[str, Any]] = []
        for tool in self._tools.values():
            if (
                agent_id is not None
                and permission_manager is not None
                and not permission_manager.is_permitted(agent_id, tool.name)
            ):
                continue

            json_schema = tool.input_schema.model_json_schema()
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": json_schema,
                    },
                }
            )

        return schemas
