"""Tool permission system.

``ToolPermissionManager`` enforces per-agent tool access control
with four permission modes.
"""

from __future__ import annotations

import logging
from enum import Enum

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.tools.config import ToolPermissionConfig
from ia_agent_fwk.tools.exceptions import ToolPermissionError

logger = logging.getLogger(__name__)


class PermissionMode(str, Enum):
    """Permission modes for tool access control."""

    allow_all = "allow_all"
    allow_list = "allow_list"
    deny_list = "deny_list"
    require_confirmation = "require_confirmation"


class ToolPermissionManager:
    """Enforce per-agent tool access control.

    Parameters
    ----------
    default_mode:
        The default permission mode when no per-agent config exists.
    agent_permissions:
        Per-agent permission configurations keyed by agent_id.

    """

    def __init__(
        self,
        default_mode: PermissionMode = PermissionMode.allow_all,
        agent_permissions: dict[str, ToolPermissionConfig] | None = None,
    ) -> None:
        self._default_mode = default_mode
        self._agent_permissions: dict[str, ToolPermissionConfig] = agent_permissions or {}

    def _get_config(self, agent_id: str) -> tuple[PermissionMode, ToolPermissionConfig]:
        """Resolve the effective permission config for an agent."""
        if agent_id in self._agent_permissions:
            cfg = self._agent_permissions[agent_id]
            mode = PermissionMode(cfg.mode)
            return mode, cfg
        # Fall back to default mode with empty lists
        return self._default_mode, ToolPermissionConfig(mode=self._default_mode.value)

    def check_permission(self, agent_id: str, tool_name: str) -> None:
        """Check whether an agent is allowed to use a tool.

        Parameters
        ----------
        agent_id:
            The agent identifier.
        tool_name:
            The tool name.

        Raises
        ------
        ToolPermissionError
            If the agent is not permitted to use the tool.

        """
        collector = get_metrics_collector()
        mode, cfg = self._get_config(agent_id)

        if mode == PermissionMode.allow_all:
            collector.increment("tool_permission_checks_total", labels={"mode": mode.value, "result": "allowed"})
            return

        if mode == PermissionMode.allow_list:
            if tool_name not in cfg.allowed:
                collector.increment("tool_permission_checks_total", labels={"mode": mode.value, "result": "denied"})
                msg = f"Tool '{tool_name}' is not in the allow list for agent '{agent_id}'."
                raise ToolPermissionError(msg)
            collector.increment("tool_permission_checks_total", labels={"mode": mode.value, "result": "allowed"})
            return

        if mode == PermissionMode.deny_list:
            if tool_name in cfg.denied:
                collector.increment("tool_permission_checks_total", labels={"mode": mode.value, "result": "denied"})
                msg = f"Tool '{tool_name}' is denied for agent '{agent_id}'."
                raise ToolPermissionError(msg)
            collector.increment("tool_permission_checks_total", labels={"mode": mode.value, "result": "allowed"})
            return

        if mode == PermissionMode.require_confirmation:
            if tool_name in cfg.require_confirmation:
                collector.increment("tool_permission_checks_total", labels={"mode": mode.value, "result": "denied"})
                msg = f"Tool '{tool_name}' requires human confirmation before execution."
                raise ToolPermissionError(msg)
            collector.increment("tool_permission_checks_total", labels={"mode": mode.value, "result": "allowed"})
            return

    def is_permitted(self, agent_id: str, tool_name: str) -> bool:
        """Check whether an agent is allowed to use a tool (boolean).

        Parameters
        ----------
        agent_id:
            The agent identifier.
        tool_name:
            The tool name.

        Returns
        -------
        bool
            ``True`` if permitted, ``False`` otherwise.

        """
        try:
            self.check_permission(agent_id, tool_name)
        except ToolPermissionError:
            return False
        return True
