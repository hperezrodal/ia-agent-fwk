"""Agent registry for type registration and lookup.

Uses class-level variables (``ClassVar``) for the registry dictionary,
consistent with the ``LLMProviderFactory`` pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from ia_agent_fwk.agents.exceptions import AgentConfigError
from ia_agent_fwk.observability.metrics import get_metrics_collector

if TYPE_CHECKING:
    from ia_agent_fwk.agents.base import Agent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for agent type registration and lookup."""

    _registry: ClassVar[dict[str, type[Agent]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        agent_class: type[Agent],
        *,
        replace: bool = False,
    ) -> None:
        """Register an agent class under *name*.

        Parameters
        ----------
        name:
            Logical name for the agent type.
        agent_class:
            The agent class to register.
        replace:
            If ``True``, silently overwrite an existing registration.

        Raises
        ------
        AgentConfigError
            If *name* is already registered and *replace* is ``False``.

        """
        if name in cls._registry and not replace:
            msg = f"Agent type '{name}' is already registered. Use replace=True to overwrite."
            raise AgentConfigError(msg)
        cls._registry[name] = agent_class
        collector = get_metrics_collector()
        collector.increment(
            "agent_registry_registrations_total",
            labels={"agent_type": name, "replaced": str(replace)},
        )
        logger.info(
            "Registered agent type '%s' (class=%s, replace=%s)",
            name,
            agent_class.__name__,
            replace,
            extra={
                "registry_data": {
                    "event": "agent_registered",
                    "agent_type": name,
                    "class": agent_class.__name__,
                    "replaced": replace,
                    "total_registered": len(cls._registry),
                }
            },
        )

    @classmethod
    def get(cls, name: str) -> type[Agent]:
        """Return the agent class registered under *name*.

        Raises
        ------
        AgentConfigError
            If *name* is not found, with a message listing available types.

        """
        collector = get_metrics_collector()
        agent_cls = cls._registry.get(name)
        if agent_cls is None:
            collector.increment(
                "agent_registry_lookups_total",
                labels={"agent_type": name, "status": "not_found"},
            )
            available = ", ".join(sorted(cls._registry)) or "(none)"
            msg = f"Unknown agent type '{name}'. Available types: {available}"
            raise AgentConfigError(msg)
        collector.increment(
            "agent_registry_lookups_total",
            labels={"agent_type": name, "status": "found"},
        )
        return agent_cls

    @classmethod
    def list(cls) -> list[str]:
        """Return a sorted list of all registered agent type names."""
        return sorted(cls._registry.keys())
