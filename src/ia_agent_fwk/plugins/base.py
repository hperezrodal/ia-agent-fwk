"""Plugin abstract base class.

The ``Plugin`` ABC defines the interface all plugins must implement.
Plugins provide a manifest describing their identity and a ``get_tools()``
method that returns tool instances to register in the framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ia_agent_fwk.plugins.models import PluginManifest
    from ia_agent_fwk.tools.base import Tool


class Plugin(ABC):
    """Abstract base class for all plugins.

    Subclasses must implement ``manifest`` and ``get_tools()``.
    Lifecycle hooks ``on_load()`` and ``on_unload()`` are optional
    and default to no-ops.
    """

    @property
    @abstractmethod
    def manifest(self) -> PluginManifest:
        """Return the plugin manifest describing this plugin."""
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return tool instances to register in the framework.

        Returns
        -------
        list[Tool]
            The tool instances provided by this plugin.

        """
        ...

    async def on_load(self) -> None:  # noqa: B027
        """Lifecycle hook called after the plugin is loaded.

        Override to perform setup such as establishing connections
        or warming caches. Default implementation is a no-op.
        """

    async def on_unload(self) -> None:  # noqa: B027
        """Lifecycle hook called before the plugin is unloaded.

        Override to perform cleanup such as closing connections
        or flushing buffers. Default implementation is a no-op.
        """
