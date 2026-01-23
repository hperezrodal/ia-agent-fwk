"""High-level plugin manager.

``PluginManager`` orchestrates plugin discovery, filtering, and loading.
It combines entry-point and directory-based discovery, applies
configuration filters, and delegates to ``PluginLoader``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ia_agent_fwk.plugins.discovery import (
    discover_plugins_from_directory,
    discover_plugins_from_entry_points,
)
from ia_agent_fwk.plugins.loader import PluginLoader

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import PluginsSettings
    from ia_agent_fwk.plugins.base import Plugin
    from ia_agent_fwk.plugins.models import PluginConfig, PluginInfo
    from ia_agent_fwk.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """High-level manager for plugin discovery and loading.

    Combines entry-point and directory-based discovery, applies
    configuration, and delegates loading to ``PluginLoader``.

    Parameters
    ----------
    tool_registry:
        The tool registry to register plugin tools into.
    config:
        Optional plugin system configuration.

    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        config: PluginsSettings | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._config = config

        # Extract plugin configs from settings
        plugin_configs: list[PluginConfig] | None = None
        if config is not None and hasattr(config, "plugins"):
            plugin_configs = config.plugins  # type: ignore[assignment]

        self._loader = PluginLoader(
            tool_registry=tool_registry,
            plugin_configs=plugin_configs,
        )

    async def discover_and_load(self) -> list[PluginInfo]:
        """Discover plugins from entry points and directories, then load them.

        Returns
        -------
        list[PluginInfo]
            Information about all discovered and loaded plugins.

        """
        if self._config is not None and not self._config.enabled:
            logger.info("Plugin system is disabled")
            return []

        discovered: list[type[Plugin]] = []

        # Discover from entry points
        ep_plugins = discover_plugins_from_entry_points()
        discovered.extend(ep_plugins)

        # Discover from configured directories
        if self._config is not None:
            dirs = getattr(self._config, "plugin_dirs", [])
            for dir_path in dirs:
                path = Path(dir_path)
                if path.is_dir():  # noqa: ASYNC240
                    try:
                        dir_plugins = discover_plugins_from_directory(path)
                        discovered.extend(dir_plugins)
                    except Exception:  # noqa: BLE001
                        logger.warning("Failed to scan plugin directory: %s", dir_path)

        if not discovered:
            logger.info("No plugins discovered")
            return []

        logger.info("Discovered %d plugin(s), loading...", len(discovered))
        return await self._loader.load_all(discovered)

    def get_plugin_info(self, name: str) -> PluginInfo | None:
        """Get information about a specific plugin.

        Parameters
        ----------
        name:
            The plugin name to look up.

        Returns
        -------
        PluginInfo | None
            The plugin info if found, ``None`` otherwise.

        """
        for info in self._loader.get_loaded_plugins():
            if info.name == name:
                return info
        return None

    def list_plugins(self) -> list[PluginInfo]:
        """List all known plugins.

        Returns
        -------
        list[PluginInfo]
            List of plugin info snapshots.

        """
        return self._loader.get_loaded_plugins()
