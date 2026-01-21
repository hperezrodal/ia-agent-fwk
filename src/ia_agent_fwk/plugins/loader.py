"""Plugin loader for instantiating and registering plugins.

``PluginLoader`` takes discovered plugin classes, instantiates them,
calls lifecycle hooks, and registers their tools in the ``ToolRegistry``
with namespaced names.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ia_agent_fwk.plugins.exceptions import PluginLoadError, PluginNotFoundError
from ia_agent_fwk.plugins.models import PluginConfig, PluginInfo
from ia_agent_fwk.tools.base import Tool, ToolContext

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ia_agent_fwk.plugins.base import Plugin
    from ia_agent_fwk.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class PluginLoader:
    """Loads, manages, and unloads plugin instances.

    Each plugin's tools are registered with namespaced names
    (``plugin_name.tool_name``) to prevent collisions.

    Parameters
    ----------
    tool_registry:
        The tool registry to register plugin tools into.
    plugin_configs:
        Optional per-plugin configuration list.

    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        plugin_configs: list[PluginConfig] | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._plugin_configs: dict[str, PluginConfig] = {cfg.name: cfg for cfg in (plugin_configs or [])}
        self._loaded_plugins: dict[str, Plugin] = {}
        self._plugin_infos: dict[str, PluginInfo] = {}
        self._plugin_tools: dict[str, list[str]] = {}

    def _get_config(self, plugin_name: str) -> PluginConfig | None:
        """Get per-plugin configuration if available."""
        return self._plugin_configs.get(plugin_name)

    def _is_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled via configuration."""
        cfg = self._get_config(plugin_name)
        if cfg is None:
            return True  # Default: enabled
        return cfg.enabled

    async def load_plugin(self, plugin_cls: type[Plugin]) -> PluginInfo:
        """Instantiate a plugin, call on_load(), and register its tools.

        Parameters
        ----------
        plugin_cls:
            The plugin class to instantiate and load.

        Returns
        -------
        PluginInfo
            Information about the loaded plugin.

        Raises
        ------
        PluginLoadError
            If the plugin fails to load.

        """
        try:
            plugin = plugin_cls()
            manifest = plugin.manifest
        except Exception as exc:
            msg = f"Failed to instantiate plugin class '{plugin_cls.__name__}': {exc}"
            raise PluginLoadError(msg) from exc

        plugin_name = manifest.name

        if not self._is_enabled(plugin_name):
            info = PluginInfo(
                name=plugin_name,
                version=manifest.version,
                description=manifest.description,
                enabled=False,
                tools_registered=[],
                load_status="disabled",
            )
            self._plugin_infos[plugin_name] = info
            logger.info("Plugin '%s' is disabled, skipping", plugin_name)
            return info

        try:
            await plugin.on_load()
        except Exception as exc:
            msg = f"Plugin '{plugin_name}' on_load() failed: {exc}"
            raise PluginLoadError(msg, plugin_name=plugin_name) from exc

        # Get and register tools with namespacing
        registered_tool_names: list[str] = []
        try:
            tools = plugin.get_tools()
            for tool in tools:
                namespaced_name = f"{plugin_name}.{tool.name}"
                wrapped = _NamespacedTool(tool, namespaced_name)
                self._tool_registry.register(wrapped, replace=True)
                registered_tool_names.append(namespaced_name)
        except Exception as exc:
            msg = f"Plugin '{plugin_name}' tool registration failed: {exc}"
            raise PluginLoadError(msg, plugin_name=plugin_name) from exc

        self._loaded_plugins[plugin_name] = plugin
        self._plugin_tools[plugin_name] = registered_tool_names

        info = PluginInfo(
            name=plugin_name,
            version=manifest.version,
            description=manifest.description,
            enabled=True,
            tools_registered=registered_tool_names,
            load_status="loaded",
        )
        self._plugin_infos[plugin_name] = info
        logger.info(
            "Plugin '%s' v%s loaded with %d tools",
            plugin_name,
            manifest.version,
            len(registered_tool_names),
        )
        return info

    async def load_all(self, plugin_classes: list[type[Plugin]]) -> list[PluginInfo]:
        """Load multiple plugins.

        Parameters
        ----------
        plugin_classes:
            List of plugin classes to load.

        Returns
        -------
        list[PluginInfo]
            Information about each plugin (loaded, disabled, or error).

        """
        results: list[PluginInfo] = []
        for plugin_cls in plugin_classes:
            try:
                info = await self.load_plugin(plugin_cls)
            except PluginLoadError as exc:
                logger.warning("Failed to load plugin: %s", exc)
                info = PluginInfo(
                    name=plugin_cls.__name__,
                    version="0.0.0",
                    enabled=False,
                    tools_registered=[],
                    load_status="error",
                )
            results.append(info)
        return results

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin, calling on_unload() and unregistering its tools.

        Parameters
        ----------
        plugin_name:
            Name of the plugin to unload.

        Returns
        -------
        bool
            ``True`` if the plugin was successfully unloaded.

        Raises
        ------
        PluginNotFoundError
            If the plugin is not loaded.

        """
        plugin = self._loaded_plugins.get(plugin_name)
        if plugin is None:
            msg = f"Plugin '{plugin_name}' is not loaded"
            raise PluginNotFoundError(msg)

        # Call on_unload lifecycle hook
        try:
            await plugin.on_unload()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plugin '%s' on_unload() raised: %s", plugin_name, exc)

        # Unregister tools
        tool_names = self._plugin_tools.get(plugin_name, [])
        for tool_name in tool_names:
            if self._tool_registry.has(tool_name):
                self._tool_registry.remove(tool_name)

        # Clean up internal state
        del self._loaded_plugins[plugin_name]
        self._plugin_tools.pop(plugin_name, None)

        self._plugin_infos[plugin_name] = PluginInfo(
            name=plugin_name,
            version=plugin.manifest.version,
            description=plugin.manifest.description,
            enabled=False,
            tools_registered=[],
            load_status="unloaded",
        )

        logger.info("Plugin '%s' unloaded", plugin_name)
        return True

    def get_loaded_plugins(self) -> list[PluginInfo]:
        """Return information about all known plugins.

        Returns
        -------
        list[PluginInfo]
            List of plugin info snapshots.

        """
        return list(self._plugin_infos.values())


class _NamespacedTool(Tool):
    """Wrapper tool that delegates to an inner tool with a namespaced name."""

    def __init__(self, inner: Tool, namespaced_name: str) -> None:
        self._inner = inner
        self._namespaced_name = namespaced_name

    @property
    def name(self) -> str:
        return self._namespaced_name

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def input_schema(self) -> type[BaseModel]:
        return self._inner.input_schema

    @property
    def output_schema(self) -> type[BaseModel]:
        return self._inner.output_schema

    @property
    def tags(self) -> list[str]:
        return self._inner.tags

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        return await self._inner.execute(validated_input, context)
