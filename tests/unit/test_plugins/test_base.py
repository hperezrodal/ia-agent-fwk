"""Tests for Plugin ABC."""

from __future__ import annotations

import pytest

from ia_agent_fwk.plugins.base import Plugin
from ia_agent_fwk.plugins.models import PluginManifest
from ia_agent_fwk.tools.base import Tool


@pytest.mark.unit
class TestPluginABC:
    def test_cannot_instantiate_abstract_plugin(self):
        with pytest.raises(TypeError):
            Plugin()  # type: ignore[abstract]

    def test_concrete_plugin_can_be_instantiated(self, example_plugin_cls):
        plugin = example_plugin_cls()
        assert isinstance(plugin, Plugin)

    def test_manifest_returns_plugin_manifest(self, example_plugin_cls):
        plugin = example_plugin_cls()
        manifest = plugin.manifest
        assert isinstance(manifest, PluginManifest)
        assert manifest.name == "example"
        assert manifest.version == "1.0.0"

    def test_get_tools_returns_tool_list(self, example_plugin_cls):
        plugin = example_plugin_cls()
        tools = plugin.get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1
        assert isinstance(tools[0], Tool)
        assert tools[0].name == "echo"

    async def test_on_load_is_called(self, example_plugin_cls):
        plugin = example_plugin_cls()
        assert not plugin._loaded
        await plugin.on_load()
        assert plugin._loaded

    async def test_on_unload_is_called(self, example_plugin_cls):
        plugin = example_plugin_cls()
        assert not plugin._unloaded
        await plugin.on_unload()
        assert plugin._unloaded

    async def test_default_on_load_is_noop(self):
        """Verify that a minimal plugin with default on_load doesn't fail."""

        class MinimalPlugin(Plugin):
            @property
            def manifest(self) -> PluginManifest:
                return PluginManifest(name="minimal")

            def get_tools(self) -> list[Tool]:
                return []

        plugin = MinimalPlugin()
        # Should not raise
        await plugin.on_load()
        await plugin.on_unload()
