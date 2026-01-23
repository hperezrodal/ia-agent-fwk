"""Tests for plugin manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ia_agent_fwk.plugins.manager import PluginManager


@pytest.mark.unit
class TestPluginManager:
    async def test_discover_and_load(self, tool_registry, example_plugin_cls):
        config = MagicMock()
        config.enabled = True
        config.plugin_dirs = []
        config.plugins = []

        manager = PluginManager(tool_registry=tool_registry, config=config)

        with patch(
            "ia_agent_fwk.plugins.manager.discover_plugins_from_entry_points",
            return_value=[example_plugin_cls],
        ):
            results = await manager.discover_and_load()

        assert len(results) == 1
        assert results[0].name == "example"
        assert results[0].load_status == "loaded"

    async def test_discover_and_load_disabled(self, tool_registry):
        config = MagicMock()
        config.enabled = False

        manager = PluginManager(tool_registry=tool_registry, config=config)
        results = await manager.discover_and_load()
        assert results == []

    async def test_discover_and_load_no_plugins(self, tool_registry):
        config = MagicMock()
        config.enabled = True
        config.plugin_dirs = []
        config.plugins = []

        manager = PluginManager(tool_registry=tool_registry, config=config)

        with patch(
            "ia_agent_fwk.plugins.manager.discover_plugins_from_entry_points",
            return_value=[],
        ):
            results = await manager.discover_and_load()

        assert results == []

    async def test_list_plugins(self, tool_registry, example_plugin_cls):
        manager = PluginManager(tool_registry=tool_registry)

        with patch(
            "ia_agent_fwk.plugins.manager.discover_plugins_from_entry_points",
            return_value=[example_plugin_cls],
        ):
            await manager.discover_and_load()

        plugins = manager.list_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "example"

    async def test_get_plugin_info(self, tool_registry, example_plugin_cls):
        manager = PluginManager(tool_registry=tool_registry)

        with patch(
            "ia_agent_fwk.plugins.manager.discover_plugins_from_entry_points",
            return_value=[example_plugin_cls],
        ):
            await manager.discover_and_load()

        info = manager.get_plugin_info("example")
        assert info is not None
        assert info.name == "example"
        assert info.load_status == "loaded"

    async def test_get_plugin_info_not_found(self, tool_registry):
        manager = PluginManager(tool_registry=tool_registry)
        info = manager.get_plugin_info("nonexistent")
        assert info is None

    async def test_discover_from_directories(self, tool_registry, example_plugin_cls, tmp_path):
        config = MagicMock()
        config.enabled = True
        config.plugin_dirs = [str(tmp_path)]
        config.plugins = []

        manager = PluginManager(tool_registry=tool_registry, config=config)

        with (
            patch(
                "ia_agent_fwk.plugins.manager.discover_plugins_from_entry_points",
                return_value=[],
            ),
            patch(
                "ia_agent_fwk.plugins.manager.discover_plugins_from_directory",
                return_value=[example_plugin_cls],
            ),
        ):
            results = await manager.discover_and_load()

        assert len(results) == 1
        assert results[0].name == "example"

    async def test_manager_without_config(self, tool_registry, example_plugin_cls):
        manager = PluginManager(tool_registry=tool_registry)

        with patch(
            "ia_agent_fwk.plugins.manager.discover_plugins_from_entry_points",
            return_value=[example_plugin_cls],
        ):
            results = await manager.discover_and_load()

        assert len(results) == 1
