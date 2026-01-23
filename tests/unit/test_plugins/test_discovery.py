"""Tests for plugin discovery mechanisms."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ia_agent_fwk.plugins.base import Plugin
from ia_agent_fwk.plugins.discovery import (
    discover_plugins_from_directory,
    discover_plugins_from_entry_points,
)
from ia_agent_fwk.plugins.exceptions import PluginLoadError


@pytest.mark.unit
class TestDiscoverFromEntryPoints:
    def test_discover_from_entry_points_finds_plugins(self, example_plugin_cls):
        mock_ep = MagicMock()
        mock_ep.name = "example"
        mock_ep.load.return_value = example_plugin_cls

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("ia_agent_fwk.plugins.discovery.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_plugins_from_entry_points()

        assert len(result) == 1
        assert result[0] is example_plugin_cls

    def test_discover_from_entry_points_skips_non_plugin(self):
        mock_ep = MagicMock()
        mock_ep.name = "not_a_plugin"
        mock_ep.load.return_value = str  # Not a Plugin subclass

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("ia_agent_fwk.plugins.discovery.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_plugins_from_entry_points()

        assert len(result) == 0

    def test_discover_from_entry_points_handles_load_error(self):
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("module not found")

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("ia_agent_fwk.plugins.discovery.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_plugins_from_entry_points()

        assert len(result) == 0

    def test_discover_no_plugins_found(self):
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch("ia_agent_fwk.plugins.discovery.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_plugins_from_entry_points()

        assert len(result) == 0

    def test_discover_uses_custom_group(self, example_plugin_cls):
        mock_ep = MagicMock()
        mock_ep.name = "example"
        mock_ep.load.return_value = example_plugin_cls

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("ia_agent_fwk.plugins.discovery.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_plugins_from_entry_points(group="custom.plugins")

        mock_eps.select.assert_called_once_with(group="custom.plugins")
        assert len(result) == 1

    def test_discover_skips_abstract_plugin_class(self):
        """The Plugin base class itself should not be discovered."""
        mock_ep = MagicMock()
        mock_ep.name = "base"
        mock_ep.load.return_value = Plugin

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("ia_agent_fwk.plugins.discovery.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_plugins_from_entry_points()

        assert len(result) == 0


@pytest.mark.unit
class TestDiscoverFromDirectory:
    def test_discover_from_directory_nonexistent(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(PluginLoadError, match="does not exist"):
            discover_plugins_from_directory(nonexistent)

    def test_discover_from_directory_finds_plugin(self, tmp_path):
        plugin_file = tmp_path / "my_plugin.py"
        plugin_file.write_text(
            """\
from __future__ import annotations
from ia_agent_fwk.plugins.base import Plugin
from ia_agent_fwk.plugins.models import PluginManifest
from ia_agent_fwk.tools.base import Tool


class DirPlugin(Plugin):
    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(name="dir_plugin")

    def get_tools(self) -> list[Tool]:
        return []
"""
        )

        result = discover_plugins_from_directory(tmp_path)
        assert len(result) == 1
        assert result[0].__name__ == "DirPlugin"

    def test_discover_from_directory_skips_init(self, tmp_path):
        init_file = tmp_path / "__init__.py"
        init_file.write_text("# empty init\n")

        result = discover_plugins_from_directory(tmp_path)
        assert len(result) == 0

    def test_discover_from_directory_skips_private_files(self, tmp_path):
        private_file = tmp_path / "_private.py"
        private_file.write_text("x = 1\n")

        result = discover_plugins_from_directory(tmp_path)
        assert len(result) == 0

    def test_discover_from_directory_handles_import_error(self, tmp_path):
        bad_file = tmp_path / "bad_plugin.py"
        bad_file.write_text("import nonexistent_module_xyz\n")

        result = discover_plugins_from_directory(tmp_path)
        assert len(result) == 0

    def test_discover_from_directory_empty(self, tmp_path):
        result = discover_plugins_from_directory(tmp_path)
        assert len(result) == 0
