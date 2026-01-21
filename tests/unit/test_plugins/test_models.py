"""Tests for plugin metadata models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ia_agent_fwk.plugins.models import PluginConfig, PluginInfo, PluginManifest


@pytest.mark.unit
class TestPluginManifest:
    def test_minimal_manifest(self):
        m = PluginManifest(name="test")
        assert m.name == "test"
        assert m.version == "0.1.0"
        assert m.description == ""
        assert m.author == ""
        assert m.tools == []
        assert m.entry_point == ""
        assert m.dependencies == []

    def test_full_manifest(self):
        m = PluginManifest(
            name="my_plugin",
            version="2.0.0",
            description="A test plugin",
            author="Test Author",
            tools=["tool_a", "tool_b"],
            entry_point="my_pkg:MyPlugin",
            dependencies=["requests"],
        )
        assert m.name == "my_plugin"
        assert m.version == "2.0.0"
        assert m.tools == ["tool_a", "tool_b"]
        assert m.dependencies == ["requests"]

    def test_manifest_is_frozen(self):
        m = PluginManifest(name="test")
        with pytest.raises(ValidationError):
            m.name = "changed"  # type: ignore[misc]

    def test_manifest_requires_name(self):
        with pytest.raises(ValidationError):
            PluginManifest()  # type: ignore[call-arg]


@pytest.mark.unit
class TestPluginInfo:
    def test_minimal_info(self):
        info = PluginInfo(name="test")
        assert info.name == "test"
        assert info.version == "0.1.0"
        assert info.enabled is True
        assert info.tools_registered == []
        assert info.load_status == "loaded"

    def test_full_info(self):
        info = PluginInfo(
            name="my_plugin",
            version="1.0.0",
            description="desc",
            enabled=False,
            tools_registered=["my_plugin.tool_a"],
            load_status="error",
        )
        assert info.enabled is False
        assert info.tools_registered == ["my_plugin.tool_a"]
        assert info.load_status == "error"

    def test_info_is_frozen(self):
        info = PluginInfo(name="test")
        with pytest.raises(ValidationError):
            info.name = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestPluginConfig:
    def test_minimal_config(self):
        cfg = PluginConfig(name="test")
        assert cfg.name == "test"
        assert cfg.enabled is True
        assert cfg.settings == {}

    def test_config_with_settings(self):
        cfg = PluginConfig(name="test", enabled=False, settings={"api_key": "secret"})
        assert cfg.enabled is False
        assert cfg.settings["api_key"] == "secret"

    def test_config_requires_name(self):
        with pytest.raises(ValidationError):
            PluginConfig()  # type: ignore[call-arg]

    def test_config_is_mutable(self):
        cfg = PluginConfig(name="test")
        cfg.enabled = False
        assert cfg.enabled is False
