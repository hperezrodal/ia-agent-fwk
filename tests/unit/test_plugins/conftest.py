"""Shared fixtures for plugin system tests."""

from __future__ import annotations

import pytest

from ia_agent_fwk.plugins.models import PluginConfig
from ia_agent_fwk.tools.registry import ToolRegistry

from .example_plugin import ExamplePlugin, FailingPlugin, MultiToolPlugin


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def example_plugin_cls():
    return ExamplePlugin


@pytest.fixture
def failing_plugin_cls():
    return FailingPlugin


@pytest.fixture
def multi_tool_plugin_cls():
    return MultiToolPlugin


@pytest.fixture
def example_plugin_config():
    return PluginConfig(name="example", enabled=True, settings={"key": "value"})


@pytest.fixture
def disabled_plugin_config():
    return PluginConfig(name="example", enabled=False)
