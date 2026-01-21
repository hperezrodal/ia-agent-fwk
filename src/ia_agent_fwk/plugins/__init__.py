"""Plugin system public API.

This module re-exports all public types from the plugins package.
"""

from __future__ import annotations

from ia_agent_fwk.plugins.base import Plugin
from ia_agent_fwk.plugins.discovery import (
    discover_plugins_from_directory,
    discover_plugins_from_entry_points,
)
from ia_agent_fwk.plugins.exceptions import (
    PluginConfigError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
)
from ia_agent_fwk.plugins.loader import PluginLoader
from ia_agent_fwk.plugins.manager import PluginManager
from ia_agent_fwk.plugins.models import PluginConfig, PluginInfo, PluginManifest

__all__ = [
    "Plugin",
    "PluginConfig",
    "PluginConfigError",
    "PluginError",
    "PluginInfo",
    "PluginLoadError",
    "PluginLoader",
    "PluginManager",
    "PluginManifest",
    "PluginNotFoundError",
    "discover_plugins_from_directory",
    "discover_plugins_from_entry_points",
]
