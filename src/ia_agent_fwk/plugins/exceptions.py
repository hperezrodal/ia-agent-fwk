"""Plugin system exception hierarchy.

All plugin-specific exceptions inherit from ``PluginError``.
"""

from __future__ import annotations


class PluginError(Exception):
    """Base exception for all plugin system errors."""


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load.

    Attributes
    ----------
    plugin_name:
        Name of the plugin that failed to load.

    """

    def __init__(self, message: str, plugin_name: str = "") -> None:
        super().__init__(message)
        self.plugin_name: str = plugin_name


class PluginConfigError(PluginError):
    """Raised when plugin configuration is invalid."""


class PluginNotFoundError(PluginError):
    """Raised when a plugin is not found."""
