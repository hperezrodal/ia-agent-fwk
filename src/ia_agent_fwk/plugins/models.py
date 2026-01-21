"""Plugin metadata models.

``PluginManifest`` describes a plugin's identity and capabilities.
``PluginInfo`` is a read-only snapshot of a loaded plugin's state.
``PluginConfig`` holds per-plugin user configuration.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PluginManifest(BaseModel):
    """Immutable manifest describing a plugin.

    Attributes
    ----------
    name:
        Unique plugin identifier.
    version:
        Semantic version string.
    description:
        Human-readable description.
    author:
        Plugin author name.
    tools:
        List of tool names the plugin provides.
    entry_point:
        Dotted path to the plugin class.
    dependencies:
        List of required Python packages.

    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    tools: list[str] = Field(default_factory=list)
    entry_point: str = ""
    dependencies: list[str] = Field(default_factory=list)


class PluginInfo(BaseModel):
    """Read-only snapshot of a loaded plugin's state.

    Attributes
    ----------
    name:
        Plugin name.
    version:
        Plugin version.
    description:
        Plugin description.
    enabled:
        Whether the plugin is enabled.
    tools_registered:
        List of namespaced tool names registered by this plugin.
    load_status:
        Current status: ``loaded``, ``unloaded``, or ``error``.

    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "0.1.0"
    description: str = ""
    enabled: bool = True
    tools_registered: list[str] = Field(default_factory=list)
    load_status: str = "loaded"


class PluginConfig(BaseModel):
    """Per-plugin user configuration.

    Attributes
    ----------
    name:
        Plugin name (must match the plugin's manifest name).
    enabled:
        Whether to load this plugin.
    settings:
        Arbitrary key-value settings passed to the plugin.

    """

    name: str
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)
