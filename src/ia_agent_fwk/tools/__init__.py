"""Tool system public API.

This module re-exports all public types from the tools package.
Imports are lazy to avoid circular import issues with the config system.
"""

from ia_agent_fwk.tools.base import Tool, ToolContext
from ia_agent_fwk.tools.config import ToolPermissionConfig, ToolsConfig
from ia_agent_fwk.tools.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
    ToolValidationError,
)
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
from ia_agent_fwk.tools.registry import ToolRegistry

__all__ = [
    "DefaultToolExecutor",
    "PermissionMode",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolPermissionConfig",
    "ToolPermissionError",
    "ToolPermissionManager",
    "ToolRegistry",
    "ToolTimeoutError",
    "ToolValidationError",
    "ToolsConfig",
]


def __getattr__(name: str) -> object:
    """Lazy import for DefaultToolExecutor to avoid circular imports."""
    if name == "DefaultToolExecutor":
        from ia_agent_fwk.tools.executor import DefaultToolExecutor  # noqa: PLC0415

        return DefaultToolExecutor
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
