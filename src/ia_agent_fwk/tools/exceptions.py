"""Tool system exception hierarchy.

All tool-specific exceptions inherit from ``ToolError``.
``ToolValidationError`` includes structured error details for
field-level validation failures.
"""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """Base exception for all tool system errors."""


class ToolNotFoundError(ToolError):
    """Raised when a tool is not found in the registry."""


class ToolValidationError(ToolError):
    """Raised when tool input or output validation fails.

    Attributes
    ----------
    details:
        Structured error information with field names, expected types,
        actual values, and validation messages.

    """

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.details: list[dict[str, Any]] = details or []


class ToolExecutionError(ToolError):
    """Raised when tool execution fails.

    Attributes
    ----------
    tool_name:
        Name of the tool that failed.

    """

    def __init__(self, message: str, tool_name: str = "") -> None:
        super().__init__(message)
        self.tool_name: str = tool_name


class ToolPermissionError(ToolError):
    """Raised when a tool call is denied by the permission manager."""


class ToolTimeoutError(ToolError):
    """Raised when tool execution exceeds the configured timeout.

    Attributes
    ----------
    timeout:
        The timeout value in seconds that was exceeded.

    """

    def __init__(self, message: str, timeout: float = 0.0) -> None:
        super().__init__(message)
        self.timeout: float = timeout
