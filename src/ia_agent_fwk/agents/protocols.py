"""ToolExecutor Protocol, ToolResult dataclass, and NoOpToolExecutor stub.

The ``ToolExecutor`` protocol defines the interface the reasoning loop uses
to delegate tool execution.  ``ToolResult`` is the structured return type.
``NoOpToolExecutor`` is the V1 stub that returns an error indicating tools
are not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ia_agent_fwk.llm.models import ToolCall


@dataclass
class ToolResult:
    """Result of a single tool execution."""

    output: str
    tool_call_id: str
    error: str | None = field(default=None)


@runtime_checkable
class ToolExecutor(Protocol):
    """Structural typing protocol for tool execution."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        ...


class NoOpToolExecutor:
    """Stub tool executor that reports tools are not yet implemented."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Return a descriptive error for any tool call."""
        return ToolResult(
            output="",
            error=f"Tool execution is not yet implemented (Epic 4). Tool: {tool_call.name}",
            tool_call_id=tool_call.id,
        )
