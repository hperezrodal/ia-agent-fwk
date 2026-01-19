"""Pydantic v2 models for streaming events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Event type literals
# ---------------------------------------------------------------------------

StreamEventType = Literal[
    "start",
    "token",
    "tool_call",
    "tool_result",
    "thinking",
    "complete",
    "error",
    "heartbeat",
]


# ---------------------------------------------------------------------------
# Core event models
# ---------------------------------------------------------------------------


class StreamEvent(BaseModel):
    """Low-level streaming event with type, data, and timestamp."""

    model_config = ConfigDict(frozen=True)

    event_type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    )


class AgentStreamEvent(BaseModel):
    """High-level agent streaming event for SSE / WebSocket consumers."""

    model_config = ConfigDict(frozen=True)

    event: StreamEventType
    agent_type: str = ""
    content: str | None = None
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    )
