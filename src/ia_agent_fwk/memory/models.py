"""Pydantic v2 models for the memory subsystem.

These models form the public data contract for memory backends.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Key-value memory models
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    """Internal storage model for a memory entry."""

    model_config = ConfigDict(frozen=True)

    key: str
    value: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # noqa: UP017


class MemoryResult(BaseModel):
    """Search result returned by ``MemoryBackend.search()``."""

    model_config = ConfigDict(frozen=True)

    key: str
    value: Any
    score: float = 0.0
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Conversation memory models
# ---------------------------------------------------------------------------


class ConversationInfo(BaseModel):
    """Summary of a conversation."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    agent_namespace: str
    title: str | None = None
    message_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # noqa: UP017
    last_message_at: datetime | None = None


class ConversationMessage(BaseModel):
    """A single message within a conversation."""

    model_config = ConfigDict(frozen=True)

    id: str
    conversation_id: str
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # noqa: UP017
