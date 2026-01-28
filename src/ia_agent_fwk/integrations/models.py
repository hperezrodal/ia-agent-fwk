"""Channel integration data models.

Frozen Pydantic models for normalising messages across channels.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IncomingMessage(BaseModel):
    """Normalised inbound message from any channel."""

    model_config = ConfigDict(frozen=True)

    channel: str
    sender: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    timestamp: str = ""
    conversation_id: str | None = None


class OutgoingMessage(BaseModel):
    """Normalised outbound message to any channel."""

    model_config = ConfigDict(frozen=True)

    channel: str
    recipient: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    format: str = "text"


class ChannelConfig(BaseModel):
    """Configuration for a registered channel."""

    channel_type: str
    enabled: bool = False
    agent_type: str = ""
    settings: dict[str, str] = Field(default_factory=dict)
