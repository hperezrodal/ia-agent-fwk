"""Streaming module -- SSE and WebSocket support for real-time agent output."""

from __future__ import annotations

from ia_agent_fwk.streaming.exceptions import (
    StreamConnectionError,
    StreamingError,
    StreamTimeoutError,
)
from ia_agent_fwk.streaming.models import AgentStreamEvent, StreamEvent
from ia_agent_fwk.streaming.sse import sse_stream
from ia_agent_fwk.streaming.websocket import WebSocketHandler

__all__ = [
    "AgentStreamEvent",
    "StreamConnectionError",
    "StreamEvent",
    "StreamTimeoutError",
    "StreamingError",
    "WebSocketHandler",
    "sse_stream",
]
