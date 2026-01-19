"""WebSocket handler for bidirectional agent communication."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.streaming.models import AgentStreamEvent

if TYPE_CHECKING:
    from fastapi import WebSocket

    from ia_agent_fwk.agents.base import Agent
    from ia_agent_fwk.config.settings import AppSettings

logger = logging.getLogger(__name__)

# WebSocket close codes
_WS_CLOSE_NORMAL = 1000
_WS_CLOSE_AUTH_FAILED = 4001
_WS_CLOSE_INVALID_MESSAGE = 4002
_WS_CLOSE_INTERNAL_ERROR = 4003

# Defaults
_DEFAULT_PING_INTERVAL: float = 30.0
_DEFAULT_MAX_CONNECTIONS: int = 100

# Module-level connection counter
_active_connections: int = 0


class WebSocketHandler:
    """Manage a single WebSocket connection for agent interaction.

    Lifecycle:

    1. Accept connection (respects ``max_connections`` limit)
    2. Authenticate (query param ``api_key`` or ``auth`` field in first message)
    3. Receive user prompts, execute agent, send response events
    4. Support multiple turns in one connection
    5. Periodic ping to detect stale connections
    6. Handle disconnection and errors

    Parameters
    ----------
    settings:
        Application settings (used for auth checks).
    agent_factory:
        Callable ``(agent_type) -> Agent`` to create agents on demand.
    ping_interval:
        Seconds between WebSocket ping frames. Set to ``0`` to disable.
    max_connections:
        Maximum concurrent WebSocket connections.

    """

    def __init__(
        self,
        settings: AppSettings,
        agent_factory: Any,
        *,
        ping_interval: float = _DEFAULT_PING_INTERVAL,
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
    ) -> None:
        self._settings = settings
        self._agent_factory = agent_factory
        self._ping_interval = ping_interval
        self._max_connections = max_connections

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle(self, websocket: WebSocket) -> None:
        """Full WebSocket lifecycle handler."""
        global _active_connections  # noqa: PLW0603

        collector = get_metrics_collector()
        t0 = time.monotonic()

        # Enforce connection limit
        if _active_connections >= self._max_connections:
            collector.increment("ws_connections_rejected_total", labels={"reason": "max_connections"})
            logger.warning(
                "WebSocket connection rejected: max connections (%d) reached",
                self._max_connections,
                extra={
                    "streaming_data": {
                        "event": "ws_connection_rejected",
                        "reason": "max_connections",
                        "active_connections": _active_connections,
                        "max_connections": self._max_connections,
                    }
                },
            )
            await websocket.accept()
            error_event = AgentStreamEvent(
                event="error",
                content="Too many connections",
            )
            await websocket.send_text(
                json.dumps(error_event.model_dump(mode="json")),
            )
            await websocket.close(code=_WS_CLOSE_INTERNAL_ERROR)
            return

        await websocket.accept()
        _active_connections += 1
        collector.increment("ws_connections_total")
        collector.observe("ws_active_connections", _active_connections)

        logger.info(
            "WebSocket connection accepted (active=%d/%d)",
            _active_connections,
            self._max_connections,
            extra={
                "streaming_data": {
                    "event": "ws_connection_accepted",
                    "active_connections": _active_connections,
                    "max_connections": self._max_connections,
                }
            },
        )

        ping_task: asyncio.Task[None] | None = None
        try:
            authenticated = await self._authenticate(websocket)
            if not authenticated:
                collector.increment("ws_auth_total", labels={"status": "failure"})
                return
            collector.increment("ws_auth_total", labels={"status": "success"})

            # Start periodic ping to detect stale connections
            if self._ping_interval > 0:
                ping_task = asyncio.create_task(self._ping_loop(websocket))

            await self._message_loop(websocket)

        except Exception:
            logger.exception("WebSocket handler error")
            collector.increment("ws_errors_total")
            try:
                error_event = AgentStreamEvent(
                    event="error",
                    content="Internal server error",
                )
                await websocket.send_text(
                    json.dumps(error_event.model_dump(mode="json")),
                )
                await websocket.close(code=_WS_CLOSE_INTERNAL_ERROR)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to send error to disconnected WebSocket client")
        finally:
            _active_connections -= 1
            duration_ms = (time.monotonic() - t0) * 1000
            collector.observe("ws_active_connections", _active_connections)
            collector.observe("ws_connection_duration_seconds", duration_ms / 1000)
            logger.info(
                "WebSocket connection closed (duration=%.0fms, active=%d)",
                duration_ms,
                _active_connections,
                extra={
                    "streaming_data": {
                        "event": "ws_connection_closed",
                        "duration_ms": round(duration_ms, 1),
                        "active_connections": _active_connections,
                    }
                },
            )
            if ping_task is not None:
                ping_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ping_task

    # ------------------------------------------------------------------
    # Ping / keep-alive
    # ------------------------------------------------------------------

    async def _ping_loop(self, websocket: WebSocket) -> None:
        """Send periodic ping frames to detect stale connections."""
        collector = get_metrics_collector()
        while True:
            await asyncio.sleep(self._ping_interval)
            try:
                await websocket.send_bytes(b"ping")
                collector.increment("ws_pings_total", labels={"status": "success"})
            except Exception:  # noqa: BLE001
                collector.increment("ws_pings_total", labels={"status": "failure"})
                collector.increment("streaming_client_disconnects_total", labels={"transport": "websocket"})
                logger.debug("Ping failed — client likely disconnected")
                return

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def _authenticate(self, websocket: WebSocket) -> bool:
        """Authenticate via query param or first message.

        Returns ``True`` if authenticated, ``False`` if connection was
        closed due to auth failure.
        """
        if not self._settings.auth.enabled:
            return True

        # Try query param first
        api_key = websocket.query_params.get("api_key")
        if api_key and self._validate_api_key(api_key):
            return True

        # If no query param, expect auth in first message
        if api_key is None:
            try:
                first_msg = await websocket.receive_text()
                data = json.loads(first_msg)
                api_key = data.get("api_key")
                if api_key and self._validate_api_key(api_key):
                    return True
            except Exception:  # noqa: BLE001
                logger.debug("WebSocket auth message parsing failed")

        # Auth failed
        error_event = AgentStreamEvent(
            event="error",
            content="Authentication failed",
        )
        await websocket.send_text(
            json.dumps(error_event.model_dump(mode="json")),
        )
        await websocket.close(code=_WS_CLOSE_AUTH_FAILED)
        return False

    def _validate_api_key(self, api_key: str) -> bool:
        """Check API key against configured valid keys."""
        raw_keys = os.environ.get("IAFWK_API_KEYS", "")
        valid_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        return any(hmac.compare_digest(api_key, vk) for vk in valid_keys)

    # ------------------------------------------------------------------
    # Message loop
    # ------------------------------------------------------------------

    async def _message_loop(self, websocket: WebSocket) -> None:
        """Receive prompts and stream agent responses."""
        collector = get_metrics_collector()
        message_count = 0
        while True:
            try:
                raw = await websocket.receive_text()
            except Exception:  # noqa: BLE001
                # Client disconnected
                collector.increment("streaming_client_disconnects_total", labels={"transport": "websocket"})
                collector.observe("ws_messages_per_connection", message_count)
                logger.info(
                    "WebSocket client disconnected after %d messages",
                    message_count,
                    extra={
                        "streaming_data": {
                            "event": "ws_client_disconnected",
                            "messages_exchanged": message_count,
                        }
                    },
                )
                return

            collector.increment("ws_messages_received_total")
            message_count += 1

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                collector.increment("streaming_events_total", labels={"transport": "websocket", "event": "error"})
                collector.increment("ws_invalid_messages_total")
                error_event = AgentStreamEvent(
                    event="error",
                    content="Invalid JSON message",
                )
                await websocket.send_text(
                    json.dumps(error_event.model_dump(mode="json")),
                )
                continue

            prompt = data.get("prompt")
            agent_type = data.get("agent_type", "")

            if not prompt:
                collector.increment("streaming_events_total", labels={"transport": "websocket", "event": "error"})
                collector.increment("ws_invalid_messages_total")
                error_event = AgentStreamEvent(
                    event="error",
                    content="Missing 'prompt' field",
                )
                await websocket.send_text(
                    json.dumps(error_event.model_dump(mode="json")),
                )
                continue

            if not agent_type:
                collector.increment("streaming_events_total", labels={"transport": "websocket", "event": "error"})
                collector.increment("ws_invalid_messages_total")
                error_event = AgentStreamEvent(
                    event="error",
                    content="Missing 'agent_type' field",
                )
                await websocket.send_text(
                    json.dumps(error_event.model_dump(mode="json")),
                )
                continue

            try:
                agent: Agent = self._agent_factory(agent_type)
            except Exception as exc:  # noqa: BLE001
                collector.increment("streaming_events_total", labels={"transport": "websocket", "event": "error"})
                error_event = AgentStreamEvent(
                    event="error",
                    agent_type=agent_type,
                    content=f"Failed to create agent: {exc}",
                )
                await websocket.send_text(
                    json.dumps(error_event.model_dump(mode="json")),
                )
                continue

            await self._run_agent(websocket, agent, prompt, agent_type)

    async def _run_agent(
        self,
        websocket: WebSocket,
        agent: Agent,
        prompt: str,
        agent_type: str,
    ) -> None:
        """Execute agent and send start/complete/error events."""
        collector = get_metrics_collector()
        t0 = time.monotonic()

        collector.increment("ws_agent_executions_total", labels={"agent_type": agent_type})
        collector.increment("streaming_events_total", labels={"transport": "websocket", "event": "start"})

        # Start event
        start_event = AgentStreamEvent(
            event="start",
            agent_type=agent_type,
        )
        await websocket.send_text(
            json.dumps(start_event.model_dump(mode="json")),
        )

        try:
            result = await agent.run(prompt)
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            collector.increment("ws_agent_executions_completed_total", labels={"agent_type": agent_type, "status": "error"})
            collector.increment("streaming_events_total", labels={"transport": "websocket", "event": "error"})
            collector.observe("ws_agent_execution_duration_seconds", duration_ms / 1000, labels={"agent_type": agent_type})
            logger.exception(
                "WebSocket agent error for '%s' (%.0fms)",
                agent_type,
                duration_ms,
                extra={
                    "streaming_data": {
                        "event": "ws_agent_error",
                        "agent_type": agent_type,
                        "duration_ms": round(duration_ms, 1),
                        "error": str(exc),
                    }
                },
            )
            error_event = AgentStreamEvent(
                event="error",
                agent_type=agent_type,
                content=str(exc),
            )
            await websocket.send_text(
                json.dumps(error_event.model_dump(mode="json")),
            )
            return

        # Complete event
        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("ws_agent_executions_completed_total", labels={"agent_type": agent_type, "status": "success"})
        collector.increment("streaming_events_total", labels={"transport": "websocket", "event": "complete"})
        collector.observe("ws_agent_execution_duration_seconds", duration_ms / 1000, labels={"agent_type": agent_type})

        logger.info(
            "WebSocket agent '%s' completed: duration=%.0fms, tokens=%d",
            agent_type,
            duration_ms,
            result.usage.total_tokens,
            extra={
                "streaming_data": {
                    "event": "ws_agent_completed",
                    "agent_type": agent_type,
                    "duration_ms": round(duration_ms, 1),
                    "total_tokens": result.usage.total_tokens,
                    "iterations": result.iterations,
                }
            },
        )

        complete_event = AgentStreamEvent(
            event="complete",
            agent_type=agent_type,
            content=result.output,
            usage={
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            metadata={
                "iterations": result.iterations,
                "duration_ms": result.duration_ms,
            },
        )
        await websocket.send_text(
            json.dumps(complete_event.model_dump(mode="json")),
        )


def get_active_connections() -> int:
    """Return the number of active WebSocket connections."""
    return _active_connections
