r"""Server-Sent Events (SSE) streaming utilities.

Provides an async generator that runs an agent and yields SSE-formatted
events: ``start``, ``complete``, and ``error``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.streaming.models import AgentStreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ia_agent_fwk.agents.base import Agent

logger = logging.getLogger(__name__)


def _format_sse(event: AgentStreamEvent) -> str:
    """Format an ``AgentStreamEvent`` as an SSE ``data:`` line."""
    payload = event.model_dump(mode="json")
    return f"data: {json.dumps(payload)}\n\n"


_DEFAULT_HEARTBEAT_INTERVAL: float = 15.0


async def sse_stream(
    agent: Agent,
    prompt: str,
    *,
    conversation_id: str | None = None,
    heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL,
) -> AsyncIterator[str]:
    r"""Stream agent execution as SSE events.

    Yields SSE-formatted strings (``data: {json}\n\n``).

    Events emitted:

    - ``start``  -- agent begins execution
    - ``heartbeat`` -- periodic keep-alive during long executions
    - ``complete`` -- agent finished successfully (includes output & usage)
    - ``error`` -- agent execution failed

    Parameters
    ----------
    agent:
        Agent instance to execute.
    prompt:
        User input text.
    conversation_id:
        Optional conversation identifier (passed in metadata).
    heartbeat_interval:
        Seconds between heartbeat events while the agent runs.
        Set to ``0`` to disable heartbeats.  Default: 15s.

    """
    collector = get_metrics_collector()
    agent_type = agent.agent_type
    t0 = time.monotonic()
    heartbeat_count = 0

    collector.increment("sse_streams_total", labels={"agent_type": agent_type})
    collector.increment("streaming_events_total", labels={"transport": "sse", "event": "start"})

    # --- start event ---
    start_event = AgentStreamEvent(
        event="start",
        agent_type=agent_type,
        metadata={"conversation_id": conversation_id} if conversation_id else {},
    )
    yield _format_sse(start_event)

    # Run agent with periodic heartbeats to keep the connection alive
    agent_task = asyncio.ensure_future(agent.run(prompt))
    try:
        while not agent_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(agent_task),
                    timeout=heartbeat_interval if heartbeat_interval > 0 else None,
                )
            except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                # Agent still running — emit heartbeat
                heartbeat_count += 1
                collector.increment("streaming_events_total", labels={"transport": "sse", "event": "heartbeat"})
                heartbeat_event = AgentStreamEvent(
                    event="heartbeat",
                    agent_type=agent_type,
                )
                yield _format_sse(heartbeat_event)

        result = agent_task.result()

    except asyncio.CancelledError:
        agent_task.cancel()
        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("sse_streams_completed_total", labels={"agent_type": agent_type, "status": "cancelled"})
        collector.increment("streaming_events_total", labels={"transport": "sse", "event": "error"})
        collector.increment("streaming_client_disconnects_total", labels={"transport": "sse"})
        collector.observe("sse_stream_duration_seconds", duration_ms / 1000, labels={"agent_type": agent_type})
        logger.info(
            "SSE stream cancelled by client for agent '%s' (%.0fms, %d heartbeats)",
            agent_type,
            duration_ms,
            heartbeat_count,
            extra={
                "streaming_data": {
                    "event": "sse_stream_cancelled",
                    "agent_type": agent_type,
                    "duration_ms": round(duration_ms, 1),
                    "heartbeat_count": heartbeat_count,
                }
            },
        )
        error_event = AgentStreamEvent(
            event="error",
            agent_type=agent_type,
            content="Stream cancelled by client",
        )
        yield _format_sse(error_event)
        return
    except Exception as exc:
        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("sse_streams_completed_total", labels={"agent_type": agent_type, "status": "error"})
        collector.increment("streaming_events_total", labels={"transport": "sse", "event": "error"})
        collector.observe("sse_stream_duration_seconds", duration_ms / 1000, labels={"agent_type": agent_type})
        logger.exception(
            "SSE stream error for agent '%s' (%.0fms)",
            agent_type,
            duration_ms,
            extra={
                "streaming_data": {
                    "event": "sse_stream_error",
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
        yield _format_sse(error_event)
        return

    # --- complete event ---
    duration_ms = (time.monotonic() - t0) * 1000
    collector.increment("sse_streams_completed_total", labels={"agent_type": agent_type, "status": "success"})
    collector.increment("streaming_events_total", labels={"transport": "sse", "event": "complete"})
    collector.observe("sse_stream_duration_seconds", duration_ms / 1000, labels={"agent_type": agent_type})
    collector.observe("sse_heartbeats_per_stream", heartbeat_count)

    logger.info(
        "SSE stream completed for agent '%s': duration=%.0fms, heartbeats=%d, tokens=%d",
        agent_type,
        duration_ms,
        heartbeat_count,
        result.usage.total_tokens,
        extra={
            "streaming_data": {
                "event": "sse_stream_completed",
                "agent_type": agent_type,
                "duration_ms": round(duration_ms, 1),
                "heartbeat_count": heartbeat_count,
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
            **({"conversation_id": conversation_id} if conversation_id else {}),
        },
    )
    yield _format_sse(complete_event)
