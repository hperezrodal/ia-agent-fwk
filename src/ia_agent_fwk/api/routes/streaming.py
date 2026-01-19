"""Streaming endpoints -- SSE and WebSocket."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Security, WebSocket
from starlette.responses import StreamingResponse

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.factory import AgentFactory
from ia_agent_fwk.agents.registry import AgentRegistry
from ia_agent_fwk.api.dependencies import check_rate_limit, get_settings, require_api_key
from ia_agent_fwk.api.models import AgentRunRequest  # noqa: TC001
from ia_agent_fwk.config.settings import AppSettings  # noqa: TC001
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.streaming.sse import sse_stream
from ia_agent_fwk.streaming.websocket import WebSocketHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["streaming"])


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/agents/{agent_type}/stream",
    dependencies=[Security(require_api_key), Depends(check_rate_limit)],
)
async def stream_agent(
    agent_type: str,
    request_body: AgentRunRequest,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> StreamingResponse:
    """Stream agent execution via Server-Sent Events."""
    collector = get_metrics_collector()
    collector.increment("streaming_api_requests_total", labels={"transport": "sse", "agent_type": agent_type})

    # Validate agent type
    AgentRegistry.get(agent_type)

    # Create agent
    agent_config = AgentConfig(
        name=f"{agent_type}-stream",
        agent_type=agent_type,
        provider_name=settings.llm.default_provider,
    )
    agent = AgentFactory.create(agent_config, settings.llm)

    return StreamingResponse(
        sse_stream(
            agent,
            request_body.prompt,
            conversation_id=request_body.conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
) -> None:
    """WebSocket endpoint for bidirectional agent conversation."""
    collector = get_metrics_collector()
    collector.increment("streaming_api_requests_total", labels={"transport": "websocket"})

    settings: AppSettings = websocket.app.state.settings

    def _agent_factory(agent_type: str) -> object:
        """Create an agent for the given type."""
        AgentRegistry.get(agent_type)
        agent_config = AgentConfig(
            name=f"{agent_type}-ws",
            agent_type=agent_type,
            provider_name=settings.llm.default_provider,
        )
        return AgentFactory.create(agent_config, settings.llm)

    handler = WebSocketHandler(
        settings=settings,
        agent_factory=_agent_factory,
    )
    await handler.handle(websocket)
