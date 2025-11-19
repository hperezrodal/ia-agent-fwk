"""Conversation CRUD endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, Security

from ia_agent_fwk.api.dependencies import check_rate_limit, get_conversation_backend, require_api_key
from ia_agent_fwk.api.models import (
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationInfoResponse,
    ConversationListResponse,
    ConversationMessageResponse,
)
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend  # noqa: TC001
from ia_agent_fwk.memory.exceptions import MemoryRetrieveError
from ia_agent_fwk.memory.models import ConversationInfo  # noqa: TC001
from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["conversations"],
    dependencies=[Security(require_api_key), Depends(check_rate_limit)],
)


def _conv_info_to_response(
    info: ConversationInfo,
) -> ConversationInfoResponse:
    """Convert a ``ConversationInfo`` to a response model."""
    return ConversationInfoResponse(
        conversation_id=info.conversation_id,
        agent_namespace=info.agent_namespace,
        title=info.title,
        message_count=info.message_count,
        created_at=info.created_at.isoformat(),
        last_message_at=info.last_message_at.isoformat() if info.last_message_at else None,
    )


@router.post("/conversations", response_model=ConversationInfoResponse, status_code=201)
async def create_conversation(
    request_body: ConversationCreateRequest,
    conversation_backend: Annotated[ConversationMemoryBackend, Depends(get_conversation_backend)],
) -> ConversationInfoResponse:
    """Create a new conversation."""
    info = await conversation_backend.create_conversation(
        agent_namespace=request_body.agent_type,
        title=request_body.title,
    )
    collector = get_metrics_collector()
    collector.increment("api_conversation_requests_total", labels={"operation": "create"})
    return _conv_info_to_response(info)


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    conversation_backend: Annotated[ConversationMemoryBackend, Depends(get_conversation_backend)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    agent_namespace: Annotated[str | None, Query()] = None,
) -> ConversationListResponse:
    """List conversations with optional filtering and pagination."""
    conversations, total = await conversation_backend.list_conversations(
        agent_namespace=agent_namespace,
        limit=limit,
        offset=offset,
    )
    collector = get_metrics_collector()
    collector.increment("api_conversation_requests_total", labels={"operation": "list"})
    return ConversationListResponse(
        conversations=[_conv_info_to_response(c) for c in conversations],
        total=total,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    conversation_backend: Annotated[ConversationMemoryBackend, Depends(get_conversation_backend)],
) -> ConversationDetailResponse:
    """Get conversation details with messages."""
    collector = get_metrics_collector()
    collector.increment("api_conversation_requests_total", labels={"operation": "get"})

    info = await conversation_backend.get_conversation(conversation_id)
    if info is None:
        msg = f"Conversation '{conversation_id}' not found"
        raise MemoryRetrieveError(msg)

    messages = await conversation_backend.get_messages(conversation_id)

    return ConversationDetailResponse(
        conversation_id=info.conversation_id,
        agent_namespace=info.agent_namespace,
        title=info.title,
        messages=[
            ConversationMessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
                token_count=m.token_count,
                metadata=m.metadata,
                created_at=m.created_at.isoformat(),
            )
            for m in messages
        ],
        message_count=info.message_count,
        created_at=info.created_at.isoformat(),
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    conversation_backend: Annotated[ConversationMemoryBackend, Depends(get_conversation_backend)],
) -> Response:
    """Delete a conversation and all its messages."""
    deleted = await conversation_backend.delete_conversation(conversation_id)
    if not deleted:
        msg = f"Conversation '{conversation_id}' not found"
        raise MemoryRetrieveError(msg)

    collector = get_metrics_collector()
    collector.increment("api_conversation_requests_total", labels={"operation": "delete"})
    return Response(status_code=204)
