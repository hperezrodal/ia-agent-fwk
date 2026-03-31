"""FastAPI endpoint registration for the conversational agent."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    agent: str = "sales-agent"


class Source(BaseModel):
    document: str
    section: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[Source]
    duration_ms: float


def mount_chat_endpoints(
    app: FastAPI,
    agent: Any,  # ConversationalRAGAgent
    *,
    prefix: str = "",
) -> None:
    """Register all conversation endpoints on the FastAPI app.

    Endpoints:
        POST {prefix}/chat          — full response
        POST {prefix}/chat/stream   — SSE streaming
        GET  {prefix}/sessions/{id} — session history (in-memory)
        DELETE {prefix}/sessions/{id} — clear session
        GET  {prefix}/debug/conversations — list recent (from DB)
        GET  {prefix}/debug/conversations/{id} — full debug view
        POST {prefix}/admin/reload-config — reload config from DB
    """

    @app.post(f"{prefix}/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        result = await agent.handle(
            req.message,
            session_id=req.session_id,
            agent=req.agent,
        )
        return ChatResponse(
            session_id=result.session_id,
            response=result.response,
            sources=[Source(**s) for s in result.sources],
            duration_ms=result.duration_ms,
        )

    @app.post(f"{prefix}/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        async def event_stream() -> Any:
            async for event in agent.handle_stream(
                req.message,
                session_id=req.session_id,
                agent=req.agent,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get(f"{prefix}/sessions/{{session_id}}")
    async def get_session(session_id: str) -> dict[str, Any]:
        history = agent._session.get_history(session_id)
        return {
            "session_id": session_id,
            "messages": history,
            "message_count": len(history),
        }

    @app.delete(f"{prefix}/sessions/{{session_id}}")
    async def clear_session(session_id: str) -> dict[str, Any]:
        agent._session.clear_session(session_id)
        return {"session_id": session_id, "cleared": True}

    @app.get(f"{prefix}/debug/conversations")
    async def list_conversations(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        conversations = await agent._session.list_conversations(limit=limit, offset=offset)
        return {"conversations": conversations, "count": len(conversations)}

    @app.get(f"{prefix}/debug/conversations/{{session_id}}")
    async def debug_conversation(session_id: str) -> dict[str, Any]:
        return await agent._session.get_conversation_debug(session_id)

    @app.post(f"{prefix}/admin/reload-config")
    async def reload_config() -> dict[str, str]:
        await agent.reload_config()
        return {"status": "reloaded"}

    @app.get(f"{prefix}/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}
