"""Pydantic v2 request/response models for the API layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AgentRunRequest(BaseModel):
    """Request body for ``POST /api/v1/agents/{agent_type}/run``."""

    prompt: str = Field(..., min_length=1, max_length=100_000, description="User input text")
    conversation_id: str | None = Field(default=None, description="Existing conversation to continue")
    options: dict[str, Any] | None = Field(default=None, description="Override options")


class ConversationCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/conversations``."""

    agent_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Agent type to associate",
    )
    title: str | None = Field(default=None, description="Conversation title")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class TokenUsageResponse(BaseModel):
    """Token usage breakdown."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentRunResponse(BaseModel):
    """Response body for agent execution."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    output: str
    usage: TokenUsageResponse
    iterations: int
    duration_ms: float
    agent_type: str


class ErrorDetail(BaseModel):
    """Structured error detail within the error envelope."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    detail: list[dict[str, Any]] | None = None
    request_id: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    model_config = ConfigDict(frozen=True)

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    model_config = ConfigDict(frozen=True)

    status: str


class ReadinessResponse(BaseModel):
    """Response for ``GET /health/ready``."""

    model_config = ConfigDict(frozen=True)

    status: str
    checks: dict[str, str]


class ConversationInfoResponse(BaseModel):
    """Conversation summary for list responses."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    agent_namespace: str
    title: str | None = None
    message_count: int = 0
    created_at: str
    last_message_at: str | None = None


class ConversationListResponse(BaseModel):
    """Response for ``GET /api/v1/conversations``."""

    model_config = ConfigDict(frozen=True)

    conversations: list[ConversationInfoResponse]
    total: int


class ConversationMessageResponse(BaseModel):
    """Message within a conversation detail response."""

    model_config = ConfigDict(frozen=True)

    id: str
    conversation_id: str
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ConversationDetailResponse(BaseModel):
    """Response for ``GET /api/v1/conversations/{conversation_id}``."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    agent_namespace: str
    title: str | None = None
    messages: list[ConversationMessageResponse]
    message_count: int
    created_at: str
