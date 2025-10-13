"""Framework-canonical Pydantic v2 models for LLM request/response normalization.

These models form the public API contract of the LLM module.  All providers
convert vendor-specific payloads to and from these types.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FinishReason(str, Enum):
    """Normalized finish reason for LLM responses."""

    stop = "stop"
    tool_calls = "tool_calls"
    length = "length"
    content_filter = "content_filter"
    error = "error"


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A tool/function call returned by the LLM."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str  # raw JSON string

    def parse_arguments(self) -> dict[str, Any]:
        """Deserialize the ``arguments`` JSON string into a dict."""
        result: dict[str, Any] = json.loads(self.arguments)
        return result


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single chat message in the framework-canonical format."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Token usage & cost
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token consumption for a single LLM call."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode="before")
    @classmethod
    def _auto_compute_total(cls, values: Any) -> Any:
        """Auto-compute ``total_tokens`` when it is missing or zero."""
        if isinstance(values, dict):
            prompt = values.get("prompt_tokens", 0)
            completion = values.get("completion_tokens", 0)
            total = values.get("total_tokens", 0)
            if total == 0 and (prompt or completion):
                values["total_tokens"] = prompt + completion
        return values


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ChatResponse(BaseModel):
    """Normalized chat completion response."""

    model_config = ConfigDict(frozen=True)

    message: Message
    usage: TokenUsage
    model: str
    finish_reason: FinishReason


class CompletionResponse(BaseModel):
    """Normalized text completion response."""

    model_config = ConfigDict(frozen=True)

    text: str
    usage: TokenUsage
    model: str
    finish_reason: FinishReason = FinishReason.stop


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class StreamChunk(BaseModel):
    """A single chunk from a streaming LLM response."""

    model_config = ConfigDict(frozen=True)

    delta: str = ""
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    """Provider health check result."""

    model_config = ConfigDict(frozen=True)

    status: Literal["healthy", "unhealthy"]
    message: str | None = None
    latency_ms: float | None = None
