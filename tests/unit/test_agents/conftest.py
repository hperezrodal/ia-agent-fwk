"""Shared fixtures and MockLLMProvider for agent tests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.llm.base import LLMProvider
from ia_agent_fwk.llm.exceptions import LLMProviderError
from ia_agent_fwk.llm.models import (
    ChatResponse,
    CompletionResponse,
    FinishReason,
    HealthStatus,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_COMPLETE_NOT_SUPPORTED = "MockLLMProvider does not support complete()"
_STREAM_NOT_SUPPORTED = "MockLLMProvider does not support stream()"


class MockLLMProvider(LLMProvider):
    """Test double for LLM provider with configurable responses.

    Accepts a sequence of ``ChatResponse`` objects returned one-per-chat-call.
    Supports error injection and latency simulation.
    """

    def __init__(
        self,
        responses: list[ChatResponse],
        *,
        error_on_call: int | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        # Skip super().__init__ -- we don't need settings for the mock
        self.responses = list(responses)
        self._call_index = 0
        self._error_on_call = error_on_call
        self._error = error or LLMProviderError("Mock LLM error")
        self._delay = delay
        self.provider_name = "mock"

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError(_COMPLETE_NOT_SUPPORTED)

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._error_on_call is not None and self._call_index == self._error_on_call:
            self._call_index += 1
            raise self._error

        if self._call_index >= len(self.responses):
            msg = (
                f"MockLLMProvider: no response configured for call index {self._call_index} "
                f"(only {len(self.responses)} responses available)"
            )
            raise IndexError(msg)

        response = self.responses[self._call_index]
        self._call_index += 1
        return response

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError(_STREAM_NOT_SUPPORTED)
        yield  # type: ignore[misc]  # pragma: no cover

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text) // 4

    async def health_check(self) -> HealthStatus:
        return HealthStatus(status="healthy", message="Mock provider is healthy")

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helper to build ChatResponses
# ---------------------------------------------------------------------------


def make_chat_response(  # noqa: PLR0913
    content: str = "Hello!",
    finish_reason: FinishReason = FinishReason.stop,
    tool_calls: list[ToolCall] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    model: str = "mock-model",
) -> ChatResponse:
    """Build a ChatResponse for testing."""
    return ChatResponse(
        message=Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        model=model,
        finish_reason=finish_reason,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """Minimal AgentConfig for testing."""
    return AgentConfig(
        name="test-agent",
        agent_type="test",
        system_prompt="You are a test agent.",
        provider_name="mock",
        max_iterations=10,
        execution_timeout=300,
        max_tokens_per_response=4096,
    )


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """MockLLMProvider with a single stop response."""
    return MockLLMProvider(responses=[make_chat_response()])


@pytest.fixture
def mock_provider_with_tool_call() -> MockLLMProvider:
    """MockLLMProvider: tool call on first call, stop on second."""
    tool_call = ToolCall(id="tc-1", name="search", arguments='{"query": "test"}')
    return MockLLMProvider(
        responses=[
            make_chat_response(
                content="Let me search for that.",
                finish_reason=FinishReason.tool_calls,
                tool_calls=[tool_call],
            ),
            make_chat_response(
                content="Here are the results.",
                finish_reason=FinishReason.stop,
            ),
        ]
    )


@pytest.fixture
def mock_provider_slow() -> MockLLMProvider:
    """MockLLMProvider with a 5-second delay."""
    return MockLLMProvider(
        responses=[make_chat_response()],
        delay=5.0,
    )


@pytest.fixture
def mock_provider_error() -> MockLLMProvider:
    """MockLLMProvider that raises LLMProviderError on the first call."""
    return MockLLMProvider(
        responses=[make_chat_response()],
        error_on_call=0,
    )
