"""Shared fixtures for streaming unit tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig, AgentResult
from ia_agent_fwk.agents.registry import AgentRegistry
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.api.app import create_app
from ia_agent_fwk.config.settings import AppSettings, AuthSettings, MemorySettings
from ia_agent_fwk.llm.base import LLMProvider
from ia_agent_fwk.llm.models import (
    ChatResponse,
    CompletionResponse,
    FinishReason,
    HealthStatus,
    Message,
    StreamChunk,
    TokenUsage,
)
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend

# ---------------------------------------------------------------------------
# Mock streaming agent
# ---------------------------------------------------------------------------


class StreamTestAgent(Agent):
    """Agent that returns a canned response, used for streaming tests."""

    @property
    def agent_type(self) -> str:
        return "stream_test"

    async def run(
        self,
        input_text: str,
        conversation_history: list[Message] | None = None,
        conversation_id: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            output=f"Streamed: {input_text}",
            state=AgentState.COMPLETED,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=15),
            iterations=1,
            duration_ms=10.0,
        )


class SlowTestAgent(Agent):
    """Agent that sleeps briefly, used for heartbeat tests."""

    @property
    def agent_type(self) -> str:
        return "slow_test"

    async def run(
        self,
        input_text: str,
        conversation_history: list[Message] | None = None,
        conversation_id: str | None = None,
    ) -> AgentResult:
        import asyncio

        await asyncio.sleep(0.15)
        return AgentResult(
            output=f"Slow: {input_text}",
            state=AgentState.COMPLETED,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=15),
            iterations=1,
            duration_ms=150.0,
        )


class ErrorTestAgent(Agent):
    """Agent that raises an exception, used for error-path tests."""

    @property
    def agent_type(self) -> str:
        return "error_test"

    async def run(
        self,
        input_text: str,
        conversation_history: list[Message] | None = None,
        conversation_id: str | None = None,
    ) -> AgentResult:
        msg = "Intentional test error"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


class _MockLLMProvider(LLMProvider):
    """Mock LLM provider for streaming tests."""

    def __init__(self) -> None:
        self.provider_name = "mock"

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content="Hello!"),
            usage=TokenUsage(prompt_tokens=5, completion_tokens=15),
            model="mock-model",
            finish_reason=FinishReason.stop,
        )

    async def stream(self, messages: list[Message], **kwargs: Any):
        raise NotImplementedError
        yield StreamChunk()  # type: ignore[misc]  # pragma: no cover

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text) // 4

    async def health_check(self) -> HealthStatus:
        return HealthStatus(status="healthy", message="Mock provider is healthy")

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider():
    return _MockLLMProvider()


@pytest.fixture
def stream_agent(mock_provider):
    config = AgentConfig(
        name="stream-test",
        agent_type="stream_test",
    )
    return StreamTestAgent(config=config, provider=mock_provider)


@pytest.fixture
def slow_agent(mock_provider):
    config = AgentConfig(
        name="slow-test",
        agent_type="slow_test",
    )
    return SlowTestAgent(config=config, provider=mock_provider)


@pytest.fixture
def error_agent(mock_provider):
    config = AgentConfig(
        name="error-test",
        agent_type="error_test",
    )
    return ErrorTestAgent(config=config, provider=mock_provider)


@pytest.fixture
def test_settings(monkeypatch) -> AppSettings:
    """Test AppSettings with auth enabled and known API keys."""
    monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1,test-key-2")
    return AppSettings(
        auth=AuthSettings(enabled=True),
        memory=MemorySettings(default_backend="in_memory"),
    )


@pytest.fixture
def streaming_app(test_settings):
    """Create a test FastAPI application with streaming agents registered."""
    if "stream_test" not in AgentRegistry._registry:
        AgentRegistry.register("stream_test", StreamTestAgent)

    mock_provider = _MockLLMProvider()

    def mock_create(config, llm_settings, **kwargs):  # noqa: ARG001
        if config.agent_type == "stream_test":
            return StreamTestAgent(config=config, provider=mock_provider)
        msg = f"Unknown agent type: {config.agent_type}"
        raise ValueError(msg)

    with (
        patch("ia_agent_fwk.api.routes.streaming.AgentFactory.create", side_effect=mock_create),
        patch("ia_agent_fwk.api.routes.agents.AgentFactory.create", side_effect=mock_create),
    ):
        app = create_app(test_settings)

        # Manually set up app.state (normally done in lifespan)
        app.state.memory_backend = InMemoryBackend()
        app.state.conversation_backend = ConversationMemoryBackend()
        app.state.job_manager = MagicMock()
        app.state.rate_limiter = None
        app.state.audit_logger = MagicMock()

        yield app

    # Cleanup
    AgentRegistry._registry.pop("stream_test", None)


@pytest.fixture
async def streaming_client(streaming_app):
    """Async HTTP client for testing streaming endpoints."""
    transport = httpx.ASGITransport(app=streaming_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers with valid API key."""
    return {"X-API-Key": "test-key-1"}
