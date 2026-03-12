"""Shared fixtures for API unit tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentResult
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

# Re-export Message so it's available for type annotations
__all__ = ["Message"]
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend

# ---------------------------------------------------------------------------
# Test agent
# ---------------------------------------------------------------------------


class _TestAgent(Agent):
    """Simple agent that returns a canned response."""

    @property
    def agent_type(self) -> str:
        return "test"

    async def run(
        self,
        input_text: str,
        conversation_history: list[Message] | None = None,
        conversation_id: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            output=f"Echo: {input_text}",
            state=AgentState.COMPLETED,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            iterations=1,
            duration_ms=42.0,
        )


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


class _MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns canned responses."""

    def __init__(self) -> None:
        self.provider_name = "mock"

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content="Hello!"),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
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
def test_settings(monkeypatch) -> AppSettings:
    """Test AppSettings with auth enabled and known API keys."""
    monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1,test-key-2")
    return AppSettings(
        auth=AuthSettings(enabled=True),
        memory=MemorySettings(default_backend="in_memory"),
    )


@pytest.fixture
def test_app(test_settings):
    """Create a test FastAPI application with a registered test agent.

    Manually sets ``app.state`` attributes to avoid depending on the
    lifespan manager (which ``httpx.ASGITransport`` does not trigger).
    """
    # Register test agent
    if "test" not in AgentRegistry._registry:
        AgentRegistry.register("test", _TestAgent)

    mock_provider = _MockLLMProvider()

    def mock_create(config, _llm_settings, **_kwargs):
        return _TestAgent(config=config, provider=mock_provider)

    with patch("ia_agent_fwk.api.routes.agents.AgentFactory.create", side_effect=mock_create):
        app = create_app(test_settings)

        # Manually set up app.state (normally done in lifespan)
        app.state.memory_backend = InMemoryBackend()
        app.state.conversation_backend = ConversationMemoryBackend()
        app.state.job_manager = MagicMock()

        yield app

    # Cleanup registry
    AgentRegistry._registry.pop("test", None)


@pytest.fixture
async def client(test_app):
    """Async HTTP client for testing the API."""
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers with valid API key."""
    return {"X-API-Key": "test-key-1"}
