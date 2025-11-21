"""Integration test fixtures for the API layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.registry import AgentRegistry
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


class _IntegrationTestAgent(Agent):
    """Test agent for integration tests."""

    @property
    def agent_type(self) -> str:
        return "integration-test"


class _IntegrationMockLLMProvider(LLMProvider):
    """Mock LLM provider for integration tests."""

    def __init__(self) -> None:
        self.provider_name = "mock"

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content="Integration response"),
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
        return HealthStatus(status="healthy")

    async def close(self) -> None:
        pass


@pytest.fixture
def integration_settings(monkeypatch) -> AppSettings:
    """Integration test settings."""
    monkeypatch.setenv("IAFWK_API_KEYS", "integration-key-1")
    return AppSettings(
        auth=AuthSettings(enabled=True),
        memory=MemorySettings(default_backend="in_memory"),
    )


@pytest.fixture
def integration_app(integration_settings):
    """Create integration test app."""
    if "integration-test" not in AgentRegistry._registry:
        AgentRegistry.register("integration-test", _IntegrationTestAgent)

    mock_provider = _IntegrationMockLLMProvider()

    def mock_create(config, llm_settings, **kwargs):  # noqa: ARG001
        return _IntegrationTestAgent(
            config=config,
            provider=mock_provider,
            memory_backend=kwargs.get("memory_backend"),
            conversation_backend=kwargs.get("conversation_backend"),
        )

    with patch("ia_agent_fwk.api.routes.agents.AgentFactory.create", side_effect=mock_create):
        app = create_app(integration_settings)

        # Manually set up app.state (normally done in lifespan)
        app.state.memory_backend = InMemoryBackend()
        app.state.conversation_backend = ConversationMemoryBackend()

        yield app

    AgentRegistry._registry.pop("integration-test", None)


@pytest.fixture
async def integration_client(integration_app):
    """Async HTTP client for integration tests."""
    transport = httpx.ASGITransport(app=integration_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def integration_auth_headers() -> dict[str, str]:
    """Auth headers for integration tests."""
    return {"X-API-Key": "integration-key-1"}
