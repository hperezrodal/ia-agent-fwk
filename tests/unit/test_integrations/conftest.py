"""Shared fixtures for integration unit tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentResult
from ia_agent_fwk.agents.registry import AgentRegistry
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.api.app import create_app
from ia_agent_fwk.config.settings import (
    AppSettings,
    AuthSettings,
    EmailIntegrationSettings,
    IntegrationsSettings,
    MemorySettings,
    SlackIntegrationSettings,
    WhatsAppIntegrationSettings,
)
from ia_agent_fwk.integrations.email_channel import EmailIntegration
from ia_agent_fwk.integrations.models import ChannelConfig, IncomingMessage, OutgoingMessage
from ia_agent_fwk.integrations.router import ChannelRouter
from ia_agent_fwk.integrations.slack import SlackIntegration
from ia_agent_fwk.integrations.whatsapp import WhatsAppIntegration
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
# Test agent
# ---------------------------------------------------------------------------


class _TestAgent(Agent):
    """Simple agent for integration tests."""

    @property
    def agent_type(self) -> str:
        return "test"

    async def run(
        self,
        input_text: str,
        conversation_history: list[Message] | None = None,
    ) -> AgentResult:
        return AgentResult(
            output=f"Agent response: {input_text}",
            state=AgentState.COMPLETED,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            iterations=1,
            duration_ms=42.0,
        )


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


class _MockLLMProvider(LLMProvider):
    """Mock LLM provider."""

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
def slack_integration():
    """Slack integration with test credentials."""
    return SlackIntegration(
        bot_token="xoxb-test-token",
        signing_secret="test-signing-secret",
        default_channel="#general",
    )


@pytest.fixture
def email_integration():
    """Email integration with test credentials."""
    return EmailIntegration(
        smtp_host="smtp.test.com",
        smtp_port=587,
        from_address="bot@test.com",
        username="bot@test.com",
        password="test-password",
    )


@pytest.fixture
def whatsapp_integration():
    """WhatsApp integration with test credentials."""
    return WhatsAppIntegration(
        access_token="test-access-token",
        phone_number_id="123456789",
        verify_token="test-verify-token",
    )


@pytest.fixture
def channel_config():
    """Channel config for test agent."""
    return ChannelConfig(
        channel_type="test",
        enabled=True,
        agent_type="test",
    )


@pytest.fixture
def channel_router():
    """Empty channel router."""
    return ChannelRouter()


@pytest.fixture
def sample_incoming_message():
    """Sample incoming message."""
    return IncomingMessage(
        channel="slack",
        sender="U12345",
        content="Hello agent",
        metadata={"channel_id": "C12345"},
        timestamp="1234567890.123456",
    )


@pytest.fixture
def sample_outgoing_message():
    """Sample outgoing message."""
    return OutgoingMessage(
        channel="slack",
        recipient="#general",
        content="Hello human",
    )


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    return mock


@pytest.fixture
def test_settings_integrations(monkeypatch) -> AppSettings:
    """Test AppSettings with integrations enabled."""
    monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
    return AppSettings(
        auth=AuthSettings(enabled=True),
        memory=MemorySettings(default_backend="in_memory"),
        integrations=IntegrationsSettings(
            slack=SlackIntegrationSettings(enabled=True, bot_token="xoxb-test", default_agent="test"),
            email=EmailIntegrationSettings(enabled=True, default_agent="test"),
            whatsapp=WhatsAppIntegrationSettings(
                enabled=True,
                access_token="test-token",
                phone_number_id="123",
                verify_token="test-verify",
                default_agent="test",
            ),
        ),
    )


@pytest.fixture
def test_settings_email_disabled(monkeypatch) -> AppSettings:
    """Test AppSettings with email integration disabled."""
    monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
    return AppSettings(
        auth=AuthSettings(enabled=True),
        memory=MemorySettings(default_backend="in_memory"),
        integrations=IntegrationsSettings(
            slack=SlackIntegrationSettings(enabled=True, bot_token="xoxb-test", default_agent="test"),
            email=EmailIntegrationSettings(enabled=False),
            whatsapp=WhatsAppIntegrationSettings(
                enabled=True,
                access_token="test-token",
                phone_number_id="123",
                verify_token="test-verify",
                default_agent="test",
            ),
        ),
    )


@pytest.fixture
def test_app_email_disabled(test_settings_email_disabled):
    """Test FastAPI app with email integration disabled."""
    if "test" not in AgentRegistry._registry:
        AgentRegistry.register("test", _TestAgent)

    mock_provider = _MockLLMProvider()

    def mock_create(config, llm_settings, **kwargs):
        return _TestAgent(config=config, provider=mock_provider)

    with patch("ia_agent_fwk.api.routes.agents.AgentFactory.create", side_effect=mock_create):
        app = create_app(test_settings_email_disabled)

        app.state.memory_backend = InMemoryBackend()
        app.state.conversation_backend = ConversationMemoryBackend()
        app.state.job_manager = MagicMock()

        yield app

    AgentRegistry._registry.pop("test", None)


@pytest.fixture
async def email_disabled_client(test_app_email_disabled):
    """Async HTTP client for testing email-disabled integration API."""
    transport = httpx.ASGITransport(app=test_app_email_disabled)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_app_integrations(test_settings_integrations):
    """Test FastAPI app with integrations router."""
    if "test" not in AgentRegistry._registry:
        AgentRegistry.register("test", _TestAgent)

    mock_provider = _MockLLMProvider()

    def mock_create(config, llm_settings, **kwargs):
        return _TestAgent(config=config, provider=mock_provider)

    with patch("ia_agent_fwk.api.routes.agents.AgentFactory.create", side_effect=mock_create):
        app = create_app(test_settings_integrations)

        app.state.memory_backend = InMemoryBackend()
        app.state.conversation_backend = ConversationMemoryBackend()
        app.state.job_manager = MagicMock()

        yield app

    AgentRegistry._registry.pop("test", None)


@pytest.fixture
async def integration_client(test_app_integrations):
    """Async HTTP client for testing integration API."""
    transport = httpx.ASGITransport(app=test_app_integrations)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers with valid API key."""
    return {"X-API-Key": "test-key-1"}
