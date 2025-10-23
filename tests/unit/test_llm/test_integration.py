"""Cross-provider integration tests (all using mocks)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import SecretStr

from ia_agent_fwk.config.settings import (
    CircuitBreakerSettings,
    LLMProviderSettings,
    LLMSettings,
    RetrySettings,
)
from ia_agent_fwk.llm import (
    ChatResponse,
    CircuitBreaker,
    CircuitState,
    CostEstimator,
    FinishReason,
    HealthStatus,
    LLMProvider,
    LLMProviderFactory,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from ia_agent_fwk.llm.exceptions import (
    CircuitOpenError,
    LLMAuthenticationError,
    LLMConfigError,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamError,
    LLMTimeoutError,
)


class TestPublicExports:
    """All public types should be importable from ia_agent_fwk.llm."""

    def test_provider_abc(self):
        assert LLMProvider is not None

    def test_factory(self):
        assert LLMProviderFactory is not None

    def test_models(self):
        assert ChatResponse is not None
        assert Message is not None
        assert TokenUsage is not None
        assert StreamChunk is not None
        assert ToolCall is not None
        assert FinishReason is not None
        assert HealthStatus is not None

    def test_exceptions(self):
        assert LLMProviderError is not None
        assert LLMStreamError is not None
        assert CircuitOpenError is not None
        assert LLMConfigError is not None
        assert LLMAuthenticationError is not None
        assert LLMRateLimitError is not None
        assert LLMTimeoutError is not None

    def test_resilience(self):
        assert CircuitBreaker is not None
        assert CircuitState is not None

    def test_cost(self):
        assert CostEstimator is not None


def _make_provider_settings(*, api_key: str = "test", base_url: str = "") -> LLMProviderSettings:
    return LLMProviderSettings(
        api_key=SecretStr(api_key),
        base_url=base_url,
        default_model="test-model",
        retry=RetrySettings(max_attempts=1),
        circuit_breaker=CircuitBreakerSettings(enabled=False),
    )


class TestFactoryCreatesCorrectProviders:
    def test_openai(self):
        settings = LLMSettings(
            default_provider="openai",
            providers={"openai": _make_provider_settings(api_key="sk-test")},
        )
        provider = LLMProviderFactory.create(settings)
        assert provider.provider_name == "openai"

    def test_ollama(self):
        settings = LLMSettings(
            default_provider="ollama",
            providers={"ollama": _make_provider_settings(base_url="http://localhost:11434")},
        )
        provider = LLMProviderFactory.create(settings)
        assert provider.provider_name == "ollama"

    def test_anthropic(self):
        settings = LLMSettings(
            default_provider="anthropic",
            providers={"anthropic": _make_provider_settings(api_key="sk-ant-test")},
        )
        provider = LLMProviderFactory.create(settings)
        assert provider.provider_name == "anthropic"


class TestCrossProviderResponseStructure:
    """Verify that mocked providers produce structurally valid ChatResponses."""

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_openai_response_structure(self, mock_cls):
        mock_client = AsyncMock()
        choice = SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=None, role="assistant"),
            finish_reason="stop",
        )
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o")
        )
        mock_cls.return_value = mock_client

        settings = LLMSettings(
            default_provider="openai",
            providers={"openai": _make_provider_settings(api_key="sk-test")},
        )
        provider = LLMProviderFactory.create(settings)
        resp = await provider.chat([Message(role="user", content="hi")])
        self._validate_response(resp)

    async def test_ollama_response_structure(self):
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "message": {"role": "assistant", "content": "hi"},
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 3,
                },
            )
        )
        settings = LLMSettings(
            default_provider="ollama",
            providers={"ollama": _make_provider_settings(base_url="http://localhost:11434")},
        )
        provider = LLMProviderFactory.create(settings)
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")  # type: ignore[attr-defined]
        resp = await provider.chat([Message(role="user", content="hi")])
        self._validate_response(resp)

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_anthropic_response_structure(self, mock_cls):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="hi")],
                usage=SimpleNamespace(input_tokens=5, output_tokens=3),
                model="claude-sonnet-4-20250514",
                stop_reason="end_turn",
            )
        )
        mock_cls.return_value = mock_client

        settings = LLMSettings(
            default_provider="anthropic",
            providers={"anthropic": _make_provider_settings(api_key="sk-ant-test")},
        )
        provider = LLMProviderFactory.create(settings)
        resp = await provider.chat([Message(role="user", content="hi")])
        self._validate_response(resp)

    def _validate_response(self, resp: ChatResponse) -> None:
        assert isinstance(resp, ChatResponse)
        assert isinstance(resp.message, Message)
        assert resp.message.role == "assistant"
        assert isinstance(resp.usage, TokenUsage)
        assert resp.usage.total_tokens > 0
        assert isinstance(resp.model, str)
        assert isinstance(resp.finish_reason, FinishReason)
