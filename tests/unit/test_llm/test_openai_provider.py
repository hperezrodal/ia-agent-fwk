"""Tests for the OpenAI provider (mocked SDK)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.llm.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
)
from ia_agent_fwk.llm.models import FinishReason, Message
from ia_agent_fwk.llm.providers.openai import OpenAIProvider


def _mock_chat_response(*, content="Hello!", tool_calls=None, finish_reason="stop"):
    """Build a mock that mirrors the openai SDK ChatCompletion structure."""
    choice = SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
            role="assistant",
        ),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o")


def _mock_tool_call():
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_weather", arguments='{"city":"London"}'),
        type="function",
    )


class TestOpenAIProvider:
    @pytest.fixture
    def provider(self, mock_openai_provider_settings):
        return OpenAIProvider(
            settings=mock_openai_provider_settings,
            provider_name="openai",
        )

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_chat_success(self, mock_cls, mock_openai_provider_settings):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_chat_response())
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        msgs = [Message(role="user", content="Hi")]
        resp = await provider.chat(msgs)

        assert resp.message.role == "assistant"
        assert resp.message.content == "Hello!"
        assert resp.usage.total_tokens == 15
        assert resp.finish_reason == FinishReason.stop

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_chat_with_tool_calls(self, mock_cls, mock_openai_provider_settings):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(
                content=None,
                tool_calls=[_mock_tool_call()],
                finish_reason="tool_calls",
            )
        )
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        resp = await provider.chat([Message(role="user", content="weather?")])

        assert resp.finish_reason == FinishReason.tool_calls
        assert resp.message.tool_calls is not None
        assert len(resp.message.tool_calls) == 1
        tc = resp.message.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.parse_arguments() == {"city": "London"}

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_complete_delegates_to_chat(self, mock_cls, mock_openai_provider_settings):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_chat_response(content="completed"))
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        resp = await provider.complete("Once upon a time")
        assert resp.text == "completed"

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_auth_error(self, mock_cls, mock_openai_provider_settings):
        import openai

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "bad key"}}
        mock_response.headers = {}
        mock_response.text = ""
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(message="bad key", response=mock_response, body=None)
        )
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        with pytest.raises(LLMAuthenticationError):
            await provider.chat([Message(role="user", content="hi")])

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_rate_limit_error(self, mock_cls, mock_openai_provider_settings):
        import openai

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "rate limited"}}
        mock_response.headers = {}
        mock_response.text = ""
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(message="rate limited", response=mock_response, body=None)
        )
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        with pytest.raises(LLMRateLimitError):
            await provider.chat([Message(role="user", content="hi")])

    def test_count_tokens(self, provider):
        count = provider.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_unknown_model(self, provider):
        count = provider.count_tokens("Hello", model="unknown-model-xyz")
        assert isinstance(count, int)
        assert count > 0

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_health_check_success(self, mock_cls, mock_openai_provider_settings):
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=[])
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        status = await provider.health_check()
        assert status.status == "healthy"
        assert status.latency_ms is not None
        assert status.latency_ms >= 0

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_health_check_failure(self, mock_cls, mock_openai_provider_settings):
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("connection refused"))
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        status = await provider.health_check()
        assert status.status == "unhealthy"
        assert "connection refused" in (status.message or "")

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_close(self, mock_cls, mock_openai_provider_settings):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        await provider.close()
        mock_client.close.assert_awaited_once()

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_stream(self, mock_cls, mock_openai_provider_settings):
        # Build async iterator of mock chunks.
        chunk1 = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"), finish_reason=None)],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"), finish_reason="stop")],
            usage=None,
        )
        chunk3 = SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )

        async def _fake_stream():
            for c in [chunk1, chunk2, chunk3]:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_fake_stream())
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=mock_openai_provider_settings, provider_name="openai")
        chunks = []
        async for c in provider.stream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[1].finish_reason == FinishReason.stop
        assert chunks[2].usage is not None
        assert chunks[2].usage.total_tokens == 7

    @patch("ia_agent_fwk.llm.providers.openai.openai.AsyncOpenAI")
    async def test_chat_kwargs_preserved_on_retry(self, mock_cls, mock_openai_provider_settings):
        """F-002: Ensure caller kwargs survive retry attempts."""
        from ia_agent_fwk.config.settings import CircuitBreakerSettings, RetrySettings

        settings = mock_openai_provider_settings.model_copy(
            update={
                "retry": RetrySettings(max_attempts=2, backoff_base=0.01, backoff_max=0.01),
                "circuit_breaker": CircuitBreakerSettings(enabled=False),
            }
        )

        mock_client = AsyncMock()
        # First call fails (retryable), second succeeds.
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                ConnectionError("transient"),
                _mock_chat_response(),
            ]
        )
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(settings=settings, provider_name="openai")
        resp = await provider.chat(
            [Message(role="user", content="Hi")],
            model="gpt-4",
            temperature=0.1,
            max_tokens=100,
        )
        assert resp.message.content == "Hello!"
        # Both calls should have received the same kwargs.
        assert mock_client.chat.completions.create.call_count == 2
        second_call = mock_client.chat.completions.create.call_args_list[1]
        assert second_call.kwargs.get("model") == "gpt-4"
        assert second_call.kwargs.get("temperature") == 0.1
        assert second_call.kwargs.get("max_tokens") == 100
