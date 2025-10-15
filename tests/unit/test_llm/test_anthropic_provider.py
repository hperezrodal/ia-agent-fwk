"""Tests for the Anthropic provider (mocked SDK)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.llm.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
)
from ia_agent_fwk.llm.models import FinishReason, Message
from ia_agent_fwk.llm.providers.anthropic import (
    AnthropicProvider,
    _extract_system_and_messages,
)


def _mock_anthropic_response(*, text="Hello!", tool_use=None, stop_reason="end_turn"):
    content = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    if tool_use:
        content.append(
            SimpleNamespace(
                type="tool_use",
                id=tool_use["id"],
                name=tool_use["name"],
                input=tool_use["input"],
            )
        )
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="claude-sonnet-4-20250514",
        stop_reason=stop_reason,
    )


class TestExtractSystemAndMessages:
    def test_single_system_message(self):
        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        system, converted = _extract_system_and_messages(msgs)
        assert system == "You are helpful."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_multiple_system_messages(self):
        msgs = [
            Message(role="system", content="Part 1"),
            Message(role="system", content="Part 2"),
            Message(role="user", content="Hi"),
        ]
        system, converted = _extract_system_and_messages(msgs)
        assert system == "Part 1\n\nPart 2"
        assert len(converted) == 1

    def test_no_system_message(self):
        msgs = [Message(role="user", content="Hi")]
        system, converted = _extract_system_and_messages(msgs)
        assert system is None
        assert len(converted) == 1

    def test_tool_message(self):
        msgs = [
            Message(role="tool", content="result", tool_call_id="tc_1"),
        ]
        _, converted = _extract_system_and_messages(msgs)
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"][0]["type"] == "tool_result"


class TestAnthropicProvider:
    @pytest.fixture
    def provider(self, mock_anthropic_provider_settings):
        return AnthropicProvider(
            settings=mock_anthropic_provider_settings,
            provider_name="anthropic",
        )

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_chat_success(self, mock_cls, mock_anthropic_provider_settings):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        resp = await provider.chat([Message(role="user", content="Hi")])

        assert resp.message.role == "assistant"
        assert resp.message.content == "Hello!"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.finish_reason == FinishReason.stop

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_chat_with_system_message(self, mock_cls, mock_anthropic_provider_settings):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        msgs = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hi"),
        ]
        resp = await provider.chat(msgs)
        assert resp.message.content == "Hello!"

        # Verify system was passed separately.
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "Be helpful"

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_chat_with_tool_use(self, mock_cls, mock_anthropic_provider_settings):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(
                text=None,
                tool_use={"id": "tu_1", "name": "get_weather", "input": {"city": "London"}},
                stop_reason="tool_use",
            )
        )
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        resp = await provider.chat([Message(role="user", content="weather?")])

        assert resp.finish_reason == FinishReason.tool_calls
        assert resp.message.tool_calls is not None
        assert len(resp.message.tool_calls) == 1
        tc = resp.message.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.parse_arguments() == {"city": "London"}

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_complete_delegates_to_chat(self, mock_cls, mock_anthropic_provider_settings):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response(text="completed"))
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        resp = await provider.complete("Once upon a time")
        assert resp.text == "completed"

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_auth_error(self, mock_cls, mock_anthropic_provider_settings):
        import anthropic

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "bad key"}}
        mock_response.headers = {}
        mock_response.text = ""
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(message="bad key", response=mock_response, body=None)
        )
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        with pytest.raises(LLMAuthenticationError):
            await provider.chat([Message(role="user", content="hi")])

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_rate_limit_error(self, mock_cls, mock_anthropic_provider_settings):
        import anthropic

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "rate limited"}}
        mock_response.headers = {}
        mock_response.text = ""
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.RateLimitError(message="rate limited", response=mock_response, body=None)
        )
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        with pytest.raises(LLMRateLimitError):
            await provider.chat([Message(role="user", content="hi")])

    def test_count_tokens_heuristic(self, provider):
        count = provider.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count >= 1

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_health_check_success(self, mock_cls, mock_anthropic_provider_settings):
        mock_client = AsyncMock()
        # F-009: health_check now uses count_tokens instead of messages.create.
        mock_client.messages.count_tokens = AsyncMock(return_value=SimpleNamespace(input_tokens=1))
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        status = await provider.health_check()
        assert status.status == "healthy"

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_health_check_failure(self, mock_cls, mock_anthropic_provider_settings):
        mock_client = AsyncMock()
        mock_client.messages.count_tokens = AsyncMock(side_effect=Exception("connection refused"))
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        status = await provider.health_check()
        assert status.status == "unhealthy"

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_close(self, mock_cls, mock_anthropic_provider_settings):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        await provider.close()
        mock_client.close.assert_awaited_once()

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_stream(self, mock_cls, mock_anthropic_provider_settings):
        """F-013: Test Anthropic streaming via mocked stream context manager."""
        # Build mock events for content_block_delta, message_start, message_delta.
        event_start = SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10),
            ),
        )
        event_delta1 = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(text="Hello"),
        )
        event_delta2 = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(text=" world"),
        )
        event_end = SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(output_tokens=5),
            delta=SimpleNamespace(stop_reason="end_turn"),
        )

        async def _fake_events():
            for e in [event_start, event_delta1, event_delta2, event_end]:
                yield e

        class _FakeStreamCM:
            async def __aenter__(self):
                return _fake_events()

            async def __aexit__(self, *args):
                pass

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=_FakeStreamCM())
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        chunks = []
        async for c in provider.stream([Message(role="user", content="hi")]):
            chunks.append(c)

        # Should have content_block_delta chunks + message_delta chunk.
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[2].finish_reason == FinishReason.stop
        assert chunks[2].usage is not None
        assert chunks[2].usage.prompt_tokens == 10
        assert chunks[2].usage.completion_tokens == 5

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_timeout_error_mapping(self, mock_cls, mock_anthropic_provider_settings):
        """F-014: Test timeout error mapping."""
        import anthropic as anthropic_sdk

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=anthropic_sdk.APITimeoutError(request=MagicMock()))
        mock_cls.return_value = mock_client

        from ia_agent_fwk.llm.exceptions import LLMTimeoutError

        provider = AnthropicProvider(settings=mock_anthropic_provider_settings, provider_name="anthropic")
        with pytest.raises(LLMTimeoutError):
            await provider.chat([Message(role="user", content="hi")])

    @patch("ia_agent_fwk.llm.providers.anthropic.anthropic.AsyncAnthropic")
    async def test_chat_kwargs_preserved_on_retry(self, mock_cls, mock_anthropic_provider_settings):
        """F-003: Ensure caller kwargs survive retry attempts."""
        from ia_agent_fwk.config.settings import CircuitBreakerSettings, RetrySettings

        settings = mock_anthropic_provider_settings.model_copy(
            update={
                "retry": RetrySettings(max_attempts=2, backoff_base=0.01, backoff_max=0.01),
                "circuit_breaker": CircuitBreakerSettings(enabled=False),
            }
        )

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[
                ConnectionError("transient"),
                _mock_anthropic_response(),
            ]
        )
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(settings=settings, provider_name="anthropic")
        resp = await provider.chat(
            [Message(role="user", content="Hi")],
            model="claude-3-opus-20240229",
            temperature=0.1,
            max_tokens=100,
        )
        assert resp.message.content == "Hello!"
        assert mock_client.messages.create.call_count == 2
        second_call = mock_client.messages.create.call_args_list[1]
        assert second_call.kwargs.get("model") == "claude-3-opus-20240229"
        assert second_call.kwargs.get("temperature") == 0.1
        assert second_call.kwargs.get("max_tokens") == 100
