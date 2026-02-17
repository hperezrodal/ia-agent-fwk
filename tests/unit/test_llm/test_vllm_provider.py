"""Tests for the vLLM provider (mocked httpx)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from ia_agent_fwk.llm.exceptions import LLMProviderError, LLMTimeoutError
from ia_agent_fwk.llm.models import FinishReason, Message, ToolCall
from ia_agent_fwk.llm.providers.vllm import VLLMProvider


def _chat_response_json(*, content="Hello!", finish_reason="stop"):
    return {
        "id": "cmpl-123",
        "object": "chat.completion",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _completion_response_json(*, text="Hello!", finish_reason="stop"):
    return {
        "id": "cmpl-456",
        "object": "text_completion",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _models_response_json():
    return {
        "object": "list",
        "data": [
            {"id": "meta-llama/Llama-3.1-8B-Instruct", "object": "model"},
        ],
    }


def _chat_response_with_tool_calls():
    return {
        "id": "cmpl-789",
        "object": "chat.completion",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


@pytest.mark.unit
class TestVLLMProvider:
    @pytest.fixture
    def provider(self, mock_vllm_provider_settings):
        return VLLMProvider(
            settings=mock_vllm_provider_settings,
            provider_name="vllm",
        )

    async def test_chat_success(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=_chat_response_json()))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        resp = await provider.chat([Message(role="user", content="Hi")])
        assert resp.message.role == "assistant"
        assert resp.message.content == "Hello!"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.finish_reason == FinishReason.stop

    async def test_chat_with_tool_calls(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=_chat_response_with_tool_calls()))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        resp = await provider.chat([Message(role="user", content="What's the weather?")])
        assert resp.finish_reason == FinishReason.tool_calls
        assert resp.message.tool_calls is not None
        assert len(resp.message.tool_calls) == 1
        assert resp.message.tool_calls[0].name == "get_weather"
        assert resp.message.tool_calls[0].id == "call_1"

    async def test_chat_finish_reason_length(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json=_chat_response_json(finish_reason="length"))
        )
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        resp = await provider.chat([Message(role="user", content="Hi")])
        assert resp.finish_reason == FinishReason.length

    async def test_complete_success(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=_completion_response_json()))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        resp = await provider.complete("Once upon a time")
        assert resp.text == "Hello!"
        assert resp.usage.total_tokens == 15

    async def test_health_check_success(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=_models_response_json()))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        status = await provider.health_check()
        assert status.status == "healthy"

    async def test_health_check_model_not_found(self, mock_vllm_provider_settings):
        response_json = {
            "object": "list",
            "data": [{"id": "some-other-model", "object": "model"}],
        }
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=response_json))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        status = await provider.health_check()
        assert status.status == "healthy"
        assert "not found" in (status.message or "")

    async def test_health_check_connection_refused(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(httpx.ConnectError("Connection refused"))
        )
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        status = await provider.health_check()
        assert status.status == "unhealthy"
        assert "Connection refused" in (status.message or "")

    def test_count_tokens(self, provider):
        count = provider.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count >= 1

    async def test_close(self, provider):
        provider._client = AsyncMock()
        await provider.close()
        provider._client.aclose.assert_awaited_once()

    async def test_chat_timeout(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        with pytest.raises(LLMTimeoutError):
            await provider.chat([Message(role="user", content="hi")])

    async def test_chat_http_error(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(500, text="Internal Server Error"))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        with pytest.raises(LLMProviderError):
            await provider.chat([Message(role="user", content="hi")])

    async def test_complete_timeout(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        with pytest.raises(LLMTimeoutError):
            await provider.complete("Once upon")

    async def test_complete_http_error(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(500, text="Internal Server Error"))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        with pytest.raises(LLMProviderError):
            await provider.complete("Once upon")

    async def test_stream(self, mock_vllm_provider_settings):
        chunk1 = (
            'data: {"id":"cmpl-1","choices":[{"index":0,'
            '"delta":{"content":"Hello"},"finish_reason":null}],"usage":null}'
        )
        chunk2 = (
            'data: {"id":"cmpl-1","choices":[{"index":0,'
            '"delta":{"content":" world"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}'
        )
        lines = [chunk1, chunk2, "data: [DONE]"]
        body = "\n".join(lines)

        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        )
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        chunks = [c async for c in provider.stream([Message(role="user", content="hi")])]

        assert len(chunks) == 2
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[1].finish_reason == FinishReason.stop
        assert chunks[1].usage is not None

    async def test_stream_timeout(self, mock_vllm_provider_settings):
        transport = httpx.MockTransport(lambda _request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")))
        provider = VLLMProvider(settings=mock_vllm_provider_settings, provider_name="vllm")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:8000/v1")

        with pytest.raises(LLMTimeoutError):
            async for _ in provider.stream([Message(role="user", content="hi")]):
                pass

    async def test_to_openai_message_basic(self, provider):
        msg = Message(role="user", content="Hello")
        result = provider._to_openai_message(msg)
        assert result == {"role": "user", "content": "Hello"}

    async def test_to_openai_message_with_tool_calls(self, provider):
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="get_weather", arguments='{"city":"London"}')],
        )
        result = provider._to_openai_message(msg)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][0]["id"] == "1"

    async def test_to_openai_message_with_tool_call_id(self, provider):
        msg = Message(role="tool", content="result", tool_call_id="tc_1")
        result = provider._to_openai_message(msg)
        assert result["tool_call_id"] == "tc_1"

    async def test_default_base_url(self, mock_vllm_provider_settings):
        """When base_url is empty, defaults to localhost:8000/v1."""
        from ia_agent_fwk.config.settings import CircuitBreakerSettings, LLMProviderSettings, RetrySettings

        settings = LLMProviderSettings(
            base_url="",
            default_model="test-model",
            retry=RetrySettings(max_attempts=1, backoff_base=0.01, backoff_max=0.01),
            circuit_breaker=CircuitBreakerSettings(enabled=False),
        )
        provider = VLLMProvider(settings=settings, provider_name="vllm")
        assert "localhost:8000" in str(provider._client.base_url)
