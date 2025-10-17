"""Tests for the Ollama provider (mocked httpx)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from ia_agent_fwk.llm.exceptions import LLMProviderError, LLMTimeoutError
from ia_agent_fwk.llm.models import FinishReason, Message
from ia_agent_fwk.llm.providers.ollama import OllamaProvider


def _chat_response_json(*, content="Hello!", done=True):
    return {
        "model": "llama3.1",
        "message": {"role": "assistant", "content": content},
        "done": done,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


def _generate_response_json(*, response="Hello!", done=True):
    return {
        "model": "llama3.1",
        "response": response,
        "done": done,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


def _tags_response_json():
    return {
        "models": [
            {"name": "llama3.1:latest", "size": 123456},
        ]
    }


class TestOllamaProvider:
    @pytest.fixture
    def provider(self, mock_ollama_provider_settings):
        return OllamaProvider(
            settings=mock_ollama_provider_settings,
            provider_name="ollama",
        )

    async def test_chat_success(self, mock_ollama_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=_chat_response_json()))
        provider = OllamaProvider(settings=mock_ollama_provider_settings, provider_name="ollama")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")

        resp = await provider.chat([Message(role="user", content="Hi")])
        assert resp.message.role == "assistant"
        assert resp.message.content == "Hello!"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.finish_reason == FinishReason.stop

    async def test_complete_success(self, mock_ollama_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=_generate_response_json()))
        provider = OllamaProvider(settings=mock_ollama_provider_settings, provider_name="ollama")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")

        resp = await provider.complete("Once upon a time")
        assert resp.text == "Hello!"
        assert resp.usage.total_tokens == 15

    async def test_health_check_success(self, mock_ollama_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=_tags_response_json()))
        provider = OllamaProvider(settings=mock_ollama_provider_settings, provider_name="ollama")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")

        status = await provider.health_check()
        assert status.status == "healthy"

    async def test_health_check_connection_refused(self, mock_ollama_provider_settings):
        transport = httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(httpx.ConnectError("Connection refused"))
        )
        provider = OllamaProvider(settings=mock_ollama_provider_settings, provider_name="ollama")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")

        status = await provider.health_check()
        assert status.status == "unhealthy"
        assert "Connection refused" in (status.message or "")

    def test_count_tokens_heuristic(self, provider):
        count = provider.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count >= 1

    async def test_close(self, provider):
        provider._client = AsyncMock()
        await provider.close()
        provider._client.aclose.assert_awaited_once()

    async def test_chat_timeout(self, mock_ollama_provider_settings):
        transport = httpx.MockTransport(lambda _request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")))
        provider = OllamaProvider(settings=mock_ollama_provider_settings, provider_name="ollama")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")

        with pytest.raises(LLMTimeoutError):
            await provider.chat([Message(role="user", content="hi")])

    async def test_chat_http_error(self, mock_ollama_provider_settings):
        transport = httpx.MockTransport(lambda _request: httpx.Response(500, text="Internal Server Error"))
        provider = OllamaProvider(settings=mock_ollama_provider_settings, provider_name="ollama")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")

        with pytest.raises(LLMProviderError):
            await provider.chat([Message(role="user", content="hi")])

    async def test_stream(self, mock_ollama_provider_settings):
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": True, "prompt_eval_count": 5, "eval_count": 2}),
        ]
        body = "\n".join(lines)

        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, text=body, headers={"content-type": "application/x-ndjson"})
        )
        provider = OllamaProvider(settings=mock_ollama_provider_settings, provider_name="ollama")
        provider._client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")

        chunks = [c async for c in provider.stream([Message(role="user", content="hi")])]

        assert len(chunks) == 2
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[1].finish_reason == FinishReason.stop
        assert chunks[1].usage is not None

    async def test_to_ollama_message_with_tool_calls(self, provider):
        """F-016: tool_calls and tool_call_id are forwarded."""
        from ia_agent_fwk.llm.models import ToolCall

        msg = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="get_weather", arguments='{"city":"London"}')],
        )
        result = provider._to_ollama_message(msg)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][0]["function"]["arguments"] == {"city": "London"}

    async def test_to_ollama_message_with_tool_call_id(self, provider):
        """F-016: tool_call_id forwarded for tool messages."""
        msg = Message(role="tool", content="result", tool_call_id="tc_1")
        result = provider._to_ollama_message(msg)
        assert result["tool_call_id"] == "tc_1"
