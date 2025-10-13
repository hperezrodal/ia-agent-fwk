"""Tests for the LLMProvider ABC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr

from ia_agent_fwk.config.settings import LLMProviderSettings
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


class ConcreteProvider(LLMProvider):
    """Minimal concrete subclass for testing."""

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content="ok"),
            usage=TokenUsage(),
            model="test",
            finish_reason=FinishReason.stop,
        )

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text="ok", usage=TokenUsage(), model="test")

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="hi", finish_reason=FinishReason.stop)

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text)

    async def health_check(self) -> HealthStatus:
        return HealthStatus(status="healthy")


class PartialProvider(LLMProvider):
    """Subclass missing abstract methods."""

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content="ok"),
            usage=TokenUsage(),
            model="test",
            finish_reason=FinishReason.stop,
        )

    # Deliberately omit complete, stream, count_tokens, health_check


class TestLLMProviderABC:
    def _make_settings(self) -> LLMProviderSettings:
        return LLMProviderSettings(api_key=SecretStr("test"))

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            LLMProvider(settings=self._make_settings(), provider_name="test")  # type: ignore[abstract]

    def test_partial_subclass_cannot_instantiate(self):
        with pytest.raises(TypeError):
            PartialProvider(settings=self._make_settings(), provider_name="test")  # type: ignore[abstract]

    def test_concrete_subclass_instantiation(self):
        provider = ConcreteProvider(settings=self._make_settings(), provider_name="test")
        assert provider.provider_name == "test"

    async def test_close_is_noop(self):
        provider = ConcreteProvider(settings=self._make_settings(), provider_name="test")
        await provider.close()  # should not raise

    async def test_embed_raises_not_implemented(self):
        provider = ConcreteProvider(settings=self._make_settings(), provider_name="test")
        with pytest.raises(NotImplementedError, match="EmbeddingProvider"):
            await provider.embed(["hello"])

    async def test_chat(self):
        provider = ConcreteProvider(settings=self._make_settings(), provider_name="test")
        resp = await provider.chat([Message(role="user", content="hi")])
        assert resp.message.content == "ok"

    async def test_stream(self):
        provider = ConcreteProvider(settings=self._make_settings(), provider_name="test")
        chunks = [c async for c in provider.stream([Message(role="user", content="hi")])]
        assert len(chunks) == 1
        assert chunks[0].delta == "hi"
