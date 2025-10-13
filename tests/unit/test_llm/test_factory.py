"""Tests for LLMProviderFactory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr

from ia_agent_fwk.config.settings import (
    CircuitBreakerSettings,
    LLMProviderSettings,
    LLMSettings,
    RetrySettings,
)
from ia_agent_fwk.llm.base import LLMProvider
from ia_agent_fwk.llm.exceptions import LLMConfigError
from ia_agent_fwk.llm.factory import LLMProviderFactory
from ia_agent_fwk.llm.models import (
    ChatResponse,
    CompletionResponse,
    FinishReason,
    HealthStatus,
    Message,
    StreamChunk,
    TokenUsage,
)


class _FakeProvider(LLMProvider):
    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content="fake"),
            usage=TokenUsage(),
            model="fake",
            finish_reason=FinishReason.stop,
        )

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text="fake", usage=TokenUsage(), model="fake")

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="fake")

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return 0

    async def health_check(self) -> HealthStatus:
        return HealthStatus(status="healthy")


def _make_settings(provider_name: str = "openai") -> LLMSettings:
    return LLMSettings(
        default_provider=provider_name,
        providers={
            provider_name: LLMProviderSettings(
                api_key=SecretStr("test-key"),
                default_model="test-model",
                retry=RetrySettings(max_attempts=1),
                circuit_breaker=CircuitBreakerSettings(enabled=False),
            )
        },
    )


class TestLLMProviderFactory:
    def test_create_openai(self):
        settings = _make_settings("openai")
        provider = LLMProviderFactory.create(settings)
        # Just check it creates something -- we can't check the exact type
        # without importing the OpenAI provider, but it should be an LLMProvider.
        assert isinstance(provider, LLMProvider)
        assert provider.provider_name == "openai"

    def test_create_with_explicit_name(self):
        settings = LLMSettings(
            default_provider="openai",
            providers={
                "ollama": LLMProviderSettings(
                    base_url="http://localhost:11434",
                    default_model="llama3.1",
                    retry=RetrySettings(max_attempts=1),
                    circuit_breaker=CircuitBreakerSettings(enabled=False),
                ),
            },
        )
        provider = LLMProviderFactory.create(settings, provider_name="ollama")
        assert provider.provider_name == "ollama"

    def test_unknown_provider_raises(self):
        settings = LLMSettings(default_provider="gpt-magic")
        with pytest.raises(LLMConfigError, match="Unknown LLM provider 'gpt-magic'"):
            LLMProviderFactory.create(settings)

    def test_register_custom_provider(self):
        LLMProviderFactory.register("fake", _FakeProvider)
        settings = LLMSettings(
            default_provider="fake",
            providers={
                "fake": LLMProviderSettings(
                    api_key=SecretStr("k"),
                    retry=RetrySettings(max_attempts=1),
                    circuit_breaker=CircuitBreakerSettings(enabled=False),
                ),
            },
        )
        provider = LLMProviderFactory.create(settings)
        assert isinstance(provider, _FakeProvider)
        # Clean up.
        del LLMProviderFactory._registry["fake"]

    def test_register_lazy_string(self):
        LLMProviderFactory.register(
            "fake_lazy",
            f"{_FakeProvider.__module__}:{_FakeProvider.__qualname__}",
        )
        settings = LLMSettings(
            default_provider="fake_lazy",
            providers={
                "fake_lazy": LLMProviderSettings(
                    api_key=SecretStr("k"),
                    retry=RetrySettings(max_attempts=1),
                    circuit_breaker=CircuitBreakerSettings(enabled=False),
                ),
            },
        )
        provider = LLMProviderFactory.create(settings)
        assert isinstance(provider, _FakeProvider)
        del LLMProviderFactory._registry["fake_lazy"]

    def test_error_lists_valid_providers(self):
        settings = LLMSettings(default_provider="unknown")
        with pytest.raises(LLMConfigError) as exc_info:
            LLMProviderFactory.create(settings)
        msg = str(exc_info.value)
        assert "anthropic" in msg
        assert "ollama" in msg
        assert "openai" in msg

    def test_register_duplicate_raises(self):
        """F-020: Registering a duplicate name without replace=True raises."""
        with pytest.raises(LLMConfigError, match="already registered"):
            LLMProviderFactory.register("openai", _FakeProvider)

    def test_register_duplicate_with_replace(self):
        """F-020: replace=True allows overwriting an existing registration."""
        original = LLMProviderFactory._registry["openai"]
        try:
            LLMProviderFactory.register("openai", _FakeProvider, replace=True)
            assert LLMProviderFactory._registry["openai"] is _FakeProvider
        finally:
            LLMProviderFactory._registry["openai"] = original
