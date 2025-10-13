"""Shared fixtures for LLM unit tests."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from ia_agent_fwk.config.settings import (
    CircuitBreakerSettings,
    LLMProviderSettings,
    LLMSettings,
    RetrySettings,
)
from ia_agent_fwk.llm.models import (
    ChatResponse,
    FinishReason,
    Message,
    TokenUsage,
)


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
    ]


@pytest.fixture
def sample_token_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)


@pytest.fixture
def sample_chat_response(sample_token_usage) -> ChatResponse:
    return ChatResponse(
        message=Message(role="assistant", content="Hi there!"),
        usage=sample_token_usage,
        model="test-model",
        finish_reason=FinishReason.stop,
    )


@pytest.fixture
def mock_openai_provider_settings() -> LLMProviderSettings:
    return LLMProviderSettings(
        api_key=SecretStr("sk-test-key-123"),
        default_model="gpt-4o",
        temperature=0.7,
        max_tokens=4096,
        timeout=60,
        retry=RetrySettings(max_attempts=1, backoff_base=0.01, backoff_max=0.01),
        circuit_breaker=CircuitBreakerSettings(enabled=False),
    )


@pytest.fixture
def mock_anthropic_provider_settings() -> LLMProviderSettings:
    return LLMProviderSettings(
        api_key=SecretStr("sk-ant-test-key-123"),
        default_model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=4096,
        timeout=60,
        retry=RetrySettings(max_attempts=1, backoff_base=0.01, backoff_max=0.01),
        circuit_breaker=CircuitBreakerSettings(enabled=False),
    )


@pytest.fixture
def mock_ollama_provider_settings() -> LLMProviderSettings:
    return LLMProviderSettings(
        api_key=SecretStr(""),
        base_url="http://localhost:11434",
        default_model="llama3.1",
        temperature=0.7,
        max_tokens=4096,
        timeout=120,
        retry=RetrySettings(max_attempts=1, backoff_base=0.01, backoff_max=0.01),
        circuit_breaker=CircuitBreakerSettings(enabled=False),
    )


@pytest.fixture
def mock_vllm_provider_settings() -> LLMProviderSettings:
    return LLMProviderSettings(
        api_key=SecretStr(""),
        base_url="http://localhost:8000/v1",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.7,
        max_tokens=4096,
        timeout=120,
        retry=RetrySettings(max_attempts=1, backoff_base=0.01, backoff_max=0.01),
        circuit_breaker=CircuitBreakerSettings(enabled=False),
    )


@pytest.fixture
def mock_huggingface_provider_settings() -> LLMProviderSettings:
    return LLMProviderSettings(
        api_key=SecretStr(""),
        base_url="cpu",
        default_model="gpt2",
        temperature=0.7,
        max_tokens=256,
        timeout=300,
        retry=RetrySettings(max_attempts=1, backoff_base=0.01, backoff_max=0.01),
        circuit_breaker=CircuitBreakerSettings(enabled=False),
    )


@pytest.fixture
def mock_llm_settings(mock_openai_provider_settings) -> LLMSettings:
    return LLMSettings(
        default_provider="openai",
        providers={"openai": mock_openai_provider_settings},
    )
