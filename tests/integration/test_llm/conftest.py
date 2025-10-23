"""Integration test fixtures and skip conditions for LLM providers."""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from ia_agent_fwk.config.settings import LLMProviderSettings

skip_no_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

skip_no_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

skip_no_ollama = pytest.mark.skipif(
    not os.environ.get("OLLAMA_AVAILABLE", ""),
    reason="OLLAMA_AVAILABLE not set (Ollama not running)",
)


@pytest.fixture
def openai_settings() -> LLMProviderSettings:
    return LLMProviderSettings(
        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
        default_model="gpt-4o",
        max_tokens=50,
        timeout=30,
    )


@pytest.fixture
def anthropic_settings() -> LLMProviderSettings:
    return LLMProviderSettings(
        api_key=SecretStr(os.environ.get("ANTHROPIC_API_KEY", "")),
        default_model="claude-sonnet-4-20250514",
        max_tokens=50,
        timeout=30,
    )


@pytest.fixture
def ollama_settings() -> LLMProviderSettings:
    return LLMProviderSettings(
        base_url="http://localhost:11434",
        default_model="llama3.1",
        max_tokens=50,
        timeout=120,
    )
