"""Integration tests for Ollama provider (requires Ollama running locally)."""

from __future__ import annotations

import pytest

from ia_agent_fwk.llm.models import Message
from ia_agent_fwk.llm.providers.ollama import OllamaProvider
from tests.integration.test_llm.conftest import skip_no_ollama


@skip_no_ollama
@pytest.mark.integration
class TestOllamaIntegration:
    @pytest.fixture
    def provider(self, ollama_settings):
        return OllamaProvider(settings=ollama_settings, provider_name="ollama")

    async def test_chat(self, provider):
        resp = await provider.chat([Message(role="user", content="Say hello")])
        assert resp.message.content
        assert resp.usage.total_tokens > 0

    async def test_stream(self, provider):
        chunks = []
        async for c in provider.stream([Message(role="user", content="Say hello")]):
            chunks.append(c)
        assert len(chunks) > 0

    async def test_health_check(self, provider):
        status = await provider.health_check()
        assert status.status == "healthy"
