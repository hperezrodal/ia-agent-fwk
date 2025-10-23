"""Integration tests for Anthropic provider (requires ANTHROPIC_API_KEY)."""

from __future__ import annotations

import pytest

from ia_agent_fwk.llm.models import Message
from ia_agent_fwk.llm.providers.anthropic import AnthropicProvider
from tests.integration.test_llm.conftest import skip_no_anthropic_key


@skip_no_anthropic_key
@pytest.mark.integration
class TestAnthropicIntegration:
    @pytest.fixture
    def provider(self, anthropic_settings):
        return AnthropicProvider(settings=anthropic_settings, provider_name="anthropic")

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
