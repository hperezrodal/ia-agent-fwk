"""Tests for ChannelRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.agents.config import AgentResult
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.config.settings import LLMSettings
from ia_agent_fwk.integrations.base import ChannelIntegration
from ia_agent_fwk.integrations.exceptions import ChannelConfigError
from ia_agent_fwk.integrations.models import ChannelConfig, IncomingMessage, OutgoingMessage
from ia_agent_fwk.llm.models import TokenUsage


class _MockChannel(ChannelIntegration):
    """Mock channel for testing the router."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name
        self.sent_messages: list[OutgoingMessage] = []

    @property
    def channel_type(self) -> str:
        return self._name

    async def send_message(self, message: OutgoingMessage) -> bool:
        self.sent_messages.append(message)
        return True

    async def process_incoming(self, raw_event: dict[str, object]) -> IncomingMessage | None:
        text = str(raw_event.get("text", ""))
        if not text:
            return None
        return IncomingMessage(
            channel=self._name,
            sender="test-sender",
            content=text,
        )


@pytest.mark.unit
class TestChannelRouter:
    def test_register_channel(self, channel_router):
        channel = _MockChannel()
        config = ChannelConfig(channel_type="mock", enabled=True, agent_type="test")
        channel_router.register(channel, config)
        assert channel_router.get_channel("mock") is channel

    def test_get_channel_not_found(self, channel_router):
        assert channel_router.get_channel("nonexistent") is None

    def test_list_channels_empty(self, channel_router):
        assert channel_router.list_channels() == []

    def test_list_channels(self, channel_router):
        channel_router.register(
            _MockChannel("beta"),
            ChannelConfig(channel_type="beta", enabled=True, agent_type="test"),
        )
        channel_router.register(
            _MockChannel("alpha"),
            ChannelConfig(channel_type="alpha", enabled=True, agent_type="test"),
        )
        assert channel_router.list_channels() == ["alpha", "beta"]

    async def test_route_unknown_channel(self, channel_router):
        llm_settings = LLMSettings()
        result = await channel_router.route_incoming("nonexistent", {}, llm_settings)
        assert result is None

    async def test_route_incoming_to_agent(self, channel_router):
        channel = _MockChannel()
        config = ChannelConfig(channel_type="mock", enabled=True, agent_type="test")
        channel_router.register(channel, config)

        llm_settings = LLMSettings()

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            return_value=AgentResult(
                output="Agent says hello",
                state=AgentState.COMPLETED,
                usage=TokenUsage(prompt_tokens=5, completion_tokens=10),
                iterations=1,
                duration_ms=10.0,
            )
        )

        with (
            patch("ia_agent_fwk.agents.factory.AgentFactory.create") as mock_create,
        ):
            mock_create.return_value = mock_agent

            result = await channel_router.route_incoming(
                "mock",
                {"text": "Hello agent"},
                llm_settings,
            )

        assert result == "Agent says hello"
        assert len(channel.sent_messages) == 1
        assert channel.sent_messages[0].content == "Agent says hello"
        assert channel.sent_messages[0].recipient == "test-sender"

    async def test_route_incoming_ignored_event(self, channel_router):
        channel = _MockChannel()
        config = ChannelConfig(channel_type="mock", enabled=True, agent_type="test")
        channel_router.register(channel, config)

        llm_settings = LLMSettings()
        result = await channel_router.route_incoming(
            "mock",
            {"text": ""},  # Empty text -> process_incoming returns None
            llm_settings,
        )
        assert result is None

    async def test_route_no_agent_type(self, channel_router):
        channel = _MockChannel()
        config = ChannelConfig(channel_type="mock", enabled=True, agent_type="")
        channel_router.register(channel, config)

        llm_settings = LLMSettings()
        with pytest.raises(ChannelConfigError, match="No agent_type"):
            await channel_router.route_incoming("mock", {"text": "Hello"}, llm_settings)
