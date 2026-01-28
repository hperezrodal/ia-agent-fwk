"""Tests for ChannelIntegration ABC."""

from __future__ import annotations

import pytest

from ia_agent_fwk.integrations.base import ChannelIntegration
from ia_agent_fwk.integrations.models import IncomingMessage, OutgoingMessage


class _ConcreteChannel(ChannelIntegration):
    """Minimal concrete implementation for testing."""

    @property
    def channel_type(self) -> str:
        return "test"

    async def send_message(self, message: OutgoingMessage) -> bool:
        return True

    async def process_incoming(self, raw_event: dict[str, object]) -> IncomingMessage | None:
        text = str(raw_event.get("text", ""))
        if not text:
            return None
        return IncomingMessage(
            channel="test",
            sender="sender",
            content=text,
        )


@pytest.mark.unit
class TestChannelIntegrationABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ChannelIntegration()  # type: ignore[abstract]

    def test_concrete_channel_type(self):
        channel = _ConcreteChannel()
        assert channel.channel_type == "test"

    async def test_send_message(self):
        channel = _ConcreteChannel()
        msg = OutgoingMessage(channel="test", recipient="user", content="Hello")
        result = await channel.send_message(msg)
        assert result is True

    async def test_process_incoming(self):
        channel = _ConcreteChannel()
        incoming = await channel.process_incoming({"text": "Hello"})
        assert incoming is not None
        assert incoming.content == "Hello"

    async def test_process_incoming_returns_none(self):
        channel = _ConcreteChannel()
        incoming = await channel.process_incoming({"text": ""})
        assert incoming is None

    async def test_start_is_noop(self):
        channel = _ConcreteChannel()
        await channel.start()  # Should not raise

    async def test_stop_is_noop(self):
        channel = _ConcreteChannel()
        await channel.stop()  # Should not raise

    async def test_health_check_default_true(self):
        channel = _ConcreteChannel()
        result = await channel.health_check()
        assert result is True
