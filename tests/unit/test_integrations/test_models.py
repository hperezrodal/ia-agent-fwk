"""Tests for integration data models."""

from __future__ import annotations

import pytest

from ia_agent_fwk.integrations.models import ChannelConfig, IncomingMessage, OutgoingMessage


@pytest.mark.unit
class TestIncomingMessage:
    def test_create_minimal(self):
        msg = IncomingMessage(channel="slack", sender="U123", content="Hello")
        assert msg.channel == "slack"
        assert msg.sender == "U123"
        assert msg.content == "Hello"
        assert msg.metadata == {}
        assert msg.timestamp == ""
        assert msg.conversation_id is None

    def test_create_full(self):
        msg = IncomingMessage(
            channel="whatsapp",
            sender="+1234567890",
            content="Hi there",
            metadata={"key": "value"},
            timestamp="2026-01-01T00:00:00Z",
            conversation_id="conv-123",
        )
        assert msg.channel == "whatsapp"
        assert msg.metadata == {"key": "value"}
        assert msg.conversation_id == "conv-123"

    def test_frozen(self):
        msg = IncomingMessage(channel="slack", sender="U123", content="Hello")
        with pytest.raises(Exception):  # noqa: B017
            msg.content = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestOutgoingMessage:
    def test_create_minimal(self):
        msg = OutgoingMessage(channel="slack", recipient="#general", content="Hello")
        assert msg.channel == "slack"
        assert msg.recipient == "#general"
        assert msg.content == "Hello"
        assert msg.format == "text"

    def test_create_with_format(self):
        msg = OutgoingMessage(
            channel="email",
            recipient="user@test.com",
            content="<h1>Hello</h1>",
            format="html",
        )
        assert msg.format == "html"

    def test_frozen(self):
        msg = OutgoingMessage(channel="slack", recipient="#general", content="Hello")
        with pytest.raises(Exception):  # noqa: B017
            msg.content = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestChannelConfig:
    def test_create_defaults(self):
        config = ChannelConfig(channel_type="slack")
        assert config.channel_type == "slack"
        assert config.enabled is False
        assert config.agent_type == ""
        assert config.settings == {}

    def test_create_full(self):
        config = ChannelConfig(
            channel_type="whatsapp",
            enabled=True,
            agent_type="customer-support",
            settings={"key": "value"},
        )
        assert config.enabled is True
        assert config.agent_type == "customer-support"

    def test_not_frozen(self):
        config = ChannelConfig(channel_type="slack")
        config.enabled = True
        assert config.enabled is True
