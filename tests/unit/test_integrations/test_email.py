"""Tests for Email integration."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.integrations.exceptions import ChannelConnectionError, MessageDeliveryError
from ia_agent_fwk.integrations.models import OutgoingMessage


def _make_mock_aiosmtplib():
    """Create a mock aiosmtplib module."""
    mock_mod = MagicMock(spec=ModuleType)
    mock_mod.send = AsyncMock()
    mock_mod.SMTPException = type("SMTPException", (Exception,), {})
    return mock_mod


@pytest.mark.unit
class TestEmailIntegration:
    def test_channel_type(self, email_integration):
        assert email_integration.channel_type == "email"

    async def test_send_message_success(self, email_integration):
        msg = OutgoingMessage(
            channel="email",
            recipient="user@test.com",
            content="Hello from agent",
            metadata={"subject": "Test Subject"},
        )

        mock_mod = _make_mock_aiosmtplib()
        with patch.dict(sys.modules, {"aiosmtplib": mock_mod}):
            result = await email_integration.send_message(msg)

        assert result is True
        mock_mod.send.assert_called_once()

    async def test_send_message_no_recipient(self, email_integration):
        msg = OutgoingMessage(channel="email", recipient="", content="Hello")

        with pytest.raises(MessageDeliveryError, match="recipient is required"):
            await email_integration.send_message(msg)

    async def test_send_message_no_aiosmtplib(self, email_integration):
        msg = OutgoingMessage(
            channel="email",
            recipient="user@test.com",
            content="Hello",
        )
        with (
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aiosmtplib",
                return_value=False,
            ),
            pytest.raises(ChannelConnectionError, match="aiosmtplib is required"),
        ):
            await email_integration.send_message(msg)

    async def test_send_message_smtp_error(self, email_integration):
        msg = OutgoingMessage(
            channel="email",
            recipient="user@test.com",
            content="Hello",
        )

        mock_mod = _make_mock_aiosmtplib()
        smtp_exc = mock_mod.SMTPException("SMTP error")
        mock_mod.send = AsyncMock(side_effect=smtp_exc)

        with (
            patch.dict(sys.modules, {"aiosmtplib": mock_mod}),
            pytest.raises(MessageDeliveryError, match="Failed to send email"),
        ):
            await email_integration.send_message(msg)

    async def test_process_incoming_webhook(self, email_integration):
        raw_event = {
            "from": "sender@test.com",
            "to": "bot@test.com",
            "subject": "Test Subject",
            "text": "Hello agent",
            "timestamp": "2026-01-01T00:00:00Z",
            "message_id": "msg-123",
        }
        msg = await email_integration.process_incoming(raw_event)
        assert msg is not None
        assert msg.channel == "email"
        assert msg.sender == "sender@test.com"
        assert msg.content == "Hello agent"
        assert msg.metadata["subject"] == "Test Subject"
        assert msg.metadata["to"] == "bot@test.com"
        assert msg.conversation_id == "msg-123"

    async def test_process_incoming_missing_sender(self, email_integration):
        raw_event = {"text": "Hello", "from": ""}
        msg = await email_integration.process_incoming(raw_event)
        assert msg is None

    async def test_process_incoming_missing_text(self, email_integration):
        raw_event = {"from": "sender@test.com", "text": ""}
        msg = await email_integration.process_incoming(raw_event)
        assert msg is None

    async def test_health_check_with_aiosmtplib(self, email_integration):
        with patch(
            "ia_agent_fwk.integrations.email_channel._has_aiosmtplib",
            return_value=True,
        ):
            result = await email_integration.health_check()
        assert result is True

    async def test_health_check_without_aiosmtplib(self, email_integration):
        with patch(
            "ia_agent_fwk.integrations.email_channel._has_aiosmtplib",
            return_value=False,
        ):
            result = await email_integration.health_check()
        assert result is False
