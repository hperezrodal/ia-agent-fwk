"""Tests for Slack integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ia_agent_fwk.integrations.exceptions import ChannelConnectionError, MessageDeliveryError
from ia_agent_fwk.integrations.models import OutgoingMessage
from ia_agent_fwk.integrations.slack import SlackIntegration


@pytest.mark.unit
class TestSlackIntegration:
    def test_channel_type(self, slack_integration):
        assert slack_integration.channel_type == "slack"

    async def test_start_creates_client(self, slack_integration):
        await slack_integration.start()
        assert slack_integration._client is not None
        await slack_integration.stop()

    async def test_stop_closes_client(self, slack_integration):
        await slack_integration.start()
        await slack_integration.stop()
        assert slack_integration._client is None

    async def test_send_message_not_started(self, slack_integration):
        msg = OutgoingMessage(channel="slack", recipient="#general", content="Hello")
        with pytest.raises(ChannelConnectionError, match="not been started"):
            await slack_integration.send_message(msg)

    async def test_send_message_success(self, slack_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_client.post.return_value = mock_response
        slack_integration._client = mock_client

        msg = OutgoingMessage(channel="slack", recipient="#general", content="Hello")
        result = await slack_integration.send_message(msg)
        assert result is True
        mock_client.post.assert_called_once_with(
            "/chat.postMessage",
            json={"channel": "#general", "text": "Hello"},
        )

    async def test_send_message_uses_default_channel(self, slack_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_client.post.return_value = mock_response
        slack_integration._client = mock_client

        msg = OutgoingMessage(channel="slack", recipient="", content="Hello")
        result = await slack_integration.send_message(msg)
        assert result is True
        mock_client.post.assert_called_once_with(
            "/chat.postMessage",
            json={"channel": "#general", "text": "Hello"},
        )

    async def test_send_message_no_recipient_no_default(self):
        integration = SlackIntegration(bot_token="xoxb-test", default_channel="")  # noqa: S106
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        integration._client = mock_client

        msg = OutgoingMessage(channel="slack", recipient="", content="Hello")
        with pytest.raises(MessageDeliveryError, match="No recipient"):
            await integration.send_message(msg)

    async def test_send_message_api_error(self, slack_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}
        mock_client.post.return_value = mock_response
        slack_integration._client = mock_client

        msg = OutgoingMessage(channel="slack", recipient="#bad", content="Hello")
        with pytest.raises(MessageDeliveryError, match="channel_not_found"):
            await slack_integration.send_message(msg)

    async def test_send_message_http_error(self, slack_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        slack_integration._client = mock_client

        msg = OutgoingMessage(channel="slack", recipient="#general", content="Hello")
        with pytest.raises(MessageDeliveryError, match="Failed to send"):
            await slack_integration.send_message(msg)

    async def test_process_incoming_message(self, slack_integration):
        raw_event = {
            "event": {
                "type": "message",
                "user": "U12345",
                "text": "Hello bot",
                "channel": "C12345",
                "ts": "1234567890.123456",
            },
            "team_id": "T12345",
        }
        msg = await slack_integration.process_incoming(raw_event)
        assert msg is not None
        assert msg.channel == "slack"
        assert msg.sender == "U12345"
        assert msg.content == "Hello bot"
        assert msg.metadata["channel_id"] == "C12345"
        assert msg.metadata["team_id"] == "T12345"

    async def test_process_incoming_ignores_bot_message(self, slack_integration):
        raw_event = {
            "event": {
                "type": "message",
                "bot_id": "B12345",
                "text": "Bot message",
                "channel": "C12345",
            },
        }
        msg = await slack_integration.process_incoming(raw_event)
        assert msg is None

    async def test_process_incoming_ignores_empty_text(self, slack_integration):
        raw_event = {
            "event": {
                "type": "message",
                "user": "U12345",
                "text": "",
                "channel": "C12345",
            },
        }
        msg = await slack_integration.process_incoming(raw_event)
        assert msg is None

    async def test_process_incoming_flat_event(self, slack_integration):
        raw_event = {
            "type": "message",
            "user": "U12345",
            "text": "Direct event",
            "channel": "C12345",
        }
        msg = await slack_integration.process_incoming(raw_event)
        assert msg is not None
        assert msg.content == "Direct event"

    async def test_health_check_success(self, slack_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_client.post.return_value = mock_response
        slack_integration._client = mock_client

        result = await slack_integration.health_check()
        assert result is True

    async def test_health_check_failure(self, slack_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("refused")
        slack_integration._client = mock_client

        result = await slack_integration.health_check()
        assert result is False

    def test_verify_signature_valid(self, slack_integration):
        import hashlib
        import hmac
        import time

        timestamp = str(int(time.time()))
        body = '{"type":"event_callback"}'
        sig_basestring = f"v0:{timestamp}:{body}"
        expected = (
            "v0="
            + hmac.new(
                b"test-signing-secret",
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        result = slack_integration.verify_signature(timestamp, body, expected)
        assert result is True

    def test_verify_signature_invalid(self, slack_integration):
        result = slack_integration.verify_signature("123", "body", "v0=wrong")
        assert result is False

    def test_verify_signature_no_secret(self):
        integration = SlackIntegration(bot_token="xoxb-test", signing_secret="")  # noqa: S106
        result = integration.verify_signature("123", "body", "v0=sig")
        assert result is False

    def test_verify_signature_old_timestamp(self, slack_integration):
        result = slack_integration.verify_signature("1000000000", "body", "v0=sig")
        assert result is False
