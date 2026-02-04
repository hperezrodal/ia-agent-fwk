"""Tests for WhatsApp integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ia_agent_fwk.integrations.exceptions import ChannelConnectionError, MessageDeliveryError
from ia_agent_fwk.integrations.models import OutgoingMessage


@pytest.mark.unit
class TestWhatsAppIntegration:
    def test_channel_type(self, whatsapp_integration):
        assert whatsapp_integration.channel_type == "whatsapp"

    async def test_start_creates_client(self, whatsapp_integration):
        await whatsapp_integration.start()
        assert whatsapp_integration._client is not None
        await whatsapp_integration.stop()

    async def test_stop_closes_client(self, whatsapp_integration):
        await whatsapp_integration.start()
        await whatsapp_integration.stop()
        assert whatsapp_integration._client is None

    async def test_send_message_not_started(self, whatsapp_integration):
        msg = OutgoingMessage(channel="whatsapp", recipient="+1234567890", content="Hello")
        with pytest.raises(ChannelConnectionError, match="not been started"):
            await whatsapp_integration.send_message(msg)

    async def test_send_message_success(self, whatsapp_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        whatsapp_integration._client = mock_client

        msg = OutgoingMessage(channel="whatsapp", recipient="+1234567890", content="Hello")
        result = await whatsapp_integration.send_message(msg)
        assert result is True

        call_args = mock_client.post.call_args
        assert "messages" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["messaging_product"] == "whatsapp"
        assert payload["to"] == "+1234567890"
        assert payload["text"]["body"] == "Hello"

    async def test_send_message_no_recipient(self, whatsapp_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        whatsapp_integration._client = mock_client

        msg = OutgoingMessage(channel="whatsapp", recipient="", content="Hello")
        with pytest.raises(MessageDeliveryError, match="recipient phone number"):
            await whatsapp_integration.send_message(msg)

    async def test_send_message_http_error(self, whatsapp_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        whatsapp_integration._client = mock_client

        msg = OutgoingMessage(channel="whatsapp", recipient="+1234567890", content="Hello")
        with pytest.raises(MessageDeliveryError, match="Failed to send"):
            await whatsapp_integration.send_message(msg)

    async def test_send_message_status_error(self, whatsapp_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )
        mock_client.post.return_value = mock_response
        whatsapp_integration._client = mock_client

        msg = OutgoingMessage(channel="whatsapp", recipient="+1234567890", content="Hello")
        with pytest.raises(MessageDeliveryError, match="WhatsApp API error"):
            await whatsapp_integration.send_message(msg)

    async def test_process_incoming_text_message(self, whatsapp_integration):
        raw_event = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "metadata": {"phone_number_id": "123456789"},
                                "messages": [
                                    {
                                        "from": "+1234567890",
                                        "id": "wamid.test123",
                                        "timestamp": "1234567890",
                                        "type": "text",
                                        "text": {"body": "Hello agent"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }
        msg = await whatsapp_integration.process_incoming(raw_event)
        assert msg is not None
        assert msg.channel == "whatsapp"
        assert msg.sender == "+1234567890"
        assert msg.content == "Hello agent"
        assert msg.metadata["message_id"] == "wamid.test123"
        assert msg.metadata["phone_number_id"] == "123456789"
        assert msg.conversation_id == "+1234567890"

    async def test_process_incoming_non_text_message(self, whatsapp_integration):
        raw_event = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "+1234567890",
                                        "type": "image",
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }
        msg = await whatsapp_integration.process_incoming(raw_event)
        assert msg is None

    async def test_process_incoming_empty_entries(self, whatsapp_integration):
        raw_event = {"entry": []}
        msg = await whatsapp_integration.process_incoming(raw_event)
        assert msg is None

    async def test_process_incoming_no_messages(self, whatsapp_integration):
        raw_event = {"entry": [{"changes": [{"value": {"messages": []}}]}]}
        msg = await whatsapp_integration.process_incoming(raw_event)
        assert msg is None

    async def test_process_incoming_invalid_payload(self, whatsapp_integration):
        raw_event = {"invalid": "payload"}
        msg = await whatsapp_integration.process_incoming(raw_event)
        assert msg is None

    def test_verify_webhook_success(self, whatsapp_integration):
        result = whatsapp_integration.verify_webhook(
            mode="subscribe",
            token="test-verify-token",
            challenge="challenge-123",
        )
        assert result == "challenge-123"

    def test_verify_webhook_wrong_mode(self, whatsapp_integration):
        result = whatsapp_integration.verify_webhook(
            mode="unsubscribe",
            token="test-verify-token",
            challenge="challenge-123",
        )
        assert result is None

    def test_verify_webhook_wrong_token(self, whatsapp_integration):
        result = whatsapp_integration.verify_webhook(
            mode="subscribe",
            token="wrong-token",
            challenge="challenge-123",
        )
        assert result is None

    async def test_health_check_not_started(self, whatsapp_integration):
        result = await whatsapp_integration.health_check()
        assert result is False

    async def test_health_check_success(self, whatsapp_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        whatsapp_integration._client = mock_client

        result = await whatsapp_integration.health_check()
        assert result is True

    async def test_health_check_failure(self, whatsapp_integration):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("refused")
        whatsapp_integration._client = mock_client

        result = await whatsapp_integration.health_check()
        assert result is False
