"""WhatsApp channel integration.

Uses ``httpx`` to call the WhatsApp Cloud API.  No additional
dependencies are required beyond the project's existing ``httpx``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ia_agent_fwk.integrations.base import ChannelIntegration
from ia_agent_fwk.integrations.exceptions import (
    ChannelConnectionError,
    MessageDeliveryError,
)
from ia_agent_fwk.integrations.models import IncomingMessage, OutgoingMessage
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

_DEFAULT_API_URL = "https://graph.facebook.com/v18.0"


class WhatsAppIntegration(ChannelIntegration):
    """WhatsApp channel integration via WhatsApp Cloud API."""

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        verify_token: str = "",
        api_url: str = _DEFAULT_API_URL,
    ) -> None:
        self._access_token = access_token
        self._phone_number_id = phone_number_id
        self._verify_token = verify_token
        self._api_url = api_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def channel_type(self) -> str:
        return "whatsapp"

    async def start(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=30.0,
        )

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send a text message via WhatsApp Cloud API."""
        collector = get_metrics_collector()
        client = self._get_client()

        if not message.recipient:
            msg = "WhatsApp recipient phone number is required"
            raise MessageDeliveryError(msg)

        url = f"{self._api_url}/{self._phone_number_id}/messages"
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": message.recipient,
            "type": "text",
            "text": {"body": message.content},
        }

        start = time.monotonic()
        with _tracer.start_as_current_span(
            "integration.send_message",
            attributes={"integration.channel": "whatsapp"},
        ) as span:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except (httpx.HTTPStatusError, httpx.HTTPError) as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.record_exception(exc)
                collector.increment(
                    "integration_messages_sent_total",
                    labels={"channel": "whatsapp", "status": "error"},
                )
                collector.observe("integration_message_send_duration_seconds", duration_ms / 1000)
                logger.warning(
                    "WhatsApp send failed: recipient=%s (%.1fms)",
                    message.recipient,
                    duration_ms,
                    extra={
                        "integration_data": {
                            "event": "message_send_failed",
                            "channel": "whatsapp",
                            "recipient": message.recipient,
                            "duration_ms": round(duration_ms, 1),
                            "error": str(exc),
                        }
                    },
                )
                if isinstance(exc, httpx.HTTPStatusError):
                    msg = f"WhatsApp API error: {exc.response.status_code}"
                else:
                    msg = f"Failed to send WhatsApp message: {exc}"
                raise MessageDeliveryError(msg) from exc
            else:
                duration_ms = (time.monotonic() - start) * 1000
                collector.increment(
                    "integration_messages_sent_total",
                    labels={"channel": "whatsapp", "status": "success"},
                )
                collector.observe("integration_message_send_duration_seconds", duration_ms / 1000)
                span.set_attribute("integration.duration_ms", duration_ms)
                logger.info(
                    "WhatsApp message sent: recipient=%s (%.1fms)",
                    message.recipient,
                    duration_ms,
                    extra={
                        "integration_data": {
                            "event": "message_sent",
                            "channel": "whatsapp",
                            "recipient": message.recipient,
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
                return True

    async def process_incoming(  # noqa: PLR0911
        self,
        raw_event: dict[str, object],
    ) -> IncomingMessage | None:
        """Parse a WhatsApp webhook payload into an ``IncomingMessage``.

        Expects the Cloud API webhook format with nested
        ``entry[0].changes[0].value.messages[0]``.
        """
        try:
            entries: list[Any] = raw_event.get("entry", [])  # type: ignore[assignment]
            if not entries:
                return None

            changes: list[Any] = entries[0].get("changes", [])
            if not changes:
                return None

            value: dict[str, Any] = changes[0].get("value", {})
            messages: list[Any] = value.get("messages", [])
            if not messages:
                return None

            msg_data: dict[str, Any] = messages[0]
            msg_type: str = msg_data.get("type", "")
            if msg_type != "text":
                collector = get_metrics_collector()
                collector.increment(
                    "integration_messages_ignored_total",
                    labels={"channel": "whatsapp", "reason": "non_text_type"},
                )
                return None

            text_body: str = msg_data.get("text", {}).get("body", "")
            if not text_body:
                return None

            sender = str(msg_data.get("from", ""))
            timestamp = str(msg_data.get("timestamp", ""))

            # Extract metadata
            metadata: dict[str, str] = {
                "message_id": str(msg_data.get("id", "")),
            }

            # Extract phone_number_id from metadata
            wa_metadata: dict[str, Any] = value.get("metadata", {})
            if wa_metadata:
                metadata["phone_number_id"] = str(wa_metadata.get("phone_number_id", ""))

        except (KeyError, IndexError, TypeError):
            collector = get_metrics_collector()
            collector.increment(
                "integration_messages_ignored_total",
                labels={"channel": "whatsapp", "reason": "parse_error"},
            )
            logger.debug("Failed to parse WhatsApp webhook payload")
            return None
        else:
            collector = get_metrics_collector()
            collector.increment(
                "integration_messages_received_total",
                labels={"channel": "whatsapp"},
            )
            logger.info(
                "WhatsApp message received: sender=%s",
                sender,
                extra={
                    "integration_data": {
                        "event": "message_received",
                        "channel": "whatsapp",
                        "sender": sender,
                    }
                },
            )
            return IncomingMessage(
                channel="whatsapp",
                sender=sender,
                content=text_body,
                metadata=metadata,
                timestamp=timestamp,
                conversation_id=sender,
            )

    def verify_webhook(self, mode: str, token: str, challenge: str) -> str | None:
        """Verify a WhatsApp webhook subscription challenge.

        Parameters
        ----------
        mode:
            The ``hub.mode`` query parameter (should be ``'subscribe'``).
        token:
            The ``hub.verify_token`` query parameter.
        challenge:
            The ``hub.challenge`` query parameter to echo back.

        Returns
        -------
        str | None
            The challenge string if verification succeeds, else ``None``.

        """
        if mode == "subscribe" and token == self._verify_token:
            return challenge
        return None

    async def health_check(self) -> bool:
        """Check WhatsApp API connectivity."""
        collector = get_metrics_collector()
        if self._client is None:
            collector.increment(
                "integration_health_checks_total",
                labels={"channel": "whatsapp", "status": "unhealthy"},
            )
            return False
        try:
            url = f"{self._api_url}/{self._phone_number_id}"
            resp = await self._client.get(url)
        except httpx.HTTPError:
            collector.increment(
                "integration_health_checks_total",
                labels={"channel": "whatsapp", "status": "unhealthy"},
            )
            return False
        else:
            healthy = resp.status_code == 200  # noqa: PLR2004
            collector.increment(
                "integration_health_checks_total",
                labels={"channel": "whatsapp", "status": "healthy" if healthy else "unhealthy"},
            )
            return healthy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, raising if not started."""
        if self._client is None:
            msg = "WhatsAppIntegration has not been started. Call start() first."
            raise ChannelConnectionError(msg)
        return self._client
