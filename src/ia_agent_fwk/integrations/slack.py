"""Slack channel integration.

Uses ``httpx`` to call the Slack Web API directly, keeping
external dependencies minimal.  The optional ``slack-sdk`` package
is **not** required at runtime -- this module relies only on
``httpx`` (already a project dependency).
"""

from __future__ import annotations

import hashlib
import hmac
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

_SLACK_API_BASE = "https://slack.com/api"


class SlackIntegration(ChannelIntegration):
    """Slack channel integration via Slack Web API."""

    def __init__(
        self,
        bot_token: str,
        signing_secret: str = "",
        default_channel: str = "",
    ) -> None:
        self._bot_token = bot_token
        self._signing_secret = signing_secret
        self._default_channel = default_channel
        self._client: httpx.AsyncClient | None = None

    @property
    def channel_type(self) -> str:
        return "slack"

    async def start(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=_SLACK_API_BASE,
            headers={"Authorization": f"Bearer {self._bot_token}"},
            timeout=30.0,
        )

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Ping Slack ``auth.test`` to verify connectivity."""
        collector = get_metrics_collector()
        client = self._get_client()
        try:
            resp = await client.post("/auth.test")
            data: dict[str, Any] = resp.json()
            healthy = bool(data.get("ok", False))
        except httpx.HTTPError:
            healthy = False
        status = "healthy" if healthy else "unhealthy"
        collector.increment(
            "integration_health_checks_total",
            labels={"channel": "slack", "status": status},
        )
        return healthy

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Post a message to a Slack channel or DM."""
        collector = get_metrics_collector()
        client = self._get_client()
        channel = message.recipient or self._default_channel
        if not channel:
            msg = "No recipient or default_channel configured for Slack"
            raise MessageDeliveryError(msg)

        start = time.monotonic()
        with _tracer.start_as_current_span(
            "integration.send_message",
            attributes={"integration.channel": "slack"},
        ) as span:
            try:
                resp = await client.post(
                    "/chat.postMessage",
                    json={"channel": channel, "text": message.content},
                )
                data: dict[str, Any] = resp.json()
                if not data.get("ok", False):
                    error = data.get("error", "unknown")
                    msg = f"Slack API error: {error}"
                    raise MessageDeliveryError(msg)
            except httpx.HTTPError as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.record_exception(exc)
                collector.increment(
                    "integration_messages_sent_total",
                    labels={"channel": "slack", "status": "error"},
                )
                collector.observe("integration_message_send_duration_seconds", duration_ms / 1000)
                logger.warning(
                    "Slack send failed: recipient=%s (%.1fms)",
                    channel,
                    duration_ms,
                    extra={
                        "integration_data": {
                            "event": "message_send_failed",
                            "channel": "slack",
                            "recipient": channel,
                            "duration_ms": round(duration_ms, 1),
                            "error": str(exc),
                        }
                    },
                )
                msg = f"Failed to send Slack message: {exc}"
                raise MessageDeliveryError(msg) from exc
            else:
                duration_ms = (time.monotonic() - start) * 1000
                collector.increment(
                    "integration_messages_sent_total",
                    labels={"channel": "slack", "status": "success"},
                )
                collector.observe("integration_message_send_duration_seconds", duration_ms / 1000)
                span.set_attribute("integration.duration_ms", duration_ms)
                logger.info(
                    "Slack message sent: recipient=%s (%.1fms)",
                    channel,
                    duration_ms,
                    extra={
                        "integration_data": {
                            "event": "message_sent",
                            "channel": "slack",
                            "recipient": channel,
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
                return True

    async def process_incoming(self, raw_event: dict[str, object]) -> IncomingMessage | None:
        """Parse a Slack event payload into an ``IncomingMessage``.

        Expects the ``event`` key from the Slack Events API envelope.
        Ignores bot messages (``bot_id`` present).
        """
        collector = get_metrics_collector()
        event: dict[str, Any] = raw_event.get("event", raw_event)  # type: ignore[assignment]

        # Ignore bot messages
        if event.get("bot_id"):
            collector.increment(
                "integration_messages_ignored_total",
                labels={"channel": "slack", "reason": "bot_message"},
            )
            return None

        text = str(event.get("text", ""))
        if not text:
            collector.increment(
                "integration_messages_ignored_total",
                labels={"channel": "slack", "reason": "empty_text"},
            )
            return None

        collector.increment(
            "integration_messages_received_total",
            labels={"channel": "slack"},
        )
        sender = str(event.get("user", ""))
        logger.info(
            "Slack message received: sender=%s, channel=%s",
            sender,
            str(event.get("channel", "")),
            extra={
                "integration_data": {
                    "event": "message_received",
                    "channel": "slack",
                    "sender": sender,
                    "channel_id": str(event.get("channel", "")),
                }
            },
        )

        return IncomingMessage(
            channel="slack",
            sender=sender,
            content=text,
            metadata={
                "channel_id": str(event.get("channel", "")),
                "team_id": str(raw_event.get("team_id", "")),
            },
            timestamp=str(event.get("ts", "")),
            conversation_id=str(event.get("channel", "")),
        )

    def verify_signature(self, timestamp: str, body: str, signature: str) -> bool:
        """Verify a Slack request signature.

        Parameters
        ----------
        timestamp:
            ``X-Slack-Request-Timestamp`` header value.
        body:
            Raw request body as a string.
        signature:
            ``X-Slack-Signature`` header value.

        """
        if not self._signing_secret:
            return False

        # Reject requests older than 5 minutes
        if abs(time.time() - float(timestamp)) > 300:  # noqa: PLR2004
            return False

        sig_basestring = f"v0:{timestamp}:{body}"
        computed = (
            "v0="
            + hmac.new(
                self._signing_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        return hmac.compare_digest(computed, signature)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, raising if not started."""
        if self._client is None:
            msg = "SlackIntegration has not been started. Call start() first."
            raise ChannelConnectionError(msg)
        return self._client
