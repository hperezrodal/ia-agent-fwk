"""Email channel integration.

Uses ``aiosmtplib`` for async SMTP sending and optionally
``aioimaplib`` for async IMAP polling.  Both libraries are
optional dependencies (``pip install ia_agent_fwk[email]`` or
``pip install ia_agent_fwk[calendar]``).

Incoming emails can arrive via a webhook (e.g. SendGrid/Mailgun
inbound parse) or via IMAP polling using ``poll_inbox()``.
"""

from __future__ import annotations

import contextlib
import email as email_stdlib
import email.policy
import logging
import time
from email.message import EmailMessage
from typing import Any

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


def _has_aiosmtplib() -> bool:
    """Check whether ``aiosmtplib`` is importable."""
    try:
        import aiosmtplib  # noqa: F401, PLC0415

    except ImportError:
        return False
    else:
        return True


def _has_aioimaplib() -> bool:
    """Check whether ``aioimaplib`` is importable."""
    try:
        import aioimaplib  # noqa: F401, PLC0415

    except ImportError:
        return False
    else:
        return True


class EmailIntegration(ChannelIntegration):
    """Email channel integration via async SMTP and optional IMAP polling."""

    def __init__(  # noqa: PLR0913
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        from_address: str = "",
        username: str = "",
        password: str = "",
        *,
        use_tls: bool = True,
        imap_host: str = "imap.gmail.com",
        imap_port: int = 993,
        imap_username: str = "",
        imap_password: str = "",
        imap_use_ssl: bool = True,
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._from_address = from_address
        self._username = username
        self._password = password
        self._use_tls = use_tls
        self._imap_host = imap_host
        self._imap_port = imap_port
        self._imap_username = imap_username or username
        self._imap_password = imap_password or password
        self._imap_use_ssl = imap_use_ssl

    @property
    def channel_type(self) -> str:
        return "email"

    # ------------------------------------------------------------------
    # SMTP sending
    # ------------------------------------------------------------------

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send an email via SMTP."""
        collector = get_metrics_collector()
        if not message.recipient:
            msg = "Email recipient is required"
            raise MessageDeliveryError(msg)

        if not _has_aiosmtplib():
            msg = "aiosmtplib is required for email sending. Install with: pip install aiosmtplib"
            raise ChannelConnectionError(msg)

        import aiosmtplib  # noqa: PLC0415

        email_msg = EmailMessage()
        email_msg["From"] = self._from_address
        email_msg["To"] = message.recipient
        email_msg["Subject"] = message.metadata.get("subject", "Agent Response")
        email_msg.set_content(message.content)

        start = time.monotonic()
        with _tracer.start_as_current_span(
            "integration.send_message",
            attributes={"integration.channel": "email"},
        ) as span:
            try:
                await aiosmtplib.send(
                    email_msg,
                    hostname=self._smtp_host,
                    port=self._smtp_port,
                    username=self._username or None,
                    password=self._password or None,
                    start_tls=self._use_tls,
                )
            except aiosmtplib.SMTPException as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.record_exception(exc)
                collector.increment(
                    "integration_messages_sent_total",
                    labels={"channel": "email", "status": "error"},
                )
                collector.observe("integration_message_send_duration_seconds", duration_ms / 1000)
                logger.warning(
                    "Email send failed: recipient=%s (%.1fms)",
                    message.recipient,
                    duration_ms,
                    extra={
                        "integration_data": {
                            "event": "message_send_failed",
                            "channel": "email",
                            "recipient": message.recipient,
                            "duration_ms": round(duration_ms, 1),
                            "error": str(exc),
                        }
                    },
                )
                msg = f"Failed to send email: {exc}"
                raise MessageDeliveryError(msg) from exc
            else:
                duration_ms = (time.monotonic() - start) * 1000
                collector.increment(
                    "integration_messages_sent_total",
                    labels={"channel": "email", "status": "success"},
                )
                collector.observe("integration_message_send_duration_seconds", duration_ms / 1000)
                span.set_attribute("integration.duration_ms", duration_ms)
                logger.info(
                    "Email sent: recipient=%s (%.1fms)",
                    message.recipient,
                    duration_ms,
                    extra={
                        "integration_data": {
                            "event": "message_sent",
                            "channel": "email",
                            "recipient": message.recipient,
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
                return True

    # ------------------------------------------------------------------
    # Webhook-based incoming
    # ------------------------------------------------------------------

    async def process_incoming(self, raw_event: dict[str, object]) -> IncomingMessage | None:
        """Parse an inbound email webhook payload.

        Expects a JSON body with at least ``from``, ``text``, and
        optionally ``subject``, ``to``, ``timestamp``.
        """
        collector = get_metrics_collector()
        sender = str(raw_event.get("from", ""))
        text = str(raw_event.get("text", ""))

        if not sender or not text:
            collector.increment(
                "integration_messages_ignored_total",
                labels={"channel": "email", "reason": "missing_fields"},
            )
            return None

        collector.increment(
            "integration_messages_received_total",
            labels={"channel": "email"},
        )
        logger.info(
            "Email received: sender=%s",
            sender,
            extra={
                "integration_data": {
                    "event": "message_received",
                    "channel": "email",
                    "sender": sender,
                }
            },
        )

        metadata: dict[str, str] = {}
        subject: Any = raw_event.get("subject")
        if subject is not None:
            metadata["subject"] = str(subject)
        to: Any = raw_event.get("to")
        if to is not None:
            metadata["to"] = str(to)

        return IncomingMessage(
            channel="email",
            sender=sender,
            content=text,
            metadata=metadata,
            timestamp=str(raw_event.get("timestamp", "")),
            conversation_id=str(raw_event.get("message_id", "")),
        )

    # ------------------------------------------------------------------
    # IMAP polling
    # ------------------------------------------------------------------

    async def poll_inbox(  # noqa: PLR0915
        self,
        folder: str = "INBOX",
        criteria: str = "UNSEEN",
    ) -> list[dict[str, object]]:
        """Poll IMAP inbox for emails matching *criteria*.

        Returns a list of dicts with keys: ``uid``, ``message_id``,
        ``subject``, ``from``, ``to``, ``date``, ``text``, ``html``.

        Requires the ``aioimaplib`` optional dependency.
        """
        if not _has_aioimaplib():
            msg = "aioimaplib is required for IMAP polling. Install with: pip install ia-agent-fwk[calendar]"
            raise ChannelConnectionError(msg)

        import aioimaplib  # noqa: PLC0415

        collector = get_metrics_collector()
        results: list[dict[str, object]] = []

        start = time.monotonic()
        with _tracer.start_as_current_span(
            "integration.poll_inbox",
            attributes={"integration.channel": "email", "imap.folder": folder},
        ) as span:
            imap: aioimaplib.IMAP4_SSL | None = None
            try:
                imap = aioimaplib.IMAP4_SSL(host=self._imap_host, port=self._imap_port)
                await imap.wait_hello_from_server()
                await imap.login(self._imap_username, self._imap_password)
                await imap.select(folder)

                response = await imap.search(criteria)
                if response.result != "OK" or not response.lines:
                    span.set_attribute("imap.emails_found", 0)
                    return results

                seq_line = response.lines[0]
                seq_nums = seq_line.split() if isinstance(seq_line, str) else seq_line.decode().split()
                span.set_attribute("imap.emails_found", len(seq_nums))

                for seq_val in seq_nums:
                    seq_str = seq_val if isinstance(seq_val, str) else seq_val.decode()
                    fetch_resp = await imap.fetch(seq_str, "(UID RFC822)")
                    if fetch_resp.result != "OK":
                        continue
                    raw_bytes = self._extract_rfc822_bytes(fetch_resp.lines)
                    if raw_bytes is None:
                        continue
                    parsed = self._parse_raw_email(raw_bytes)
                    uid_val_extracted = self._extract_uid(fetch_resp.lines)
                    parsed["uid"] = uid_val_extracted or seq_str
                    results.append(parsed)

            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.record_exception(exc)
                collector.increment(
                    "integration_imap_poll_errors_total",
                    labels={"channel": "email"},
                )
                logger.warning(
                    "IMAP poll failed (%.1fms): %s",
                    duration_ms,
                    exc,
                    extra={
                        "integration_data": {
                            "event": "imap_poll_failed",
                            "channel": "email",
                            "error": str(exc),
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
                err_msg = f"IMAP poll failed: {exc}"
                raise ChannelConnectionError(err_msg) from exc
            else:
                duration_ms = (time.monotonic() - start) * 1000
                collector.increment(
                    "integration_imap_poll_total",
                    labels={"channel": "email"},
                )
                span.set_attribute("integration.duration_ms", duration_ms)
                logger.info(
                    "IMAP poll: %d emails fetched (%.1fms)",
                    len(results),
                    duration_ms,
                    extra={
                        "integration_data": {
                            "event": "imap_poll_complete",
                            "channel": "email",
                            "emails_fetched": len(results),
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
            finally:
                if imap is not None:
                    try:
                        await imap.logout()
                    except Exception:  # noqa: BLE001
                        logger.debug("IMAP logout failed (non-critical)")

        return results

    async def mark_as_read(self, message_uid: str, folder: str = "INBOX") -> None:
        r"""Mark a message as read (set ``\Seen`` flag) via IMAP."""
        if not _has_aioimaplib():
            msg = "aioimaplib is required for IMAP operations. Install with: pip install ia-agent-fwk[calendar]"
            raise ChannelConnectionError(msg)

        import aioimaplib  # noqa: PLC0415

        imap: aioimaplib.IMAP4_SSL | None = None
        try:
            imap = aioimaplib.IMAP4_SSL(host=self._imap_host, port=self._imap_port)
            await imap.wait_hello_from_server()
            await imap.login(self._imap_username, self._imap_password)
            await imap.select(folder)
            await imap.uid("store", message_uid, "+FLAGS", "(\\Seen)")
        except Exception as exc:
            logger.warning("IMAP mark_as_read failed for uid=%s: %s", message_uid, exc)
            err_msg = f"IMAP mark_as_read failed: {exc}"
            raise ChannelConnectionError(err_msg) from exc
        finally:
            if imap is not None:
                with contextlib.suppress(Exception):
                    await imap.logout()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_rfc822_bytes(lines: list[Any]) -> bytes | None:
        """Extract the raw RFC822 bytes from an IMAP FETCH response."""
        for line in lines:
            if isinstance(line, (bytes, bytearray)) and len(line) > 50:  # noqa: PLR2004
                return bytes(line)
        return None

    @staticmethod
    def _extract_uid(lines: list[Any]) -> str | None:
        """Extract UID from a FETCH response line like ``* 1 FETCH (UID 42 RFC822 ...``."""
        import re  # noqa: PLC0415

        for line in lines:
            text = (
                line
                if isinstance(line, str)
                else (line.decode("utf-8", errors="replace") if isinstance(line, (bytes, bytearray)) else str(line))
            )
            match = re.search(r"UID\s+(\d+)", text)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _parse_raw_email(raw_bytes: bytes) -> dict[str, object]:
        """Parse raw email bytes into a structured dict."""
        msg = email_stdlib.message_from_bytes(raw_bytes, policy=email_stdlib.policy.default)

        text_body = ""
        html_body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain" and not text_body:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        text_body = payload.decode("utf-8", errors="replace")
                elif content_type == "text/html" and not html_body:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        html_body = payload.decode("utf-8", errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                decoded = payload.decode("utf-8", errors="replace")
                if msg.get_content_type() == "text/html":
                    html_body = decoded
                else:
                    text_body = decoded

        return {
            "message_id": msg.get("Message-ID", ""),
            "subject": msg.get("Subject", ""),
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "date": msg.get("Date", ""),
            "text": text_body,
            "html": html_body,
            "in_reply_to": msg.get("In-Reply-To", ""),
            "references": msg.get("References", ""),
        }

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check SMTP connectivity (returns True if aiosmtplib is available)."""
        healthy = _has_aiosmtplib()
        collector = get_metrics_collector()
        collector.increment(
            "integration_health_checks_total",
            labels={"channel": "email", "status": "healthy" if healthy else "unhealthy"},
        )
        return healthy
