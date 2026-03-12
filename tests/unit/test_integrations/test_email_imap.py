"""Tests for EmailIntegration IMAP polling and mark_as_read."""

from __future__ import annotations

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.integrations.email_channel import EmailIntegration
from ia_agent_fwk.integrations.exceptions import ChannelConnectionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_email(  # noqa: PLR0913
    subject: str = "Test Subject",
    from_addr: str = "sender@test.com",
    to_addr: str = "inbox@test.com",
    text_body: str = "Hello, world!",
    html_body: str = "",
    message_id: str = "<msg-001@test.com>",
) -> bytes:
    """Build a raw RFC822 email as bytes."""
    if html_body:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
    else:
        msg = MIMEText(text_body, "plain")

    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Message-ID"] = message_id
    msg["Date"] = "Mon, 11 Mar 2026 10:00:00 +0000"
    return msg.as_bytes()


def _make_mock_imap(
    uids: list[str] | None = None,
    raw_emails: dict[str, bytes] | None = None,
    search_result: str = "OK",
):
    """Create a mock aioimaplib.IMAP4_SSL with configurable responses."""
    mock_imap = MagicMock()
    mock_imap.wait_hello_from_server = AsyncMock()
    mock_imap.login = AsyncMock()
    mock_imap.select = AsyncMock()
    mock_imap.logout = AsyncMock()

    # search response
    search_resp = MagicMock()
    search_resp.result = search_result
    if uids:
        search_resp.lines = [" ".join(uids)]
    else:
        search_resp.lines = []
    mock_imap.search = AsyncMock(return_value=search_resp)

    # fetch response (sequence number based)
    def fetch_handler(seq_num, *_args):
        if raw_emails and seq_num in raw_emails:
            resp = MagicMock()
            resp.result = "OK"
            resp.lines = [f"* {seq_num} FETCH (UID {seq_num} RFC822 {{...}})".encode(), raw_emails[seq_num], b")"]
            return resp
        resp = MagicMock()
        resp.result = "NO"
        resp.lines = []
        return resp

    mock_imap.fetch = AsyncMock(side_effect=fetch_handler)

    # uid store response (for mark_as_read)
    def uid_handler(cmd, _uid_val, *_args):
        if cmd == "store":
            resp = MagicMock()
            resp.result = "OK"
            return resp
        resp = MagicMock()
        resp.result = "NO"
        resp.lines = []
        return resp

    mock_imap.uid = AsyncMock(side_effect=uid_handler)

    return mock_imap


def _make_mock_aioimaplib(mock_imap):
    """Create a mock aioimaplib module wrapping the given mock IMAP instance."""
    mock_mod = MagicMock(spec=ModuleType)
    mock_mod.IMAP4_SSL = MagicMock(return_value=mock_imap)
    return mock_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmailImapInit:
    def test_imap_defaults(self):
        ei = EmailIntegration()
        assert ei._imap_host == "imap.gmail.com"
        assert ei._imap_port == 993
        assert ei._imap_use_ssl is True

    def test_imap_username_falls_back_to_smtp_username(self):
        ei = EmailIntegration(username="smtp@test.com")
        assert ei._imap_username == "smtp@test.com"

    def test_imap_explicit_credentials(self):
        ei = EmailIntegration(
            imap_host="imap.custom.com",
            imap_port=143,
            imap_username="imap@custom.com",
            imap_password="imap-pass",  # noqa: S106
            imap_use_ssl=False,
        )
        assert ei._imap_host == "imap.custom.com"
        assert ei._imap_port == 143
        assert ei._imap_username == "imap@custom.com"
        assert ei._imap_password == "imap-pass"  # noqa: S105
        assert ei._imap_use_ssl is False


@pytest.mark.unit
class TestPollInbox:
    async def test_no_aioimaplib_raises(self):
        ei = EmailIntegration()
        with (
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=False,
            ),
            pytest.raises(ChannelConnectionError, match="aioimaplib is required"),
        ):
            await ei.poll_inbox()

    async def test_poll_empty_inbox(self):
        ei = EmailIntegration()
        mock_imap = _make_mock_imap(uids=[], search_result="OK")
        mock_mod = _make_mock_aioimaplib(mock_imap)

        import sys

        with (
            patch.dict(sys.modules, {"aioimaplib": mock_mod}),
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=True,
            ),
        ):
            results = await ei.poll_inbox()

        assert results == []
        mock_imap.logout.assert_called_once()

    async def test_poll_with_emails(self):
        raw_email = _make_raw_email(
            subject="Meeting Invite",
            from_addr="alice@test.com",
            text_body="Please join the standup at 10am.",
            message_id="<msg-001@test.com>",
        )
        mock_imap = _make_mock_imap(
            uids=["1"],
            raw_emails={"1": raw_email},
        )
        mock_mod = _make_mock_aioimaplib(mock_imap)

        ei = EmailIntegration()

        import sys

        with (
            patch.dict(sys.modules, {"aioimaplib": mock_mod}),
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=True,
            ),
        ):
            results = await ei.poll_inbox()

        assert len(results) == 1
        assert results[0]["subject"] == "Meeting Invite"
        assert results[0]["from"] == "alice@test.com"
        assert "standup" in str(results[0]["text"])
        assert results[0]["uid"] == "1"

    async def test_poll_multiple_emails(self):
        raw1 = _make_raw_email(subject="Email 1", message_id="<msg-001>")
        raw2 = _make_raw_email(subject="Email 2", message_id="<msg-002>")
        mock_imap = _make_mock_imap(
            uids=["1", "2"],
            raw_emails={"1": raw1, "2": raw2},
        )
        mock_mod = _make_mock_aioimaplib(mock_imap)

        ei = EmailIntegration()

        import sys

        with (
            patch.dict(sys.modules, {"aioimaplib": mock_mod}),
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=True,
            ),
        ):
            results = await ei.poll_inbox()

        assert len(results) == 2

    async def test_poll_search_not_ok(self):
        mock_imap = _make_mock_imap(uids=[], search_result="NO")
        mock_mod = _make_mock_aioimaplib(mock_imap)

        ei = EmailIntegration()

        import sys

        with (
            patch.dict(sys.modules, {"aioimaplib": mock_mod}),
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=True,
            ),
        ):
            results = await ei.poll_inbox()

        assert results == []

    async def test_poll_connection_error(self):
        ei = EmailIntegration()
        mock_imap = MagicMock()
        mock_imap.wait_hello_from_server = AsyncMock(side_effect=OSError("Connection refused"))
        mock_imap.logout = AsyncMock()

        mock_mod = MagicMock(spec=ModuleType)
        mock_mod.IMAP4_SSL = MagicMock(return_value=mock_imap)

        import sys

        with (
            patch.dict(sys.modules, {"aioimaplib": mock_mod}),
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=True,
            ),
            pytest.raises(ChannelConnectionError, match="IMAP poll failed"),
        ):
            await ei.poll_inbox()

    async def test_poll_skips_unfetchable_uid(self):
        # Only uid "1" has a raw email; uid "2" will get fetch result "NO"
        raw1 = _make_raw_email(subject="Good Email", message_id="<msg-001>")
        mock_imap = _make_mock_imap(
            uids=["1", "2"],
            raw_emails={"1": raw1},
        )
        mock_mod = _make_mock_aioimaplib(mock_imap)

        ei = EmailIntegration()

        import sys

        with (
            patch.dict(sys.modules, {"aioimaplib": mock_mod}),
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=True,
            ),
        ):
            results = await ei.poll_inbox()

        # Only the fetchable email should be returned
        assert len(results) == 1
        assert results[0]["subject"] == "Good Email"


@pytest.mark.unit
class TestMarkAsRead:
    async def test_no_aioimaplib_raises(self):
        ei = EmailIntegration()
        with (
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=False,
            ),
            pytest.raises(ChannelConnectionError, match="aioimaplib is required"),
        ):
            await ei.mark_as_read("1")

    async def test_mark_as_read_success(self):
        mock_imap = _make_mock_imap()
        mock_mod = _make_mock_aioimaplib(mock_imap)

        ei = EmailIntegration()

        import sys

        with (
            patch.dict(sys.modules, {"aioimaplib": mock_mod}),
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=True,
            ),
        ):
            await ei.mark_as_read("42")

        # Verify the store command was called with \\Seen
        mock_imap.uid.assert_called_with("store", "42", "+FLAGS", "(\\Seen)")

    async def test_mark_as_read_error(self):
        mock_imap = MagicMock()
        mock_imap.wait_hello_from_server = AsyncMock()
        mock_imap.login = AsyncMock()
        mock_imap.select = AsyncMock()
        mock_imap.uid = AsyncMock(side_effect=OSError("IMAP error"))
        mock_imap.logout = AsyncMock()

        mock_mod = MagicMock(spec=ModuleType)
        mock_mod.IMAP4_SSL = MagicMock(return_value=mock_imap)

        ei = EmailIntegration()

        import sys

        with (
            patch.dict(sys.modules, {"aioimaplib": mock_mod}),
            patch(
                "ia_agent_fwk.integrations.email_channel._has_aioimaplib",
                return_value=True,
            ),
            pytest.raises(ChannelConnectionError, match="mark_as_read failed"),
        ):
            await ei.mark_as_read("42")


@pytest.mark.unit
class TestParseRawEmail:
    def test_plain_text_email(self):
        raw = _make_raw_email(
            subject="Test",
            from_addr="alice@test.com",
            to_addr="bob@test.com",
            text_body="Hello world",
            message_id="<msg-001>",
        )
        result = EmailIntegration._parse_raw_email(raw)
        assert result["subject"] == "Test"
        assert result["from"] == "alice@test.com"
        assert result["to"] == "bob@test.com"
        assert "Hello world" in result["text"]
        assert result["message_id"] == "<msg-001>"

    def test_multipart_email(self):
        raw = _make_raw_email(
            subject="Multi",
            text_body="Plain version",
            html_body="<p>HTML version</p>",
        )
        result = EmailIntegration._parse_raw_email(raw)
        assert "Plain version" in result["text"]
        assert "<p>HTML version</p>" in result["html"]

    def test_html_only_email(self):
        msg = MIMEText("<h1>HTML Only</h1>", "html")
        msg["Subject"] = "HTML"
        msg["From"] = "test@test.com"
        msg["Message-ID"] = "<msg-html>"
        raw = msg.as_bytes()
        result = EmailIntegration._parse_raw_email(raw)
        assert result["text"] == ""
        assert "<h1>HTML Only</h1>" in result["html"]


@pytest.mark.unit
class TestExtractRfc822Bytes:
    def test_extracts_large_bytes(self):
        lines = [
            b"* 1 FETCH (RFC822 {500}",
            b"X" * 100,  # This is the RFC822 content (> 50 bytes)
            b")",
        ]
        result = EmailIntegration._extract_rfc822_bytes(lines)
        assert result == b"X" * 100

    def test_returns_none_for_small_lines(self):
        lines = [b"small", b"tiny"]
        result = EmailIntegration._extract_rfc822_bytes(lines)
        assert result is None

    def test_returns_none_for_empty(self):
        result = EmailIntegration._extract_rfc822_bytes([])
        assert result is None
