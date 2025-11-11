"""Tests for the HTTP request built-in tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.builtin.http_request import (
    HttpRequestInput,
    HttpRequestOutput,
    HttpRequestTool,
    _is_private_ip,
    _redact_headers,
)
from ia_agent_fwk.tools.exceptions import ToolExecutionError


@pytest.fixture
def ctx():
    return ToolContext(execution_id="test-http", timeout=10.0)


@pytest.fixture
def tool():
    return HttpRequestTool(allowed_domains=["example.com", "api.example.com"])


@pytest.fixture
def unrestricted_tool():
    return HttpRequestTool()


class TestDomainAllowlist:
    async def test_allowed_domain_passes(self, tool, ctx):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.headers = {"content-type": "text/plain"}

        with patch("ia_agent_fwk.tools.builtin.http_request._is_private_ip", return_value=False):
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await tool.execute(
                    HttpRequestInput(url="https://example.com/api"),
                    ctx,
                )
        assert isinstance(result, HttpRequestOutput)
        assert result.status_code == 200

    async def test_non_allowed_domain_rejected(self, tool, ctx):
        with pytest.raises(ToolExecutionError, match="not in the allowed domains"):
            await tool.execute(
                HttpRequestInput(url="https://evil.com/api"),
                ctx,
            )


class TestSSRFPrevention:
    def test_localhost_is_private(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[
                (2, 1, 6, "", ("127.0.0.1", 0)),
            ],
        ):
            assert _is_private_ip("localhost") is True

    def test_private_ip_10(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[
                (2, 1, 6, "", ("10.0.0.1", 0)),
            ],
        ):
            assert _is_private_ip("internal.example.com") is True

    def test_private_ip_172(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[
                (2, 1, 6, "", ("172.16.0.1", 0)),
            ],
        ):
            assert _is_private_ip("internal.example.com") is True

    def test_private_ip_192(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[
                (2, 1, 6, "", ("192.168.1.1", 0)),
            ],
        ):
            assert _is_private_ip("internal.example.com") is True

    def test_public_ip_not_private(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[
                (2, 1, 6, "", ("8.8.8.8", 0)),
            ],
        ):
            assert _is_private_ip("dns.google.com") is False

    async def test_private_ip_blocked(self, unrestricted_tool, ctx):
        with (
            patch("ia_agent_fwk.tools.builtin.http_request._is_private_ip", return_value=True),
            pytest.raises(ToolExecutionError, match="private/internal IP"),
        ):
            await unrestricted_tool.execute(
                HttpRequestInput(url="http://localhost:8080/admin"),
                ctx,
            )


class TestHTTPMethods:
    async def test_unsupported_method_rejected(self, tool, ctx):
        with pytest.raises(ToolExecutionError, match="Unsupported HTTP method"):
            await tool.execute(
                HttpRequestInput(url="https://example.com/api", method="PATCH"),
                ctx,
            )


class TestHeaderRedaction:
    def test_redacts_authorization(self):
        headers = {"Authorization": "Bearer token123", "Content-Type": "application/json"}
        redacted = _redact_headers(headers)
        assert redacted["Authorization"] == "***REDACTED***"
        assert redacted["Content-Type"] == "application/json"

    def test_redacts_cookie(self):
        headers = {"Cookie": "session=abc123"}
        redacted = _redact_headers(headers)
        assert redacted["Cookie"] == "***REDACTED***"


class TestInvalidURL:
    async def test_empty_host(self, tool, ctx):
        with pytest.raises(ToolExecutionError):
            await tool.execute(
                HttpRequestInput(url="not-a-url"),
                ctx,
            )


class TestToolProperties:
    def test_name(self, tool):
        assert tool.name == "http_request"

    def test_tags(self, tool):
        assert "http" in tool.tags
        assert "builtin" in tool.tags

    def test_description(self, tool):
        assert "HTTP" in tool.description
