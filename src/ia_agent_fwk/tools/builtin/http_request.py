"""HTTP request built-in tool with domain sandboxing and SSRF prevention.

Makes async HTTP requests using ``httpx``. Enforces domain allowlists
and blocks requests to private IP ranges.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.tools.base import Tool, ToolContext
from ia_agent_fwk.tools.exceptions import ToolExecutionError

# Default maximum response body size: 5 MB
_DEFAULT_MAX_RESPONSE_SIZE = 5 * 1024 * 1024

# Sensitive headers to redact from error messages
_SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie", "x-api-key"}


def _is_private_ip(host: str) -> bool:
    """Check whether a hostname resolves to a private IP address.

    Uses fail-closed behavior: if DNS resolution fails, the host is
    treated as private (blocked) to prevent SSRF bypass via
    unresolvable hostnames.
    """
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        # Fail closed: unresolvable hosts are treated as private/blocked
        return True

    if not infos:
        return True

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact sensitive header values."""
    return {k: ("***REDACTED***" if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()}


class HttpRequestInput(BaseModel):
    """Input schema for the HTTP request tool."""

    model_config = ConfigDict(frozen=True)

    url: str
    method: str = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None


class HttpRequestOutput(BaseModel):
    """Output schema for the HTTP request tool."""

    model_config = ConfigDict(frozen=True)

    status_code: int
    body: str
    headers: dict[str, str]


class HttpRequestTool(Tool):
    """Make async HTTP requests with domain sandboxing and SSRF prevention.

    Parameters
    ----------
    allowed_domains:
        List of allowed domains. If empty, all (non-private) domains are allowed.
    max_response_size:
        Maximum response body size in bytes. Default: 5 MB.

    """

    def __init__(
        self,
        allowed_domains: list[str] | None = None,
        max_response_size: int = _DEFAULT_MAX_RESPONSE_SIZE,
    ) -> None:
        self._allowed_domains = allowed_domains or []
        self._max_response_size = max_response_size

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return "Make HTTP requests (GET, POST, PUT, DELETE) to allowed domains."

    @property
    def input_schema(self) -> type[BaseModel]:
        return HttpRequestInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return HttpRequestOutput

    @property
    def tags(self) -> list[str]:
        return ["http", "network", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        """Execute the HTTP request."""
        assert isinstance(validated_input, HttpRequestInput)  # noqa: S101

        url = validated_input.url
        method = validated_input.method.upper()
        headers = dict(validated_input.headers)
        body = validated_input.body

        if method not in {"GET", "POST", "PUT", "DELETE"}:
            msg = f"Unsupported HTTP method: {method}. Allowed: GET, POST, PUT, DELETE."
            raise ToolExecutionError(msg, tool_name="http_request")

        # Parse URL to get hostname
        try:
            parsed = httpx.URL(url)
            host = parsed.host
        except Exception as exc:
            msg = f"Invalid URL: {url} ({exc})"
            raise ToolExecutionError(msg, tool_name="http_request") from exc

        if not host:
            msg = f"URL has no host: {url}"
            raise ToolExecutionError(msg, tool_name="http_request")

        # Domain allowlist check
        if self._allowed_domains and host not in self._allowed_domains:
            msg = f"Domain '{host}' is not in the allowed domains list."
            raise ToolExecutionError(msg, tool_name="http_request")

        # SSRF prevention: check for private IPs
        if _is_private_ip(host):
            msg = f"Request to private/internal IP address is blocked (host: {host})."
            raise ToolExecutionError(msg, tool_name="http_request")

        # Make the request
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(context.timeout),
                follow_redirects=False,
            ) as client:
                request_kwargs: dict[str, Any] = {
                    "method": method,
                    "url": url,
                    "headers": headers,
                }
                if body is not None and method in {"POST", "PUT"}:
                    request_kwargs["content"] = body

                response = await client.request(**request_kwargs)

        except httpx.TimeoutException as exc:
            msg = f"HTTP request timed out: {exc}"
            raise ToolExecutionError(msg, tool_name="http_request") from exc
        except httpx.HTTPError as exc:
            # Redact sensitive headers from error messages
            safe_headers = _redact_headers(headers)
            msg = f"HTTP request failed: {exc} (headers: {safe_headers})"
            raise ToolExecutionError(msg, tool_name="http_request") from exc

        # Check response size
        response_body = response.text
        if len(response_body.encode()) > self._max_response_size:
            msg = f"Response body too large: exceeds {self._max_response_size} bytes."
            raise ToolExecutionError(msg, tool_name="http_request")

        response_headers = dict(response.headers)

        return HttpRequestOutput(
            status_code=response.status_code,
            body=response_body,
            headers=response_headers,
        )
