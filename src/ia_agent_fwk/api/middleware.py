"""Request ID middleware for the API layer.

Provides an ASGI middleware that generates a unique ``X-Request-ID`` header
for every request and propagates it via a ``ContextVar``.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="unknown")


def get_request_id() -> str:
    """Return the current request ID from the context variable."""
    return request_id_ctx.get()


class RequestIdMiddleware:
    """ASGI middleware that sets a unique request ID per request.

    Checks the incoming ``X-Request-ID`` header and uses it if present;
    otherwise generates a new UUID4. The request ID is stored in a
    ``ContextVar`` and added to the response headers.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process an ASGI request."""
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        # Extract and validate client-provided request ID from headers.
        # Only accept values that match UUID4 format to prevent log injection
        # and XSS attacks via crafted request IDs.
        headers = dict(scope.get("headers", []))
        client_id_raw = headers.get(b"x-request-id", b"").decode("utf-8", errors="replace")
        client_id = client_id_raw if _UUID4_RE.match(client_id_raw) else None
        request_id = client_id or str(uuid.uuid4())

        token = request_id_ctx.set(request_id)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw_headers: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                raw_headers.append((b"x-request-id", request_id.encode("utf-8")))
                message["headers"] = raw_headers
            await send(message)

        try:
            await self._app(scope, receive, send_with_request_id)
        finally:
            request_id_ctx.reset(token)
