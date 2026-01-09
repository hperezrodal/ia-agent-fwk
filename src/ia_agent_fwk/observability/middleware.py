"""Observability middleware for FastAPI.

``ObservabilityMiddleware`` adds per-request tracing spans and records
HTTP request metrics. It integrates with the existing
``RequestIdMiddleware`` which runs first (outermost).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

_tracer = get_tracer(__name__)


class ObservabilityMiddleware:
    """ASGI middleware that instruments HTTP requests with tracing and metrics.

    For every HTTP request this middleware:

    1. Starts an OTel span named ``HTTP {method} {path}``.
    2. Records ``http_requests_total`` counter (with method and status labels).
    3. Records ``http_request_duration_seconds`` histogram.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process an ASGI request."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method: str = scope.get("method", "UNKNOWN")
        path: str = scope.get("path", "/")
        collector = get_metrics_collector()

        status_code = 500  # default in case we never see a response start

        start_time = time.monotonic()

        with _tracer.start_as_current_span(f"HTTP {method} {path}") as span:
            span.set_attribute("http.method", method)
            span.set_attribute("http.target", path)

            async def send_wrapper(message: Message) -> None:
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message.get("status", 500)
                    span.set_attribute("http.status_code", status_code)
                await send(message)

            try:
                await self._app(scope, receive, send_wrapper)
            finally:
                duration = time.monotonic() - start_time
                collector.increment(
                    "http_requests_total",
                    labels={"method": method, "status": str(status_code)},
                )
                collector.observe("http_request_duration_seconds", duration)
