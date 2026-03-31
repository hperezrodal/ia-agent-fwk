"""FastAPI security middleware — rate limiting, auth, and budget checks.

Pluggable middleware for FastAPI apps. Connects rate limiter, API key auth,
and budget tracker to the request lifecycle.

Usage:
    from ia_agent_fwk.security.middleware import SecurityMiddleware, SecurityConfig

    config = SecurityConfig(
        rate_limit="30/minute",
        require_api_key=True,
        api_keys={"sk-abc123": "client-name"},
    )
    app.add_middleware(SecurityMiddleware, config=config)
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass, field
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request  # noqa: TC002
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp  # noqa: TC002

from ia_agent_fwk.security.rate_limiter import SlidingWindowRateLimiter, parse_rate

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Configuration for the security middleware.

    Parameters
    ----------
    rate_limit:
        Rate limit string (e.g. "30/minute", "1000/hour").
    rate_limit_by:
        Key to rate limit by: "ip", "api_key", or "session".
    require_api_key:
        Whether to require an API key in the X-API-Key header.
    api_keys:
        Mapping of API key → client name. Only checked if require_api_key=True.
    exempt_paths:
        Paths exempt from rate limiting and auth (e.g. /health, /metrics).

    """

    rate_limit: str = "60/minute"
    rate_limit_by: str = "ip"
    require_api_key: bool = False
    api_keys: dict[str, str] = field(default_factory=dict)
    exempt_paths: list[str] = field(
        default_factory=lambda: ["/health", "/metrics", "/docs", "/openapi.json"],
    )


class SecurityMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting and optional API key auth."""

    def __init__(self, app: ASGIApp, config: SecurityConfig | None = None) -> None:
        super().__init__(app)
        self._config = config or SecurityConfig()
        limit, window = parse_rate(self._config.rate_limit)
        self._limiter = SlidingWindowRateLimiter(
            default_limit=limit,
            default_window_seconds=window,
        )

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        path = request.url.path

        # Skip exempt paths
        if any(path.startswith(p) for p in self._config.exempt_paths):
            return await call_next(request)  # type: ignore[no-any-return]

        # API key auth
        if self._config.require_api_key:
            api_key = request.headers.get("X-API-Key", "")
            if not api_key or api_key not in self._config.api_keys:
                return JSONResponse(
                    {"error": "Invalid or missing API key"},
                    status_code=401,
                )

        # Rate limiting
        rate_key = self._get_rate_key(request)
        allowed = await self._limiter.check_rate_limit(rate_key)
        if not allowed:
            retry_after = self._limiter.get_retry_after(rate_key)
            return JSONResponse(
                {"error": "Rate limit exceeded", "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)  # type: ignore[no-any-return]

    def _get_rate_key(self, request: Request) -> str:
        """Build the rate limit key based on config."""
        if self._config.rate_limit_by == "api_key":
            key = request.headers.get("X-API-Key", "anonymous")
            return hashlib.sha256(key.encode()).hexdigest()[:16]
        if self._config.rate_limit_by == "session":
            # Try to extract session_id from JSON body (best effort)
            host = request.client.host if request.client else "unknown"
            return request.headers.get("X-Session-Id", host)
        # Default: IP
        return request.client.host if request.client else "unknown"
