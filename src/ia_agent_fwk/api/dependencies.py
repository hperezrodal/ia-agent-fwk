"""FastAPI dependency functions for the API layer."""

from __future__ import annotations

import hmac
import os
from typing import TYPE_CHECKING, NoReturn

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from ia_agent_fwk.api.middleware import get_request_id
from ia_agent_fwk.api.models import ErrorDetail, ErrorResponse
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.security.audit import AuditLogger, hash_api_key
from ia_agent_fwk.security.exceptions import RateLimitExceededError

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import AppSettings
    from ia_agent_fwk.execution.manager import JobManager
    from ia_agent_fwk.execution.scheduler import ScheduleManager
    from ia_agent_fwk.execution.triggers import TriggerManager
    from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.security.rate_limiter import SlidingWindowRateLimiter

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_settings(request: Request) -> AppSettings:
    """Retrieve ``AppSettings`` from ``app.state``."""
    settings: AppSettings = request.app.state.settings
    return settings


async def get_memory_backend(request: Request) -> MemoryBackend:
    """Retrieve ``MemoryBackend`` from ``app.state``."""
    backend: MemoryBackend = request.app.state.memory_backend
    return backend


async def get_conversation_backend(request: Request) -> ConversationMemoryBackend:
    """Retrieve ``ConversationMemoryBackend`` from ``app.state``."""
    backend: ConversationMemoryBackend = request.app.state.conversation_backend
    return backend


async def get_job_manager(request: Request) -> JobManager:
    """Retrieve ``JobManager`` from ``app.state``."""
    manager: JobManager = request.app.state.job_manager
    return manager


async def get_schedule_manager(request: Request) -> ScheduleManager:
    """Retrieve ``ScheduleManager`` from ``app.state``."""
    manager: ScheduleManager = request.app.state.schedule_manager
    return manager


async def get_trigger_manager(request: Request) -> TriggerManager:
    """Retrieve ``TriggerManager`` from ``app.state``."""
    manager: TriggerManager = request.app.state.trigger_manager
    return manager


async def require_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Validate API key from the ``X-API-Key`` header.

    Reads valid keys from the ``IAFWK_API_KEYS`` environment variable
    (comma-separated). Uses ``hmac.compare_digest`` for constant-time
    comparison. Raises ``HTTPException(401)`` on failure. Skipped when
    ``auth.enabled`` is ``False``.
    """
    settings: AppSettings = request.app.state.settings
    if not settings.auth.enabled:
        return

    audit_logger: AuditLogger | None = getattr(request.app.state, "audit_logger", None)
    resource = str(request.url.path)

    collector = get_metrics_collector()

    if api_key is None:
        collector.increment("api_auth_total", labels={"result": "failure", "reason": "missing_key"})
        if audit_logger is not None:
            audit_logger.log_auth_failure(resource=resource, reason="missing_key")
        _raise_auth_error()

    raw_keys = os.environ.get("IAFWK_API_KEYS", "")
    valid_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not valid_keys:
        collector.increment("api_auth_total", labels={"result": "failure", "reason": "no_keys_configured"})
        if audit_logger is not None:
            audit_logger.log_auth_failure(resource=resource, reason="no_keys_configured")
        _raise_auth_error()

    for valid_key in valid_keys:
        if hmac.compare_digest(api_key, valid_key):
            collector.increment("api_auth_total", labels={"result": "success", "reason": ""})
            if audit_logger is not None:
                audit_logger.log_auth_success(api_key=api_key, resource=resource)
            return

    collector.increment("api_auth_total", labels={"result": "failure", "reason": "invalid_key"})
    if audit_logger is not None:
        audit_logger.log_auth_failure(resource=resource, reason="invalid_key")
    _raise_auth_error()


def _raise_auth_error() -> NoReturn:
    """Raise an HTTP 401 error with a standard error envelope."""
    import datetime  # noqa: PLC0415

    error_response = ErrorResponse(
        error=ErrorDetail(
            code="AUTHENTICATION_REQUIRED",
            message="Invalid or missing API key",
            request_id=get_request_id(),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017
        )
    )
    raise HTTPException(
        status_code=401,
        detail=error_response.model_dump(),
    )


async def check_rate_limit(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Check per-API-key rate limits.

    Uses the ``SlidingWindowRateLimiter`` stored on ``app.state``. If rate
    limiting is disabled or no limiter is configured, this is a no-op.
    Raises ``RateLimitExceededError`` (handled as HTTP 429) when the
    client exceeds the configured rate.
    """
    rate_limiter: SlidingWindowRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is None:
        return

    collector = get_metrics_collector()

    # Use hashed API key as the rate-limit key, or "anonymous" if no key
    key = hash_api_key(api_key) if api_key else "anonymous"
    resource = str(request.url.path)

    allowed = await rate_limiter.check_rate_limit(key)
    if not allowed:
        collector.increment("api_rate_limit_exceeded_total")
        retry_after = rate_limiter.get_retry_after(key)

        # Log the rate limit hit via audit logger if available
        audit_logger: AuditLogger | None = getattr(request.app.state, "audit_logger", None)
        if audit_logger is not None:
            audit_logger.log_rate_limit_hit(key=key, resource=resource)

        raise RateLimitExceededError(key=key, retry_after=retry_after)

    collector.increment("api_rate_limit_allowed_total")
