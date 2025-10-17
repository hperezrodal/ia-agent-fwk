"""Configurable exponential-backoff retry logic for LLM calls.

The retry wrapper classifies errors as retryable or non-retryable and
retries transient failures with exponential backoff and jitter.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ia_agent_fwk.config.settings import RetrySettings

from ia_agent_fwk.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes considered retryable.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# HTTP status codes that must NOT be retried.
_NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404})


def _extract_status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception."""
    # openai, anthropic, httpx all expose ``status_code``
    code: Any = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "status", None)
    if isinstance(code, int):
        return code
    return None


def _extract_retry_after(exc: BaseException) -> float | None:
    """Extract a ``Retry-After`` header value (seconds) if available."""
    headers: Any = getattr(exc, "response", None)
    if headers is not None:
        headers = getattr(headers, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is not None:
        retry_after: str | None = None
        if hasattr(headers, "get"):
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
    return None


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if *exc* represents a transient/retryable error."""
    if isinstance(exc, LLMAuthenticationError):
        return False

    # Connection errors / timeouts are always retryable.
    if isinstance(exc, (ConnectionError, TimeoutError, LLMTimeoutError, LLMRateLimitError)):
        return True

    # OSError subclasses (e.g. ConnectionRefusedError) are retryable.
    if isinstance(exc, OSError):
        return True

    status = _extract_status_code(exc)
    if status is not None:
        if status in _NON_RETRYABLE_STATUS_CODES:
            return False
        if status in _RETRYABLE_STATUS_CODES:
            return True

    return False


def _compute_backoff(attempt: int, settings: RetrySettings) -> float:
    """Compute backoff delay with full jitter."""
    base_delay = min(settings.backoff_base**attempt, settings.backoff_max)
    return random.uniform(0, base_delay)  # noqa: S311


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    retry_settings: RetrySettings,
    **kwargs: Any,
) -> T:
    """Execute *fn* with exponential-backoff retry.

    Parameters
    ----------
    fn:
        The async callable to invoke.
    retry_settings:
        Retry policy (``max_attempts``, ``backoff_base``, ``backoff_max``).
    *args:
        Positional arguments forwarded to *fn*.
    **kwargs:
        Keyword arguments forwarded to *fn*.

    Returns
    -------
    T
        The return value of *fn* on success.

    Raises
    ------
    LLMProviderError
        After all attempts are exhausted, the last exception is re-raised
        (wrapped in ``LLMProviderError`` if it was not already one).

    """
    last_exc: BaseException | None = None

    for attempt in range(retry_settings.max_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

            if not is_retryable(exc):
                raise

            remaining = retry_settings.max_attempts - attempt - 1
            if remaining <= 0:
                break

            # Respect Retry-After header when available.
            retry_after = _extract_retry_after(exc)
            delay = retry_after if retry_after is not None else _compute_backoff(attempt, retry_settings)
            delay = min(delay, retry_settings.backoff_max)

            collector = get_metrics_collector()
            collector.increment(
                "llm_retry_attempts_total",
                labels={"attempt": str(attempt + 1), "error_type": type(exc).__name__},
            )
            collector.observe("llm_retry_backoff_seconds", delay)
            logger.warning(
                "Retry attempt %d/%d after %.2fs - %s: %s",
                attempt + 1,
                retry_settings.max_attempts,
                delay,
                type(exc).__name__,
                exc,
                extra={
                    "llm_data": {
                        "event": "retry_attempt",
                        "attempt": attempt + 1,
                        "max_attempts": retry_settings.max_attempts,
                        "delay_seconds": round(delay, 2),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                },
            )
            await asyncio.sleep(delay)

    # All attempts exhausted.
    assert last_exc is not None  # noqa: S101
    collector = get_metrics_collector()
    collector.increment(
        "llm_retry_exhausted_total",
        labels={"error_type": type(last_exc).__name__},
    )
    logger.error(
        "All %d retry attempts exhausted - %s: %s",
        retry_settings.max_attempts,
        type(last_exc).__name__,
        last_exc,
        extra={
            "llm_data": {
                "event": "retry_exhausted",
                "max_attempts": retry_settings.max_attempts,
                "error_type": type(last_exc).__name__,
                "error": str(last_exc),
            }
        },
    )
    if isinstance(last_exc, LLMProviderError):
        raise last_exc
    raise LLMProviderError(str(last_exc)) from last_exc
