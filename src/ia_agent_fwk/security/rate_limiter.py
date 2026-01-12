"""In-memory sliding-window rate limiter.

Provides a ``SlidingWindowRateLimiter`` that tracks request timestamps
per key and enforces configurable limits within a time window. No external
dependencies (Redis, etc.) are required for V1.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


def parse_rate(rate_str: str) -> tuple[int, int]:
    """Parse a rate string like ``"60/minute"`` into ``(limit, window_seconds)``.

    Supported units: ``second``, ``minute``, ``hour``, ``day``.

    Parameters
    ----------
    rate_str:
        A string in the format ``"<count>/<unit>"``.

    Returns
    -------
    tuple[int, int]:
        ``(limit, window_seconds)``

    Raises
    ------
    ValueError:
        If the rate string is malformed or uses an unknown unit.

    """
    parts = rate_str.strip().split("/")
    if len(parts) != 2:  # noqa: PLR2004
        msg = f"Invalid rate format: '{rate_str}'. Expected '<count>/<unit>'."
        raise ValueError(msg)

    try:
        limit = int(parts[0])
    except ValueError as exc:
        msg = f"Invalid rate count: '{parts[0]}'. Must be an integer."
        raise ValueError(msg) from exc

    unit = parts[1].strip().lower()
    unit_map = {
        "second": 1,
        "minute": 60,
        "hour": 3600,
        "day": 86400,
    }

    if unit not in unit_map:
        msg = f"Unknown rate unit: '{unit}'. Supported: {', '.join(unit_map)}."
        raise ValueError(msg)

    return limit, unit_map[unit]


class SlidingWindowRateLimiter:
    """In-memory sliding-window rate limiter.

    Tracks request timestamps per key in a dictionary. Old entries outside
    the window are pruned on each check.

    Parameters
    ----------
    default_limit:
        Maximum number of requests allowed per window.
    default_window_seconds:
        Window duration in seconds.

    """

    def __init__(self, default_limit: int = 60, default_window_seconds: int = 60) -> None:
        self._default_limit = default_limit
        self._default_window = default_window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def check_rate_limit(
        self,
        key: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> bool:
        """Check whether a request is within the rate limit.

        If the request is allowed, the current timestamp is recorded and
        ``True`` is returned. If the limit is exceeded, ``False`` is
        returned and no timestamp is recorded.

        Parameters
        ----------
        key:
            The rate-limit key (e.g. hashed API key).
        limit:
            Override the default limit for this check.
        window_seconds:
            Override the default window for this check.

        Returns
        -------
        bool:
            ``True`` if the request is allowed, ``False`` if rate-limited.

        """
        effective_limit = limit if limit is not None else self._default_limit
        effective_window = window_seconds if window_seconds is not None else self._default_window

        now = time.monotonic()
        cutoff = now - effective_window

        # Prune old entries
        timestamps = self._requests[key]
        self._requests[key] = [t for t in timestamps if t > cutoff]

        collector = get_metrics_collector()
        current_count = len(self._requests[key])

        if current_count >= effective_limit:
            collector.increment("rate_limit_checks_total", labels={"status": "denied"})
            collector.observe("rate_limit_window_usage_ratio", 1.0)
            logger.info(
                "Rate limit denied for key '%s': %d/%d requests in window",
                key[:8],
                current_count,
                effective_limit,
                extra={
                    "security_data": {
                        "event": "rate_limit_denied",
                        "key_prefix": key[:8],
                        "current_count": current_count,
                        "limit": effective_limit,
                        "window_seconds": effective_window,
                    }
                },
            )
            return False

        self._requests[key].append(now)
        collector.increment("rate_limit_checks_total", labels={"status": "allowed"})
        usage_ratio = (current_count + 1) / effective_limit
        collector.observe("rate_limit_window_usage_ratio", usage_ratio)
        return True

    def get_retry_after(self, key: str, window_seconds: int | None = None) -> int:
        """Calculate seconds until the oldest request in the window expires.

        Parameters
        ----------
        key:
            The rate-limit key.
        window_seconds:
            Override the default window.

        Returns
        -------
        int:
            Seconds to wait (rounded up), or 0 if no entries.

        """
        effective_window = window_seconds if window_seconds is not None else self._default_window

        timestamps = self._requests.get(key, [])
        if not timestamps:
            return 0

        now = time.monotonic()
        oldest = min(timestamps)
        expires_at = oldest + effective_window
        remaining = expires_at - now

        return max(1, int(remaining) + 1)

    def reset(self, key: str | None = None) -> None:
        """Reset rate limit state.

        Parameters
        ----------
        key:
            If provided, reset only this key. Otherwise reset all keys.

        """
        if key is not None:
            self._requests.pop(key, None)
        else:
            self._requests.clear()
