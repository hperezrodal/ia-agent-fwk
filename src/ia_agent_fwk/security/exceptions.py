"""Security module exception hierarchy.

All security-related exceptions inherit from ``SecurityError`` which itself
inherits from the built-in ``Exception``.
"""

from __future__ import annotations


class SecurityError(Exception):
    """Base exception for all security-related errors."""


class RateLimitExceededError(SecurityError):
    """Raised when a client exceeds the configured rate limit.

    Attributes
    ----------
    key:
        The rate-limit key (e.g. hashed API key) that was throttled.
    retry_after:
        Seconds the client should wait before retrying.

    """

    def __init__(self, key: str, retry_after: int) -> None:
        super().__init__(f"Rate limit exceeded for key '{key}'. Retry after {retry_after}s.")
        self.key = key
        self.retry_after = retry_after


class AuditLogError(SecurityError):
    """Raised when an audit logging operation fails."""
