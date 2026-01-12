"""Security module: rate limiting, audit logging, and input sanitization."""

from __future__ import annotations

from ia_agent_fwk.security.audit import AuditEvent, AuditEventType, AuditLogger, hash_api_key
from ia_agent_fwk.security.exceptions import AuditLogError, RateLimitExceededError, SecurityError
from ia_agent_fwk.security.rate_limiter import SlidingWindowRateLimiter, parse_rate
from ia_agent_fwk.security.sanitizer import mask_secret, sanitize_error_message, sanitize_log_value

__all__ = [
    "AuditEvent",
    "AuditEventType",
    "AuditLogError",
    "AuditLogger",
    "RateLimitExceededError",
    "SecurityError",
    "SlidingWindowRateLimiter",
    "hash_api_key",
    "mask_secret",
    "parse_rate",
    "sanitize_error_message",
    "sanitize_log_value",
]
