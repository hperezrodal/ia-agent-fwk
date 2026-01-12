"""Input sanitization utilities for security-sensitive operations.

Provides helpers to sanitize log values, mask secrets, and clean error
messages before they are exposed to external callers.
"""

from __future__ import annotations

import logging
import re

from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# Control characters to strip (C0 controls except space, plus DEL)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_log_value(value: str, max_length: int = 1000) -> str:
    """Sanitize a value for safe inclusion in log output.

    Strips control characters (except newline and carriage return) and
    truncates to ``max_length``.

    Parameters
    ----------
    value:
        The raw string to sanitize.
    max_length:
        Maximum length of the returned string. Defaults to 1000.

    Returns
    -------
    str:
        The sanitized, truncated string.

    """
    collector = get_metrics_collector()
    collector.increment("sanitization_operations_total", labels={"operation": "log_value"})

    cleaned = _CONTROL_CHAR_RE.sub("", value)
    if _CONTROL_CHAR_RE.search(value):
        collector.increment("sanitization_detections_total", labels={"type": "control_chars"})

    if len(cleaned) > max_length:
        collector.increment("sanitization_detections_total", labels={"type": "truncation"})
        return cleaned[:max_length] + "...[truncated]"
    return cleaned


def mask_secret(value: str) -> str:
    """Mask a secret value, showing only the first 4 and last 4 characters.

    For values shorter than 12 characters, the entire value is masked.

    Parameters
    ----------
    value:
        The secret value to mask.

    Returns
    -------
    str:
        The masked string (e.g. ``"sk-a****xyz1"``).

    """
    collector = get_metrics_collector()
    collector.increment("sanitization_operations_total", labels={"operation": "mask_secret"})

    if len(value) < 12:  # noqa: PLR2004
        return "*" * len(value)

    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def sanitize_error_message(exc: Exception) -> str:
    """Produce a safe error message from an exception.

    Removes internal details such as file paths, stack traces, and
    module names. Returns a generic message if the exception message
    contains potentially sensitive patterns.

    Parameters
    ----------
    exc:
        The exception to sanitize.

    Returns
    -------
    str:
        A sanitized error message suitable for external display.

    """
    collector = get_metrics_collector()
    collector.increment("sanitization_operations_total", labels={"operation": "error_message"})

    raw = str(exc)

    # Patterns that indicate internal details
    sensitive_patterns = [
        re.compile(r"/[a-zA-Z0-9_./]+\.py"),  # file paths
        re.compile(r"line \d+"),  # line numbers
        re.compile(r"Traceback"),  # traceback markers
        re.compile(r"File \""),  # traceback file references
        re.compile(r"(?:password|secret|token|api_key)\s*=\s*\S+", re.IGNORECASE),  # credentials
    ]

    for pattern in sensitive_patterns:
        if pattern.search(raw):
            collector.increment("sanitization_detections_total", labels={"type": "sensitive_error"})
            logger.warning(
                "Sensitive pattern detected in error message, returning generic response",
                extra={
                    "security_data": {
                        "event": "sensitive_error_sanitized",
                        "exception_type": type(exc).__name__,
                    }
                },
            )
            return "An internal error occurred. Please contact support."

    # Truncate and clean the message
    return sanitize_log_value(raw, max_length=200)
