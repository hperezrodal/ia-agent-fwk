"""Shared fixtures for security module unit tests."""

from __future__ import annotations

import pytest

from ia_agent_fwk.security.audit import AuditLogger
from ia_agent_fwk.security.rate_limiter import SlidingWindowRateLimiter


@pytest.fixture
def rate_limiter():
    """Create a rate limiter with a small limit for testing."""
    return SlidingWindowRateLimiter(default_limit=3, default_window_seconds=60)


@pytest.fixture
def audit_logger():
    """Create an audit logger instance."""
    return AuditLogger()
