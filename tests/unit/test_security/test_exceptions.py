"""Tests for security module exception hierarchy."""

from __future__ import annotations

import pytest

from ia_agent_fwk.security.exceptions import (
    AuditLogError,
    RateLimitExceededError,
    SecurityError,
)


@pytest.mark.unit
class TestSecurityExceptionHierarchy:
    def test_security_error_is_exception(self):
        assert issubclass(SecurityError, Exception)

    def test_rate_limit_exceeded_is_security_error(self):
        assert issubclass(RateLimitExceededError, SecurityError)

    def test_audit_log_error_is_security_error(self):
        assert issubclass(AuditLogError, SecurityError)

    def test_rate_limit_exceeded_attributes(self):
        exc = RateLimitExceededError(key="test-key", retry_after=30)
        assert exc.key == "test-key"
        assert exc.retry_after == 30
        assert "test-key" in str(exc)
        assert "30" in str(exc)

    def test_rate_limit_exceeded_catchable_as_security_error(self):
        with pytest.raises(SecurityError):
            raise RateLimitExceededError(key="k", retry_after=10)

    def test_audit_log_error_message(self):
        exc = AuditLogError("Failed to write audit log")
        assert str(exc) == "Failed to write audit log"
