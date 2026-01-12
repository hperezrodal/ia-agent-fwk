"""Tests for input sanitization utilities."""

from __future__ import annotations

import pytest

from ia_agent_fwk.security.sanitizer import (
    mask_secret,
    sanitize_error_message,
    sanitize_log_value,
)


@pytest.mark.unit
class TestSanitizeLogValue:
    def test_sanitize_log_value_passthrough(self):
        assert sanitize_log_value("hello world") == "hello world"

    def test_sanitize_log_value_truncates(self):
        long_value = "a" * 2000
        result = sanitize_log_value(long_value, max_length=100)
        assert len(result) == 100 + len("...[truncated]")
        assert result.endswith("...[truncated]")

    def test_sanitize_log_value_strips_control_chars(self):
        value = "hello\x00world\x01test\x7f"
        result = sanitize_log_value(value)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x7f" not in result
        assert "helloworld" in result

    def test_sanitize_log_value_preserves_newline_tab(self):
        # Newline and tab are NOT stripped (they are common in log output)
        # Our regex strips \x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f
        # \x09 (tab) and \x0a (newline) and \x0d (carriage return) are NOT matched
        value = "line1\nline2\ttabbed"
        result = sanitize_log_value(value)
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_log_value_custom_max_length(self):
        result = sanitize_log_value("abcdefghij", max_length=5)
        assert result == "abcde...[truncated]"

    def test_sanitize_log_value_exact_length(self):
        result = sanitize_log_value("abcde", max_length=5)
        assert result == "abcde"

    def test_sanitize_log_value_empty_string(self):
        assert sanitize_log_value("") == ""


@pytest.mark.unit
class TestMaskSecret:
    def test_mask_secret_long_value(self):
        result = mask_secret("sk-abcdef123456xyz")
        assert result.startswith("sk-a")
        assert result.endswith("6xyz")
        assert "*" in result
        # First 4 + masked middle + last 4
        assert len(result) == len("sk-abcdef123456xyz")

    def test_mask_secret_short_value(self):
        result = mask_secret("short")
        assert result == "*****"

    def test_mask_secret_exactly_12(self):
        result = mask_secret("123456789012")
        assert result == "1234****9012"

    def test_mask_secret_less_than_12(self):
        result = mask_secret("12345678901")
        assert result == "***********"

    def test_mask_secret_empty(self):
        assert mask_secret("") == ""

    def test_mask_secret_api_key(self):
        result = mask_secret("sk-proj-abcdefghijklmnop")
        assert result[:4] == "sk-p"
        assert result[-4:] == "mnop"
        assert "*" * 10 in result


@pytest.mark.unit
class TestSanitizeErrorMessage:
    def test_sanitize_error_message_safe(self):
        exc = ValueError("Invalid input: expected integer")
        result = sanitize_error_message(exc)
        assert result == "Invalid input: expected integer"

    def test_sanitize_error_message_file_path(self):
        exc = RuntimeError("Error in /home/user/app/main.py at line 42")
        result = sanitize_error_message(exc)
        assert result == "An internal error occurred. Please contact support."

    def test_sanitize_error_message_traceback(self):
        exc = RuntimeError("Traceback (most recent call last):")
        result = sanitize_error_message(exc)
        assert result == "An internal error occurred. Please contact support."

    def test_sanitize_error_message_credentials(self):
        exc = RuntimeError("Connection failed: password=s3cret123")
        result = sanitize_error_message(exc)
        assert result == "An internal error occurred. Please contact support."

    def test_sanitize_error_message_line_number(self):
        exc = RuntimeError("Failed at line 123")
        result = sanitize_error_message(exc)
        assert result == "An internal error occurred. Please contact support."

    def test_sanitize_error_message_truncates_long(self):
        exc = ValueError("A" * 500)
        result = sanitize_error_message(exc)
        assert len(result) <= 200 + len("...[truncated]")

    def test_sanitize_error_message_file_reference(self):
        exc = RuntimeError('File "module.py", line 10')
        result = sanitize_error_message(exc)
        assert result == "An internal error occurred. Please contact support."
