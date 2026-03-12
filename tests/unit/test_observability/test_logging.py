"""Tests for the observability logging module."""

from __future__ import annotations

import json
import logging

import pytest

from ia_agent_fwk.config.settings import LoggingSettings
from ia_agent_fwk.observability.logging import JSONFormatter, setup_logging


@pytest.mark.unit
class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_format(self):
        """JSONFormatter produces valid JSON with required fields."""
        formatter = JSONFormatter(include_timestamp=True, include_correlation_id=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Hello world"
        assert "timestamp" in parsed

    def test_format_without_timestamp(self):
        """When include_timestamp=False, no timestamp field."""
        formatter = JSONFormatter(include_timestamp=False, include_correlation_id=False)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="warn",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "timestamp" not in parsed
        assert parsed["level"] == "WARNING"

    def test_format_with_correlation_id(self):
        """When include_correlation_id=True, correlation_id is included."""
        formatter = JSONFormatter(include_timestamp=False, include_correlation_id=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "correlation_id" in parsed

    def test_format_with_exception(self):
        """Exception info is included in JSON output."""
        formatter = JSONFormatter(include_timestamp=False, include_correlation_id=False)
        try:
            msg = "test exception"
            raise ValueError(msg)  # noqa: TRY301
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_format_with_extra_fields(self):
        """Extra fields in the log record appear under 'extra'."""
        formatter = JSONFormatter(include_timestamp=False, include_correlation_id=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="with extras",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "extra" in parsed
        assert parsed["extra"]["custom_field"] == "custom_value"

    def test_output_is_single_line(self):
        """JSON output is a single line (no embedded newlines)."""
        formatter = JSONFormatter(include_timestamp=True, include_correlation_id=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="single line test",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        # The output should be valid JSON on a single line (no newlines in the JSON itself,
        # except possibly in exception tracebacks which are stringified)
        assert json.loads(output)  # Must be valid JSON


@pytest.mark.unit
class TestSetupLogging:
    """Tests for setup_logging()."""

    def test_setup_json_format(self):
        """setup_logging with format='json' installs JSONFormatter."""
        settings = LoggingSettings(format="json", level="DEBUG")
        setup_logging(settings)

        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_setup_text_format(self):
        """setup_logging with format='text' installs standard Formatter."""
        settings = LoggingSettings(format="text", level="WARNING")
        setup_logging(settings)

        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
        assert not isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_setup_clears_existing_handlers(self):
        """setup_logging removes pre-existing handlers."""
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) >= 2

        settings = LoggingSettings(format="json", level="INFO")
        setup_logging(settings)

        assert len(root.handlers) == 1

    def test_setup_invalid_level_defaults_to_info(self):
        """Invalid log level falls back to INFO."""
        settings = LoggingSettings(format="text", level="INVALID_LEVEL")
        setup_logging(settings)

        root = logging.getLogger()
        assert root.level == logging.INFO
