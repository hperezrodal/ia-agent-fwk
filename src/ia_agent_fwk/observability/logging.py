"""Structured JSON logging configuration.

Provides a ``JSONFormatter`` that outputs log records as JSON objects
and a ``setup_logging`` function that configures the root logger.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import LoggingSettings


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Fields included in every record:

    - ``timestamp`` (ISO 8601 UTC)
    - ``level``
    - ``logger``
    - ``message``
    - ``correlation_id`` (from ``request_id_ctx`` when available)

    Additional fields from ``record.__dict__`` are merged under ``extra``.
    """

    _BUILTIN_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        include_correlation_id: bool = True,
    ) -> None:
        super().__init__()
        self._include_timestamp = include_timestamp
        self._include_correlation_id = include_correlation_id

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a JSON string."""
        record.message = record.getMessage()

        log_dict: dict[str, Any] = {}

        if self._include_timestamp:
            log_dict["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()  # noqa: UP017

        log_dict["level"] = record.levelname
        log_dict["logger"] = record.name
        log_dict["message"] = record.message

        if self._include_correlation_id:
            log_dict["correlation_id"] = self._get_correlation_id()

        # Inject OTel trace context for Loki ↔ Tempo correlation
        trace_id, span_id = self._get_trace_context()
        if trace_id:
            log_dict["trace_id"] = trace_id
            log_dict["span_id"] = span_id

        # Collect extra fields (anything not in the standard LogRecord attrs)
        extra: dict[str, Any] = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._BUILTIN_ATTRS and not key.startswith("_")
        }
        if extra:
            log_dict["extra"] = extra

        # Append exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            log_dict["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_dict["stack_info"] = record.stack_info

        return json.dumps(log_dict, default=str)

    @staticmethod
    def _get_correlation_id() -> str:
        """Read the current request ID from the API middleware context."""
        try:
            from ia_agent_fwk.api.middleware import request_id_ctx  # noqa: PLC0415

            return request_id_ctx.get()
        except Exception:  # noqa: BLE001
            return "unknown"

    @staticmethod
    def _get_trace_context() -> tuple[str, str]:
        """Extract the OTel trace_id and span_id from the current span.

        Returns ``("", "")`` when tracing is disabled or no span is active.
        """
        try:
            from opentelemetry import trace as otel_trace  # noqa: PLC0415

            span = otel_trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.trace_id:
                return (
                    format(ctx.trace_id, "032x"),
                    format(ctx.span_id, "016x"),
                )
        except Exception:  # noqa: BLE001, S110
            pass
        return ("", "")


def setup_logging(settings: LoggingSettings) -> None:
    """Configure the root logger based on *settings*.

    When ``settings.format`` is ``"json"``, a ``JSONFormatter`` is
    installed on the root logger.  Otherwise the default text formatter
    is used.

    This also reconfigures ``uvicorn.access`` and ``uvicorn.error`` loggers
    to use the same formatter and propagate to root, ensuring all output is
    uniformly structured (important for Loki / log aggregation).

    Parameters
    ----------
    settings:
        Logging configuration from ``ObservabilitySettings.logging``.

    """
    log_level = getattr(logging, settings.level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)

    if settings.format.lower() == "json":
        formatter: logging.Formatter = JSONFormatter(
            include_timestamp=settings.include_timestamp,
            include_correlation_id=settings.include_correlation_id,
        )
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(fmt)

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Override uvicorn loggers so ALL output goes through our formatter
    for uvicorn_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(uvicorn_logger_name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
