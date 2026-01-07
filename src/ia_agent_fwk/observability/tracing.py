"""OpenTelemetry tracing integration.

Provides a ``TracingManager`` that configures the OTel ``TracerProvider``
with configurable exporters (console, OTLP), a ``get_tracer`` helper,
and a ``@traced`` decorator for adding spans to async functions.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import StatusCode

from ia_agent_fwk.observability.exceptions import TracingConfigError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opentelemetry.trace import Tracer

    from ia_agent_fwk.config.settings import TracingSettings

logger = logging.getLogger(__name__)

_VALID_EXPORTERS = frozenset({"console", "otlp", "otlp-http"})


class TracingManager:
    """Configure and manage the OpenTelemetry tracer provider.

    Parameters
    ----------
    settings:
        Tracing configuration from ``ObservabilitySettings.tracing``.

    """

    def __init__(self, settings: TracingSettings) -> None:
        self._settings = settings
        self._provider: TracerProvider | None = None

    def setup(self) -> None:
        """Initialise the OTel tracer provider based on settings.

        When ``settings.enabled`` is ``False`` the global tracer provider
        is left as the default no-op provider.
        """
        if not self._settings.enabled:
            logger.info("Tracing is disabled")
            return

        exporter_name = self._settings.exporter.lower()
        if exporter_name not in _VALID_EXPORTERS:
            msg = f"Unsupported tracing exporter: {exporter_name!r}. Valid: {sorted(_VALID_EXPORTERS)}"
            raise TracingConfigError(msg)

        resource = Resource.create(
            {"service.name": self._settings.service_name},
        )
        self._provider = TracerProvider(resource=resource)

        processor: SpanProcessor
        if exporter_name == "console":
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
        elif exporter_name == "otlp-http":
            # OTLP HTTP exporter (lighter than gRPC, used with Tempo)
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
                    OTLPSpanExporter as OTLPHTTPSpanExporter,
                )
            except ImportError as exc:
                msg = (
                    "opentelemetry-exporter-otlp-proto-http is required "
                    "for OTLP HTTP tracing. Install with: "
                    "pip install opentelemetry-exporter-otlp-proto-http"
                )
                raise TracingConfigError(msg) from exc
            endpoint = self._settings.endpoint.rstrip("/") + "/v1/traces"
            processor = BatchSpanProcessor(
                OTLPHTTPSpanExporter(endpoint=endpoint),
            )
        else:
            # OTLP gRPC exporter -- imported lazily so the dependency is optional.
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
                    OTLPSpanExporter,
                )
            except ImportError as exc:
                msg = (
                    "opentelemetry-exporter-otlp-proto-grpc is required "
                    "for OTLP tracing. Install with: "
                    "pip install opentelemetry-exporter-otlp-proto-grpc"
                )
                raise TracingConfigError(msg) from exc
            processor = BatchSpanProcessor(
                OTLPSpanExporter(endpoint=self._settings.endpoint),
            )

        self._provider.add_span_processor(processor)
        trace.set_tracer_provider(self._provider)
        logger.info(
            "Tracing initialised (exporter=%s, service=%s)",
            exporter_name,
            self._settings.service_name,
        )

    def shutdown(self) -> None:
        """Flush and shut down the tracer provider."""
        if self._provider is not None:
            self._provider.shutdown()
            logger.info("Tracing shut down")

    @property
    def provider(self) -> TracerProvider | None:
        """Return the configured tracer provider (``None`` if disabled)."""
        return self._provider


def get_tracer(name: str) -> Tracer:
    """Return an OTel tracer for the given *name*.

    Uses the global tracer provider, which is the no-op provider when
    tracing has not been initialised.
    """
    return trace.get_tracer(name)


def traced(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> Callable[..., Callable[..., Awaitable[Any]]]:
    """Wrap an async function in an OTel span.

    Parameters
    ----------
    name:
        Span name.  Defaults to ``<module>.<function>``.
    attributes:
        Static span attributes to set on every invocation.

    """

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"
        tracer = get_tracer(fn.__module__)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise
                else:
                    span.set_status(StatusCode.OK)
                    return result

        return wrapper

    return decorator
