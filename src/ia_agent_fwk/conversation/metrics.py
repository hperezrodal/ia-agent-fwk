"""Prometheus metrics + observability setup for the conversational RAG agent.

Defines all metrics emitted by the conversation module and provides
setup_observability() to configure tracing, logging, and /metrics endpoint.

Usage:
    from ia_agent_fwk.conversation.metrics import setup_observability
    setup_observability(app)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

TEMPO_ENDPOINT = os.environ.get("TEMPO_ENDPOINT", "")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "ia-agent")

# ═══════════════════════════════════════════════════════════════════════════
# Prometheus Metrics — conversation agent
# ═══════════════════════════════════════════════════════════════════════════

chat_requests_total = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["agent", "mode"],
)

chat_latency = Histogram(
    "chat_latency_seconds",
    "Chat request latency",
    ["agent"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
)

llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["provider", "model", "purpose"],
)

llm_latency = Histogram(
    "llm_latency_seconds",
    "LLM call latency",
    ["provider", "purpose"],
    buckets=[0.1, 0.3, 0.5, 1, 2, 5, 10, 30],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["provider", "direction"],
)

rag_latency = Histogram(
    "rag_latency_seconds",
    "RAG search+rerank latency",
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

rag_results = Histogram(
    "rag_results_count",
    "Number of RAG results returned",
    buckets=[0, 1, 2, 3, 5, 10],
)

active_sessions = Counter(
    "sessions_created_total",
    "Total sessions created",
)

# ═══════════════════════════════════════════════════════════════════════════
# Tracing (OpenTelemetry → Tempo)
# ═══════════════════════════════════════════════════════════════════════════


def _setup_tracing() -> None:
    if not TEMPO_ENDPOINT:
        logger.info("Tracing disabled (TEMPO_ENDPOINT not set)")
        return
    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415

        resource = Resource.create({"service.name": SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{TEMPO_ENDPOINT}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("Tracing enabled → %s", TEMPO_ENDPOINT)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to setup tracing", exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════
# Structured JSON Logging
# ═══════════════════════════════════════════════════════════════════════════


class _JSONFormatter(logging.Formatter):
    """JSON log formatter with trace_id for Loki correlation."""

    def format(self, record: logging.LogRecord) -> str:
        import json  # noqa: PLC0415

        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        try:
            from opentelemetry import trace  # noqa: PLC0415

            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.trace_id:
                entry["trace_id"] = format(ctx.trace_id, "032x")
                entry["span_id"] = format(ctx.span_id, "016x")
        except Exception:  # noqa: BLE001
            logger.debug("Failed to get trace context")
        for key in (
            "session_id",
            "query",
            "provider",
            "model",
            "purpose",
            "tokens_in",
            "tokens_out",
            "duration_ms",
            "mode",
        ):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def _setup_logging() -> None:
    log_format = os.environ.get("LOG_FORMAT", "json")
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root.handlers = [handler]
    for name in ("httpx", "httpcore", "opentelemetry", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════
# Setup entrypoint
# ═══════════════════════════════════════════════════════════════════════════


def setup_observability(app: FastAPI) -> None:
    """Initialize tracing, logging, metrics, and FastAPI instrumentation."""
    _setup_logging()
    _setup_tracing()

    if TEMPO_ENDPOINT:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415

            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI auto-instrumented with OTel")
        except Exception:  # noqa: BLE001
            logger.warning("Failed to instrument FastAPI", exc_info=True)

    from fastapi.responses import Response  # noqa: PLC0415

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    logger.info("Observability initialized (service=%s)", SERVICE_NAME)


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════


@contextmanager
def timed(histogram: Histogram, **labels: str) -> Any:
    """Time a block and record to a Prometheus histogram."""
    t0 = time.monotonic()
    yield
    histogram.labels(**labels).observe(time.monotonic() - t0)
