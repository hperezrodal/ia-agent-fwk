"""Observability subsystem: tracing, logging, metrics, prompt logging.

Public API
----------
.. autoclass:: TracingManager
.. autoclass:: JSONFormatter
.. autoclass:: MetricsCollector
.. autoclass:: PromptLogger
.. autoclass:: ObservabilityMiddleware
.. autoclass:: ObservabilityError
"""

from __future__ import annotations

from ia_agent_fwk.observability.exceptions import (
    MetricsError,
    ObservabilityError,
    TracingConfigError,
)
from ia_agent_fwk.observability.logging import JSONFormatter, setup_logging
from ia_agent_fwk.observability.metrics import MetricsCollector, get_metrics_collector
from ia_agent_fwk.observability.middleware import ObservabilityMiddleware
from ia_agent_fwk.observability.prompt_log import PromptLogger
from ia_agent_fwk.observability.tracing import TracingManager, get_tracer, traced

__all__ = [
    "JSONFormatter",
    "MetricsCollector",
    "MetricsError",
    "ObservabilityError",
    "ObservabilityMiddleware",
    "PromptLogger",
    "TracingConfigError",
    "TracingManager",
    "get_metrics_collector",
    "get_tracer",
    "setup_logging",
    "traced",
]
