"""Observability exception hierarchy.

All observability-specific exceptions inherit from ``ObservabilityError``
which itself inherits from the built-in ``Exception``.
"""

from __future__ import annotations


class ObservabilityError(Exception):
    """Base exception for all observability errors."""


class TracingConfigError(ObservabilityError):
    """Raised when tracing configuration is invalid."""


class MetricsError(ObservabilityError):
    """Raised when a metrics operation fails."""
