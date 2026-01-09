"""In-memory metrics collection.

Provides a ``MetricsCollector`` that tracks counters and histograms
for agent executions, tool calls, LLM requests, and token usage.
A module-level singleton is accessible via ``get_metrics_collector()``.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


class MetricsCollector:
    """Thread-safe in-memory metrics collector.

    Supports two metric types:

    - **Counters** -- monotonically increasing integers (e.g. total requests).
    - **Histograms** -- record duration/size observations and compute basic
      statistics (count, sum, min, max).

    All operations are guarded by a ``threading.Lock`` for thread safety.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, int]] = {}
        self._histograms: dict[str, list[float]] = {}
        self._labeled_histograms: dict[str, dict[str, list[float]]] = {}

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def increment(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter by *value*.

        Parameters
        ----------
        name:
            Counter name (e.g. ``"agent_executions_total"``).
        value:
            Amount to increment (must be positive).
        labels:
            Optional label dict to create sub-counters (serialised as a
            sorted tuple key).

        """
        label_key = self._label_key(labels)
        with self._lock:
            bucket = self._counters.setdefault(name, {})
            bucket[label_key] = bucket.get(label_key, 0) + value

    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> int:
        """Return the current value of a counter."""
        label_key = self._label_key(labels)
        with self._lock:
            return self._counters.get(name, {}).get(label_key, 0)

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record an observation in a histogram.

        Parameters
        ----------
        name:
            Histogram name (e.g. ``"agent_execution_duration_seconds"``).
        value:
            Observed value (e.g. duration in seconds).
        labels:
            Optional label dict to create labeled sub-histograms.

        """
        if labels:
            label_key = self._label_key(labels)
            with self._lock:
                bucket = self._labeled_histograms.setdefault(name, {})
                bucket.setdefault(label_key, []).append(value)
        else:
            with self._lock:
                self._histograms.setdefault(name, []).append(value)

    def get_histogram(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return summary statistics for a histogram.

        Returns a dict with ``count``, ``sum``, ``min``, ``max`` keys.
        Returns all-zero values when no observations exist.
        """
        with self._lock:
            if labels:
                label_key = self._label_key(labels)
                observations = list(self._labeled_histograms.get(name, {}).get(label_key, []))
            else:
                observations = list(self._histograms.get(name, []))
        if not observations:
            return {"count": 0, "sum": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(observations),
            "sum": sum(observations),
            "min": min(observations),
            "max": max(observations),
        }

    # ------------------------------------------------------------------
    # Timer context manager
    # ------------------------------------------------------------------

    @contextmanager
    def timer(self, histogram_name: str) -> Iterator[None]:
        """Context manager that records elapsed time in a histogram.

        Usage::

            with collector.timer("agent_execution_duration_seconds"):
                await agent.run(...)

        """
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self.observe(histogram_name, elapsed)

    # ------------------------------------------------------------------
    # Snapshot / reset
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of all counters and histograms."""
        with self._lock:
            counters_copy: dict[str, dict[str, int]] = {name: dict(buckets) for name, buckets in self._counters.items()}
            histograms_copy: dict[str, dict[str, Any]] = {}
            for name, observations in self._histograms.items():
                if observations:
                    histograms_copy[name] = {
                        "count": len(observations),
                        "sum": sum(observations),
                        "min": min(observations),
                        "max": max(observations),
                    }
                else:
                    histograms_copy[name] = {
                        "count": 0,
                        "sum": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }
            # Include labeled histograms
            for name, labeled_buckets in self._labeled_histograms.items():
                for label_key, observations in labeled_buckets.items():
                    composite_name = f"{name}{{{label_key}}}" if label_key else name
                    if observations:
                        histograms_copy[composite_name] = {
                            "count": len(observations),
                            "sum": sum(observations),
                            "min": min(observations),
                            "max": max(observations),
                        }
                    else:
                        histograms_copy[composite_name] = {
                            "count": 0,
                            "sum": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                        }
        return {"counters": counters_copy, "histograms": histograms_copy}

    def reset(self) -> None:
        """Clear all counters and histograms."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._labeled_histograms.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label_key(labels: dict[str, str] | None) -> str:
        """Serialise labels into a stable string key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


# Module-level singleton
_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Return the global ``MetricsCollector`` singleton."""
    global _collector  # noqa: PLW0603
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector
