"""Tests for the observability metrics module."""

from __future__ import annotations

import time

import pytest

from ia_agent_fwk.observability.metrics import MetricsCollector, get_metrics_collector


@pytest.mark.unit
class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_increment_default(self, metrics_collector):
        """Incrementing a counter defaults to +1."""
        metrics_collector.increment("requests_total")
        assert metrics_collector.get_counter("requests_total") == 1

    def test_increment_by_value(self, metrics_collector):
        """Incrementing a counter by a specific value."""
        metrics_collector.increment("tokens_total", 100)
        metrics_collector.increment("tokens_total", 50)
        assert metrics_collector.get_counter("tokens_total") == 150

    def test_increment_with_labels(self, metrics_collector):
        """Labels create separate counter buckets."""
        metrics_collector.increment("requests_total", labels={"method": "GET"})
        metrics_collector.increment("requests_total", labels={"method": "POST"})
        metrics_collector.increment("requests_total", labels={"method": "GET"})

        assert metrics_collector.get_counter("requests_total", labels={"method": "GET"}) == 2
        assert metrics_collector.get_counter("requests_total", labels={"method": "POST"}) == 1

    def test_get_counter_nonexistent(self, metrics_collector):
        """Getting a non-existent counter returns 0."""
        assert metrics_collector.get_counter("nonexistent") == 0

    def test_observe_histogram(self, metrics_collector):
        """Observing values in a histogram."""
        metrics_collector.observe("duration_seconds", 0.5)
        metrics_collector.observe("duration_seconds", 1.0)
        metrics_collector.observe("duration_seconds", 0.3)

        stats = metrics_collector.get_histogram("duration_seconds")
        assert stats["count"] == 3
        assert abs(stats["sum"] - 1.8) < 0.001
        assert abs(stats["min"] - 0.3) < 0.001
        assert abs(stats["max"] - 1.0) < 0.001

    def test_get_histogram_nonexistent(self, metrics_collector):
        """Getting a non-existent histogram returns zero stats."""
        stats = metrics_collector.get_histogram("nonexistent")
        assert stats["count"] == 0
        assert stats["sum"] == 0.0

    def test_timer_context_manager(self, metrics_collector):
        """Timer context manager records elapsed time."""
        with metrics_collector.timer("operation_duration"):
            time.sleep(0.01)

        stats = metrics_collector.get_histogram("operation_duration")
        assert stats["count"] == 1
        assert stats["min"] > 0

    def test_snapshot(self, metrics_collector):
        """Snapshot returns a copy of all metrics."""
        metrics_collector.increment("counter_a", 5)
        metrics_collector.observe("hist_a", 1.0)
        metrics_collector.observe("hist_a", 2.0)

        snap = metrics_collector.snapshot()
        assert "counters" in snap
        assert "histograms" in snap
        assert snap["counters"]["counter_a"][""] == 5
        assert snap["histograms"]["hist_a"]["count"] == 2

    def test_reset(self, metrics_collector):
        """Reset clears all metrics."""
        metrics_collector.increment("counter_a", 5)
        metrics_collector.observe("hist_a", 1.0)

        metrics_collector.reset()

        assert metrics_collector.get_counter("counter_a") == 0
        assert metrics_collector.get_histogram("hist_a")["count"] == 0

    def test_label_key_stability(self, metrics_collector):
        """Labels with different ordering produce the same key."""
        metrics_collector.increment("c", labels={"a": "1", "b": "2"})
        metrics_collector.increment("c", labels={"b": "2", "a": "1"})

        assert metrics_collector.get_counter("c", labels={"a": "1", "b": "2"}) == 2

    def test_observe_labeled_histogram(self, metrics_collector):
        """Observing a histogram with labels records the value."""
        metrics_collector.observe("my_hist", 1.5, labels={"route": "/api"})

        stats = metrics_collector.get_histogram("my_hist", labels={"route": "/api"})
        assert stats["count"] == 1
        assert abs(stats["sum"] - 1.5) < 0.001

    def test_get_labeled_histogram(self, metrics_collector):
        """get_histogram with labels returns correct stats for that label combination."""
        metrics_collector.observe("req_dur", 0.2, labels={"route": "/api"})
        metrics_collector.observe("req_dur", 0.8, labels={"route": "/api"})
        metrics_collector.observe("req_dur", 5.0, labels={"route": "/health"})

        api_stats = metrics_collector.get_histogram("req_dur", labels={"route": "/api"})
        assert api_stats["count"] == 2
        assert abs(api_stats["sum"] - 1.0) < 0.001
        assert abs(api_stats["min"] - 0.2) < 0.001
        assert abs(api_stats["max"] - 0.8) < 0.001

        health_stats = metrics_collector.get_histogram("req_dur", labels={"route": "/health"})
        assert health_stats["count"] == 1
        assert abs(health_stats["sum"] - 5.0) < 0.001

    def test_get_labeled_histogram_nonexistent(self, metrics_collector):
        """Getting a labeled histogram that doesn't exist returns zero stats."""
        stats = metrics_collector.get_histogram("nope", labels={"x": "y"})
        assert stats["count"] == 0
        assert stats["sum"] == 0.0

    def test_snapshot_with_labeled_histograms(self, metrics_collector):
        """Snapshot includes labeled histograms with composite key names."""
        metrics_collector.observe("dur", 1.0, labels={"route": "/api"})
        metrics_collector.observe("dur", 3.0, labels={"route": "/api"})
        metrics_collector.observe("dur", 7.0, labels={"route": "/health"})

        snap = metrics_collector.snapshot()
        histograms = snap["histograms"]

        api_key = "dur{route=/api}"
        assert api_key in histograms
        assert histograms[api_key]["count"] == 2
        assert abs(histograms[api_key]["sum"] - 4.0) < 0.001
        assert abs(histograms[api_key]["min"] - 1.0) < 0.001
        assert abs(histograms[api_key]["max"] - 3.0) < 0.001

        health_key = "dur{route=/health}"
        assert health_key in histograms
        assert histograms[health_key]["count"] == 1
        assert abs(histograms[health_key]["sum"] - 7.0) < 0.001

    def test_reset_clears_labeled_histograms(self, metrics_collector):
        """Reset clears labeled histograms as well."""
        metrics_collector.observe("dur", 1.0, labels={"route": "/api"})
        metrics_collector.reset()

        stats = metrics_collector.get_histogram("dur", labels={"route": "/api"})
        assert stats["count"] == 0


@pytest.mark.unit
class TestGetMetricsCollector:
    """Tests for get_metrics_collector singleton."""

    def test_returns_instance(self):
        """get_metrics_collector returns a MetricsCollector."""
        collector = get_metrics_collector()
        assert isinstance(collector, MetricsCollector)

    def test_returns_same_instance(self):
        """get_metrics_collector returns the same instance."""
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2
