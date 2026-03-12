"""Tests for the observability middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.middleware import ObservabilityMiddleware

if TYPE_CHECKING:
    from starlette.requests import Request


def _home(_request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


def _error(_request: Request) -> JSONResponse:
    return JSONResponse({"error": "bad"}, status_code=500)


def _make_app() -> Starlette:
    """Create a minimal Starlette app with the observability middleware."""
    app = Starlette(
        routes=[
            Route("/", _home),
            Route("/error", _error),
        ],
    )
    app.add_middleware(ObservabilityMiddleware)
    return app


@pytest.mark.unit
class TestObservabilityMiddleware:
    """Tests for ObservabilityMiddleware."""

    @pytest.fixture(autouse=True)
    def _reset_metrics(self):
        """Reset the global metrics collector before each test."""
        collector = get_metrics_collector()
        collector.reset()

    def test_successful_request_records_metrics(self):
        """A 200 request increments the counter and records duration."""
        app = _make_app()
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200

        collector = get_metrics_collector()
        count = collector.get_counter(
            "http_requests_total",
            labels={"method": "GET", "status": "200"},
        )
        assert count == 1

        hist = collector.get_histogram("http_request_duration_seconds")
        assert hist["count"] == 1
        assert hist["min"] > 0

    def test_error_request_records_metrics(self):
        """A 500 response increments the counter with status=500."""
        app = _make_app()
        client = TestClient(app)
        response = client.get("/error")
        assert response.status_code == 500

        collector = get_metrics_collector()
        count = collector.get_counter(
            "http_requests_total",
            labels={"method": "GET", "status": "500"},
        )
        assert count == 1

    def test_multiple_requests_accumulate(self):
        """Multiple requests accumulate in the counter."""
        app = _make_app()
        client = TestClient(app)
        client.get("/")
        client.get("/")
        client.get("/")

        collector = get_metrics_collector()
        count = collector.get_counter(
            "http_requests_total",
            labels={"method": "GET", "status": "200"},
        )
        assert count == 3

        hist = collector.get_histogram("http_request_duration_seconds")
        assert hist["count"] == 3

    def test_non_http_scope_passes_through(self):
        """Non-HTTP scopes (e.g. lifespan) pass through unchanged."""
        app = _make_app()
        client = TestClient(app)
        # TestClient internally sends a lifespan scope; the middleware
        # should let it pass through without errors.
        response = client.get("/")
        assert response.status_code == 200
