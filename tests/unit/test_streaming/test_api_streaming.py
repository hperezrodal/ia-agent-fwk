"""Tests for streaming API endpoints (SSE + WebSocket route existence)."""

from __future__ import annotations

import json

import pytest


@pytest.mark.unit
class TestSSEEndpoint:
    async def test_sse_endpoint_returns_event_stream(self, streaming_client, auth_headers):
        """POST /api/v1/agents/{type}/stream returns text/event-stream."""
        response = await streaming_client.post(
            "/api/v1/agents/stream_test/stream",
            json={"prompt": "Hello"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events from response body
        body = response.text
        lines = [line for line in body.split("\n") if line.startswith("data: ")]
        assert len(lines) >= 2  # start + complete

        events = [json.loads(line[6:]) for line in lines]
        assert events[0]["event"] == "start"
        assert events[1]["event"] == "complete"
        assert events[1]["content"] == "Streamed: Hello"

    async def test_sse_endpoint_requires_auth(self, streaming_client):
        """POST /api/v1/agents/{type}/stream returns 401 without API key."""
        response = await streaming_client.post(
            "/api/v1/agents/stream_test/stream",
            json={"prompt": "Hello"},
        )
        assert response.status_code == 401

    async def test_sse_endpoint_unknown_agent(self, streaming_client, auth_headers):
        """POST /api/v1/agents/unknown/stream returns 404."""
        response = await streaming_client.post(
            "/api/v1/agents/nonexistent/stream",
            json={"prompt": "Hello"},
            headers=auth_headers,
        )
        assert response.status_code == 404

    async def test_sse_endpoint_empty_prompt(self, streaming_client, auth_headers):
        """POST with empty prompt returns 422."""
        response = await streaming_client.post(
            "/api/v1/agents/stream_test/stream",
            json={"prompt": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_sse_endpoint_has_cache_control_header(self, streaming_client, auth_headers):
        """SSE response includes Cache-Control: no-cache."""
        response = await streaming_client.post(
            "/api/v1/agents/stream_test/stream",
            json={"prompt": "Hi"},
            headers=auth_headers,
        )
        assert response.headers.get("cache-control") == "no-cache"

    async def test_sse_complete_event_usage(self, streaming_client, auth_headers):
        """Complete event includes token usage."""
        response = await streaming_client.post(
            "/api/v1/agents/stream_test/stream",
            json={"prompt": "usage test"},
            headers=auth_headers,
        )
        body = response.text
        lines = [line for line in body.split("\n") if line.startswith("data: ")]
        complete = json.loads(lines[-1][6:])
        assert complete["usage"]["prompt_tokens"] == 5
        assert complete["usage"]["completion_tokens"] == 15


@pytest.mark.unit
class TestWebSocketEndpoint:
    def test_websocket_endpoint_exists(self, streaming_app):
        """Verify the WebSocket route is registered in the application."""
        ws_routes = [route for route in streaming_app.routes if hasattr(route, "path") and route.path == "/api/v1/ws"]
        assert len(ws_routes) == 1
