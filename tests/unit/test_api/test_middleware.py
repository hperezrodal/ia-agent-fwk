"""Tests for middleware and authentication."""

from __future__ import annotations

import re

import pytest


@pytest.mark.unit
class TestMiddleware:
    async def test_request_id_generated(self, client):
        response = await client.get("/health")
        request_id = response.headers.get("x-request-id", "")
        # UUID4 format
        uuid4_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        assert uuid4_pattern.match(request_id), f"Expected UUID4 format, got: {request_id}"

    async def test_auth_valid_key(self, client, auth_headers):
        response = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": "Hello!"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    async def test_auth_missing_key(self, client):
        response = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": "Hello!"},
        )
        assert response.status_code == 401

    async def test_auth_invalid_key(self, client):
        response = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": "Hello!"},
            headers={"X-API-Key": "totally-wrong-key"},
        )
        assert response.status_code == 401

    async def test_error_response_includes_request_id(self, client):
        response = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": ""},
            headers={"X-API-Key": "test-key-1"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "request_id" in data["error"]
        assert data["error"]["request_id"] != "unknown"
