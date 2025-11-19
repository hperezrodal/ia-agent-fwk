"""Tests for the app factory."""

from __future__ import annotations

import pytest
from fastapi import FastAPI


@pytest.mark.unit
class TestAppFactory:
    def test_create_app_returns_fastapi(self, test_app):
        assert isinstance(test_app, FastAPI)

    async def test_request_id_header_present(self, client, auth_headers):
        response = await client.get("/health")
        assert "x-request-id" in response.headers

    async def test_request_id_client_provided_valid_uuid(self, client, auth_headers):
        valid_uuid = "12345678-1234-4abc-8def-1234567890ab"
        response = await client.get(
            "/health",
            headers={"X-Request-ID": valid_uuid},
        )
        assert response.headers["x-request-id"] == valid_uuid

    async def test_request_id_client_provided_invalid_ignored(self, client, auth_headers):
        response = await client.get(
            "/health",
            headers={"X-Request-ID": "my-custom-id"},
        )
        # Invalid format should be ignored; a new UUID4 is generated instead
        assert response.headers["x-request-id"] != "my-custom-id"

    async def test_openapi_docs_available(self, client):
        response = await client.get("/docs")
        assert response.status_code == 200
