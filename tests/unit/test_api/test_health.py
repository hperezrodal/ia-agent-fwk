"""Tests for health check endpoints."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestHealthEndpoints:
    async def test_liveness_probe(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_liveness_no_auth_required(self, client):
        """Health endpoint should work without API key."""
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_readiness_probe_healthy(self, client):
        response = await client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["memory"] == "healthy"
