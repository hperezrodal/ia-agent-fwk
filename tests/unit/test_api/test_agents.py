"""Tests for agent execution endpoints."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestAgentEndpoints:
    async def test_run_agent_success(self, client, auth_headers):
        response = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": "Hello!"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["output"] == "Echo: Hello!"
        assert data["agent_type"] == "test"
        assert "conversation_id" in data
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 20

    async def test_run_agent_unknown_type(self, client, auth_headers):
        response = await client.post(
            "/api/v1/agents/nonexistent/run",
            json={"prompt": "Hello!"},
            headers=auth_headers,
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "RESOURCE_NOT_FOUND"

    async def test_run_agent_empty_prompt(self, client, auth_headers):
        response = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    async def test_run_agent_no_auth(self, client):
        response = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": "Hello!"},
        )
        assert response.status_code == 401

    async def test_run_agent_invalid_auth(self, client):
        response = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": "Hello!"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    async def test_run_agent_with_conversation(self, client, auth_headers):
        # First run creates conversation
        response1 = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": "First message"},
            headers=auth_headers,
        )
        assert response1.status_code == 200
        conversation_id = response1.json()["conversation_id"]

        # Second run with same conversation_id
        response2 = await client.post(
            "/api/v1/agents/test/run",
            json={"prompt": "Second message", "conversation_id": conversation_id},
            headers=auth_headers,
        )
        assert response2.status_code == 200
        assert response2.json()["conversation_id"] == conversation_id
