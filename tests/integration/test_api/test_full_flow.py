"""End-to-end integration tests for the API layer."""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestFullFlow:
    async def test_agent_execution_flow(self, integration_client, integration_auth_headers):
        """Execute an agent, verify response, check conversation history."""
        # Run agent
        run_response = await integration_client.post(
            "/api/v1/agents/integration-test/run",
            json={"prompt": "Hello integration!"},
            headers=integration_auth_headers,
        )
        assert run_response.status_code == 200
        data = run_response.json()
        assert data["output"] == "Integration response"
        conversation_id = data["conversation_id"]

        # Check conversation detail
        detail_response = await integration_client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=integration_auth_headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["conversation_id"] == conversation_id
        assert len(detail["messages"]) == 2  # user + assistant
        assert detail["messages"][0]["role"] == "user"
        assert detail["messages"][1]["role"] == "assistant"

    async def test_multi_turn_conversation(self, integration_client, integration_auth_headers):
        """Run agent twice with same conversation_id, verify history grows."""
        # First turn
        r1 = await integration_client.post(
            "/api/v1/agents/integration-test/run",
            json={"prompt": "First turn"},
            headers=integration_auth_headers,
        )
        assert r1.status_code == 200
        conversation_id = r1.json()["conversation_id"]

        # Second turn
        r2 = await integration_client.post(
            "/api/v1/agents/integration-test/run",
            json={"prompt": "Second turn", "conversation_id": conversation_id},
            headers=integration_auth_headers,
        )
        assert r2.status_code == 200
        assert r2.json()["conversation_id"] == conversation_id

        # Check history
        detail = await integration_client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=integration_auth_headers,
        )
        assert detail.status_code == 200
        messages = detail.json()["messages"]
        assert len(messages) == 4  # 2 turns x (user + assistant)

    async def test_health_check_flow(self, integration_client):
        """Verify liveness and readiness probes."""
        liveness = await integration_client.get("/health")
        assert liveness.status_code == 200
        assert liveness.json()["status"] == "healthy"

        readiness = await integration_client.get("/health/ready")
        assert readiness.status_code == 200
        assert readiness.json()["status"] == "ready"

    async def test_error_handling_flow(self, integration_client, integration_auth_headers):
        """Trigger validation errors and 404s, verify error envelope."""
        # Validation error
        val_resp = await integration_client.post(
            "/api/v1/agents/integration-test/run",
            json={"prompt": ""},
            headers=integration_auth_headers,
        )
        assert val_resp.status_code == 422
        assert val_resp.json()["error"]["code"] == "VALIDATION_ERROR"

        # Unknown agent type
        not_found = await integration_client.post(
            "/api/v1/agents/unknown-agent/run",
            json={"prompt": "Hello!"},
            headers=integration_auth_headers,
        )
        assert not_found.status_code == 404
        assert not_found.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

        # Missing auth
        no_auth = await integration_client.post(
            "/api/v1/agents/integration-test/run",
            json={"prompt": "Hello!"},
        )
        assert no_auth.status_code == 401
