"""Tests for conversation CRUD endpoints."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestConversationEndpoints:
    async def test_create_conversation(self, client, auth_headers):
        response = await client.post(
            "/api/v1/conversations",
            json={"agent_type": "test"},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "conversation_id" in data
        assert data["agent_namespace"] == "test"
        assert data["message_count"] == 0

    async def test_list_conversations_empty(self, client, auth_headers):
        response = await client.get(
            "/api/v1/conversations",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["conversations"] == []
        assert data["total"] == 0

    async def test_list_conversations_with_data(self, client, auth_headers):
        # Create a conversation first
        await client.post(
            "/api/v1/conversations",
            json={"agent_type": "test"},
            headers=auth_headers,
        )

        response = await client.get(
            "/api/v1/conversations",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["conversations"]) >= 1
        assert data["total"] >= 1

    async def test_get_conversation_detail(self, client, auth_headers):
        # Create a conversation
        create_response = await client.post(
            "/api/v1/conversations",
            json={"agent_type": "test", "title": "Test Chat"},
            headers=auth_headers,
        )
        conversation_id = create_response.json()["conversation_id"]

        # Get detail
        response = await client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conversation_id
        assert data["title"] == "Test Chat"
        assert data["messages"] == []

    async def test_get_conversation_not_found(self, client, auth_headers):
        response = await client.get(
            "/api/v1/conversations/nonexistent-id",
            headers=auth_headers,
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "RESOURCE_NOT_FOUND"

    async def test_delete_conversation(self, client, auth_headers):
        # Create a conversation first
        create_response = await client.post(
            "/api/v1/conversations",
            json={"agent_type": "test", "title": "To Delete"},
            headers=auth_headers,
        )
        conversation_id = create_response.json()["conversation_id"]

        # Delete it
        response = await client.delete(
            f"/api/v1/conversations/{conversation_id}",
            headers=auth_headers,
        )
        assert response.status_code == 204

        # Verify it's gone
        get_response = await client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=auth_headers,
        )
        assert get_response.status_code == 404

    async def test_delete_conversation_not_found(self, client, auth_headers):
        response = await client.delete(
            "/api/v1/conversations/nonexistent-id",
            headers=auth_headers,
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "RESOURCE_NOT_FOUND"

    async def test_conversations_require_auth(self, client):
        response = await client.get("/api/v1/conversations")
        assert response.status_code == 401
