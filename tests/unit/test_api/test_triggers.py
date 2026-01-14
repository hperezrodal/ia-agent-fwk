"""Tests for trigger and webhook API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from ia_agent_fwk.api.app import create_app
from ia_agent_fwk.config.settings import AppSettings, AuthSettings, MemorySettings
from ia_agent_fwk.execution.triggers import TriggerManager
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend


@pytest.fixture
def mock_trigger_manager():
    return MagicMock(spec=TriggerManager)


@pytest.fixture
def test_settings_for_triggers(monkeypatch):
    monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
    return AppSettings(
        auth=AuthSettings(enabled=True),
        memory=MemorySettings(default_backend="in_memory"),
    )


@pytest.fixture
def test_app_with_triggers(test_settings_for_triggers, mock_trigger_manager):
    with patch("ia_agent_fwk.api.app.get_celery_app"), patch("ia_agent_fwk.api.app.JobManager"):
        app = create_app(test_settings_for_triggers)

    app.state.memory_backend = InMemoryBackend()
    app.state.conversation_backend = ConversationMemoryBackend()
    app.state.job_manager = MagicMock()
    app.state.schedule_manager = MagicMock()
    app.state.trigger_manager = mock_trigger_manager

    return app


@pytest.fixture
async def triggers_client(test_app_with_triggers):
    transport = httpx.ASGITransport(app=test_app_with_triggers)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key-1"}


@pytest.mark.unit
class TestRegisterTriggerEndpoint:
    async def test_register_trigger(self, triggers_client, auth_headers, mock_trigger_manager):
        mock_trigger_manager.register_trigger.return_value = "trig-123"

        response = await triggers_client.post(
            "/api/v1/triggers",
            headers=auth_headers,
            json={
                "name": "deploy-check",
                "agent_type": "monitor",
                "prompt_template": "Check deployment for {service}",
                "event_type": "deploy",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["trigger_id"] == "trig-123"
        assert data["name"] == "deploy-check"
        assert data["event_type"] == "deploy"


@pytest.mark.unit
class TestFireWebhook:
    async def test_fire_webhook(self, triggers_client, auth_headers, mock_trigger_manager):
        mock_trigger_manager.fire_trigger.return_value = ("trig-123", "job-abc")

        response = await triggers_client.post(
            "/api/v1/webhooks/deploy",
            headers=auth_headers,
            json={"data": {"service": "api-server"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["event_type"] == "deploy"
        assert data["trigger_id"] == "trig-123"
        assert data["job_id"] == "job-abc"
        assert data["status"] == "submitted"

    async def test_fire_webhook_no_match(self, triggers_client, auth_headers, mock_trigger_manager):
        mock_trigger_manager.fire_trigger.return_value = None

        response = await triggers_client.post(
            "/api/v1/webhooks/unknown",
            headers=auth_headers,
            json={"data": {}},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["event_type"] == "unknown"


@pytest.mark.unit
class TestTriggersRequireAuth:
    async def test_no_auth_returns_401(self, triggers_client):
        response = await triggers_client.get("/api/v1/triggers")
        assert response.status_code == 401

    async def test_webhook_no_auth_returns_401(self, triggers_client):
        response = await triggers_client.post(
            "/api/v1/webhooks/deploy",
            json={"data": {}},
        )
        assert response.status_code == 401
