"""Tests for schedule API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from ia_agent_fwk.api.app import create_app
from ia_agent_fwk.config.settings import AppSettings, AuthSettings, MemorySettings
from ia_agent_fwk.execution.exceptions import InvalidCronExpressionError
from ia_agent_fwk.execution.models import ScheduleDefinition
from ia_agent_fwk.execution.scheduler import ScheduleManager
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend


@pytest.fixture
def mock_schedule_manager():
    return MagicMock(spec=ScheduleManager)


@pytest.fixture
def test_settings_for_schedules(monkeypatch):
    monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
    return AppSettings(
        auth=AuthSettings(enabled=True),
        memory=MemorySettings(default_backend="in_memory"),
    )


@pytest.fixture
def test_app_with_schedules(test_settings_for_schedules, mock_schedule_manager):
    with patch("ia_agent_fwk.api.app.get_celery_app"), patch("ia_agent_fwk.api.app.JobManager"):
        app = create_app(test_settings_for_schedules)

    app.state.memory_backend = InMemoryBackend()
    app.state.conversation_backend = ConversationMemoryBackend()
    app.state.job_manager = MagicMock()
    app.state.schedule_manager = mock_schedule_manager
    app.state.trigger_manager = MagicMock()

    return app


@pytest.fixture
async def schedules_client(test_app_with_schedules):
    transport = httpx.ASGITransport(app=test_app_with_schedules)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key-1"}


@pytest.mark.unit
class TestCreateSchedule:
    async def test_create_schedule(self, schedules_client, auth_headers, mock_schedule_manager):
        mock_schedule_manager.add_schedule.return_value = "sched-123"

        response = await schedules_client.post(
            "/api/v1/schedules",
            headers=auth_headers,
            json={
                "name": "daily-report",
                "agent_type": "report",
                "prompt": "Generate daily report",
                "cron_expression": "0 9 * * *",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["schedule_id"] == "sched-123"
        assert data["name"] == "daily-report"
        assert data["cron_expression"] == "0 9 * * *"

    async def test_create_schedule_invalid_cron(self, schedules_client, auth_headers, mock_schedule_manager):
        mock_schedule_manager.add_schedule.side_effect = InvalidCronExpressionError("Invalid cron expression: 'bad'")

        response = await schedules_client.post(
            "/api/v1/schedules",
            headers=auth_headers,
            json={
                "name": "bad-schedule",
                "agent_type": "test",
                "prompt": "test",
                "cron_expression": "bad",
            },
        )
        assert response.status_code == 422


@pytest.mark.unit
class TestListSchedules:
    async def test_list_schedules(self, schedules_client, auth_headers, mock_schedule_manager):
        mock_schedule_manager.list_schedules.return_value = [
            (
                "sched-1",
                ScheduleDefinition(
                    name="daily",
                    agent_type="report",
                    prompt="Daily report",
                    cron_expression="0 9 * * *",
                ),
            ),
        ]

        response = await schedules_client.get(
            "/api/v1/schedules",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["schedules"][0]["schedule_id"] == "sched-1"


@pytest.mark.unit
class TestGetSchedule:
    async def test_get_schedule(self, schedules_client, auth_headers, mock_schedule_manager):
        mock_schedule_manager.get_schedule.return_value = ScheduleDefinition(
            name="daily",
            agent_type="report",
            prompt="Daily report",
            cron_expression="0 9 * * *",
        )

        response = await schedules_client.get(
            "/api/v1/schedules/sched-1",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["schedule_id"] == "sched-1"
        assert data["name"] == "daily"

    async def test_get_schedule_not_found(self, schedules_client, auth_headers, mock_schedule_manager):
        mock_schedule_manager.get_schedule.return_value = None

        response = await schedules_client.get(
            "/api/v1/schedules/nonexistent",
            headers=auth_headers,
        )
        assert response.status_code == 404


@pytest.mark.unit
class TestDeleteSchedule:
    async def test_delete_schedule(self, schedules_client, auth_headers, mock_schedule_manager):
        mock_schedule_manager.remove_schedule.return_value = True

        response = await schedules_client.delete(
            "/api/v1/schedules/sched-1",
            headers=auth_headers,
        )
        assert response.status_code == 204

    async def test_delete_schedule_not_found(self, schedules_client, auth_headers, mock_schedule_manager):
        mock_schedule_manager.remove_schedule.return_value = False

        response = await schedules_client.delete(
            "/api/v1/schedules/nonexistent",
            headers=auth_headers,
        )
        assert response.status_code == 404


@pytest.mark.unit
class TestSchedulesRequireAuth:
    async def test_no_auth_returns_401(self, schedules_client):
        response = await schedules_client.get("/api/v1/schedules")
        assert response.status_code == 401

    async def test_wrong_key_returns_401(self, schedules_client):
        response = await schedules_client.get(
            "/api/v1/schedules",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
