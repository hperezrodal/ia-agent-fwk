"""Tests for job API endpoints with mocked JobManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from ia_agent_fwk.agents.registry import AgentRegistry
from ia_agent_fwk.api.app import create_app
from ia_agent_fwk.config.settings import AppSettings, AuthSettings, MemorySettings
from ia_agent_fwk.execution.manager import JobManager
from ia_agent_fwk.execution.models import JobInfo, JobStatus
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend


@pytest.fixture
def mock_job_manager():
    """Create a mock JobManager."""
    return MagicMock(spec=JobManager)


@pytest.fixture
def test_settings_for_jobs(monkeypatch):
    monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
    return AppSettings(
        auth=AuthSettings(enabled=True),
        memory=MemorySettings(default_backend="in_memory"),
    )


@pytest.fixture
def test_app_with_jobs(test_settings_for_jobs, mock_job_manager):
    """Create a test FastAPI app with a mock JobManager."""
    # Register a fake agent type for validation
    if "test" not in AgentRegistry._registry:
        from tests.unit.test_api.conftest import _TestAgent

        AgentRegistry.register("test", _TestAgent)

    with (
        patch("ia_agent_fwk.api.app.get_celery_app"),
        patch("ia_agent_fwk.api.app.JobManager", return_value=mock_job_manager),
    ):
        app = create_app(test_settings_for_jobs)

    app.state.memory_backend = InMemoryBackend()
    app.state.conversation_backend = ConversationMemoryBackend()
    app.state.job_manager = mock_job_manager

    yield app

    AgentRegistry._registry.pop("test", None)


@pytest.fixture
async def jobs_client(test_app_with_jobs):
    transport = httpx.ASGITransport(app=test_app_with_jobs)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key-1"}


@pytest.mark.unit
class TestSubmitJobEndpoint:
    async def test_submit_job_with_request_body(self, jobs_client, auth_headers, mock_job_manager):
        """The submit endpoint on /api/v1/jobs accepts a JSON request body."""
        mock_job_manager.submit.return_value = "job-abc-123"

        response = await jobs_client.post(
            "/api/v1/jobs",
            headers=auth_headers,
            json={"agent_type": "test", "prompt": "Hello"},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "job-abc-123"
        assert data["status"] == "pending"
        assert "/api/v1/jobs/job-abc-123" in data["status_url"]

    async def test_submit_job_missing_prompt_returns_422(self, jobs_client, auth_headers):
        """Missing required field returns 422 Unprocessable Entity."""
        response = await jobs_client.post(
            "/api/v1/jobs",
            headers=auth_headers,
            json={"agent_type": "test"},
        )
        assert response.status_code == 422

    async def test_submit_job_empty_prompt_returns_422(self, jobs_client, auth_headers):
        """Empty prompt violates min_length=1 and returns 422."""
        response = await jobs_client.post(
            "/api/v1/jobs",
            headers=auth_headers,
            json={"agent_type": "test", "prompt": ""},
        )
        assert response.status_code == 422


@pytest.mark.unit
class TestGetJobStatusEndpoint:
    async def test_get_job_status_pending(self, jobs_client, auth_headers, mock_job_manager):
        mock_job_manager.get_status.return_value = JobInfo(
            job_id="job-123",
            status=JobStatus.PENDING,
            agent_type="test",
        )
        mock_job_manager.get_result.return_value = None

        response = await jobs_client.get(
            "/api/v1/jobs/job-123",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-123"
        assert data["status"] == "pending"
        assert data["result"] is None

    async def test_get_job_status_completed(self, jobs_client, auth_headers, mock_job_manager):
        mock_job_manager.get_status.return_value = JobInfo(
            job_id="job-123",
            status=JobStatus.SUCCESS,
            agent_type="test",
        )
        mock_job_manager.get_result.return_value = {"output": "Hello!"}

        response = await jobs_client.get(
            "/api/v1/jobs/job-123",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"] == {"output": "Hello!"}


@pytest.mark.unit
class TestCancelJobEndpoint:
    async def test_cancel_job_success(self, jobs_client, auth_headers, mock_job_manager):
        mock_job_manager.cancel.return_value = True

        response = await jobs_client.delete(
            "/api/v1/jobs/job-123",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    async def test_cancel_completed_job(self, jobs_client, auth_headers, mock_job_manager):
        mock_job_manager.cancel.return_value = False

        response = await jobs_client.delete(
            "/api/v1/jobs/job-123",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_completed"


@pytest.mark.unit
class TestListJobsEndpoint:
    async def test_list_jobs_empty(self, jobs_client, auth_headers, mock_job_manager):
        mock_job_manager.list_jobs.return_value = ([], 0)

        response = await jobs_client.get(
            "/api/v1/jobs",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    async def test_list_jobs_with_pagination(self, jobs_client, auth_headers, mock_job_manager):
        mock_job_manager.list_jobs.return_value = (
            [
                JobInfo(job_id="job-1", status=JobStatus.SUCCESS, agent_type="test"),
                JobInfo(job_id="job-2", status=JobStatus.PENDING, agent_type="test"),
            ],
            5,
        )

        response = await jobs_client.get(
            "/api/v1/jobs?limit=2&offset=0",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 0


@pytest.mark.unit
class TestAuthRequired:
    async def test_no_auth_returns_401(self, jobs_client):
        response = await jobs_client.get("/api/v1/jobs")
        assert response.status_code == 401

    async def test_wrong_key_returns_401(self, jobs_client):
        response = await jobs_client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
