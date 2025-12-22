"""Tests for JobManager with mocked Celery app."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ia_agent_fwk.execution.models import JobStatus


@pytest.mark.unit
class TestJobManagerSubmit:
    @patch("ia_agent_fwk.execution.tasks.execute_agent_task")
    def test_submit_returns_job_id(self, mock_task, job_manager):
        mock_result = MagicMock()
        mock_result.id = "job-abc-123"
        mock_task.apply_async.return_value = mock_result

        job_id = job_manager.submit(agent_type="test", prompt="Hello")

        assert job_id == "job-abc-123"
        mock_task.apply_async.assert_called_once()

    @patch("ia_agent_fwk.execution.tasks.execute_agent_task")
    def test_submit_stores_in_redis_index(self, mock_task, job_manager, mock_redis):
        mock_result = MagicMock()
        mock_result.id = "job-xyz"
        mock_task.apply_async.return_value = mock_result

        job_manager.submit(agent_type="test", prompt="Hello")

        mock_redis.zadd.assert_called_once()
        mock_redis.hset.assert_called_once()

    @patch("ia_agent_fwk.execution.tasks.execute_agent_task")
    def test_submit_without_redis(self, mock_task, job_manager_no_redis):
        mock_result = MagicMock()
        mock_result.id = "job-no-redis"
        mock_task.apply_async.return_value = mock_result

        job_id = job_manager_no_redis.submit(agent_type="test", prompt="Hello")

        assert job_id == "job-no-redis"

    @patch("ia_agent_fwk.execution.tasks.execute_agent_task")
    def test_submit_passes_kwargs(self, mock_task, job_manager):
        mock_result = MagicMock()
        mock_result.id = "job-999"
        mock_task.apply_async.return_value = mock_result

        job_manager.submit(
            agent_type="test",
            prompt="Hello",
            conversation_id="conv-1",
            config_overrides={"temperature": 0.5},
        )

        call_kwargs = mock_task.apply_async.call_args
        assert call_kwargs.kwargs["kwargs"]["conversation_id"] == "conv-1"
        assert call_kwargs.kwargs["kwargs"]["config_overrides"] == {"temperature": 0.5}


@pytest.mark.unit
class TestJobManagerGetStatus:
    def test_pending_status(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "PENDING"
        mock_async_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_async_result

        info = job_manager.get_status("job-123")

        assert info.job_id == "job-123"
        assert info.status == JobStatus.PENDING

    def test_started_status(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "STARTED"
        mock_async_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_async_result

        info = job_manager.get_status("job-123")
        assert info.status == JobStatus.STARTED

    def test_success_status(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "SUCCESS"
        mock_async_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_async_result

        info = job_manager.get_status("job-123")
        assert info.status == JobStatus.SUCCESS

    def test_failure_status_with_error(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "FAILURE"
        mock_async_result.result = RuntimeError("boom")
        mock_celery_app.AsyncResult.return_value = mock_async_result

        info = job_manager.get_status("job-123")
        assert info.status == JobStatus.FAILURE
        assert info.error == "boom"

    def test_revoked_status(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "REVOKED"
        mock_async_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_async_result

        info = job_manager.get_status("job-123")
        assert info.status == JobStatus.REVOKED

    def test_running_custom_state(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "RUNNING"
        mock_async_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_async_result

        info = job_manager.get_status("job-123")
        assert info.status == JobStatus.STARTED

    def test_unknown_state(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "SOME_WEIRD_STATE"
        mock_async_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_async_result

        info = job_manager.get_status("job-123")
        assert info.status == JobStatus.UNKNOWN

    def test_status_with_redis_metadata(self, job_manager, mock_celery_app, mock_redis):
        mock_async_result = MagicMock()
        mock_async_result.state = "STARTED"
        mock_async_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_async_result

        mock_redis.hgetall.return_value = {
            "agent_type": "conversational",
            "created_at": "1709942400.0",
        }

        info = job_manager.get_status("job-123")
        assert info.agent_type == "conversational"
        assert info.created_at == "1709942400.0"


@pytest.mark.unit
class TestJobManagerGetResult:
    def test_result_when_ready(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.ready.return_value = True
        mock_async_result.result = {"output": "Hello!"}
        mock_celery_app.AsyncResult.return_value = mock_async_result

        result = job_manager.get_result("job-123")
        assert result == {"output": "Hello!"}

    def test_result_when_not_ready(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.ready.return_value = False
        mock_celery_app.AsyncResult.return_value = mock_async_result

        result = job_manager.get_result("job-123")
        assert result is None

    def test_result_when_not_dict(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.ready.return_value = True
        mock_async_result.result = "not-a-dict"
        mock_celery_app.AsyncResult.return_value = mock_async_result

        result = job_manager.get_result("job-123")
        assert result is None


@pytest.mark.unit
class TestJobManagerCancel:
    def test_cancel_pending_job(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "PENDING"
        mock_celery_app.AsyncResult.return_value = mock_async_result

        cancelled = job_manager.cancel("job-123")

        assert cancelled is True
        mock_async_result.revoke.assert_called_once_with(terminate=True, signal="SIGTERM")

    def test_cancel_running_job(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "STARTED"
        mock_celery_app.AsyncResult.return_value = mock_async_result

        cancelled = job_manager.cancel("job-123")
        assert cancelled is True

    def test_cancel_completed_job_returns_false(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "SUCCESS"
        mock_celery_app.AsyncResult.return_value = mock_async_result

        cancelled = job_manager.cancel("job-123")
        assert cancelled is False
        mock_async_result.revoke.assert_not_called()

    def test_cancel_already_revoked(self, job_manager, mock_celery_app):
        mock_async_result = MagicMock()
        mock_async_result.state = "REVOKED"
        mock_celery_app.AsyncResult.return_value = mock_async_result

        cancelled = job_manager.cancel("job-123")
        assert cancelled is False


@pytest.mark.unit
class TestJobManagerListJobs:
    def test_list_empty(self, job_manager, mock_redis):
        mock_redis.zcard.return_value = 0
        mock_redis.zrevrange.return_value = []

        jobs, total = job_manager.list_jobs()

        assert jobs == []
        assert total == 0

    def test_list_without_redis(self, job_manager_no_redis):
        jobs, total = job_manager_no_redis.list_jobs()
        assert jobs == []
        assert total == 0

    def test_list_with_jobs(self, job_manager, mock_celery_app, mock_redis):
        mock_redis.zcard.return_value = 2
        mock_redis.zrevrange.return_value = [b"job-1", b"job-2"]
        mock_redis.hgetall.return_value = {"agent_type": "test"}

        mock_async_result = MagicMock()
        mock_async_result.state = "SUCCESS"
        mock_async_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_async_result

        jobs, total = job_manager.list_jobs(limit=10, offset=0)

        assert total == 2
        assert len(jobs) == 2
        assert jobs[0].job_id == "job-1"

    def test_list_pagination(self, job_manager, mock_redis):
        job_manager.list_jobs(limit=5, offset=10)

        mock_redis.zrevrange.assert_called_once_with(
            "ia_agent_fwk:jobs",
            10,
            14,
        )
