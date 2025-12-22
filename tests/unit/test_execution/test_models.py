"""Tests for execution layer Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ia_agent_fwk.execution.models import (
    JobInfo,
    JobStatus,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
)


@pytest.mark.unit
class TestJobStatus:
    def test_pending_value(self):
        assert JobStatus.PENDING.value == "pending"

    def test_started_value(self):
        assert JobStatus.STARTED.value == "running"

    def test_success_value(self):
        assert JobStatus.SUCCESS.value == "completed"

    def test_failure_value(self):
        assert JobStatus.FAILURE.value == "failed"

    def test_revoked_value(self):
        assert JobStatus.REVOKED.value == "cancelled"

    def test_unknown_value(self):
        assert JobStatus.UNKNOWN.value == "unknown"

    def test_is_str_enum(self):
        assert isinstance(JobStatus.PENDING, str)


@pytest.mark.unit
class TestJobInfo:
    def test_minimal(self):
        info = JobInfo(job_id="abc-123")
        assert info.job_id == "abc-123"
        assert info.status == JobStatus.UNKNOWN
        assert info.agent_type is None
        assert info.error is None

    def test_full(self):
        info = JobInfo(
            job_id="abc-123",
            agent_type="conversational",
            status=JobStatus.SUCCESS,
            created_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:05Z",
        )
        assert info.agent_type == "conversational"
        assert info.status == JobStatus.SUCCESS

    def test_frozen(self):
        info = JobInfo(job_id="abc-123")
        with pytest.raises(ValidationError):
            info.job_id = "new-id"  # type: ignore[misc]


@pytest.mark.unit
class TestJobSubmitRequest:
    def test_valid(self):
        req = JobSubmitRequest(agent_type="test", prompt="Hello")
        assert req.agent_type == "test"
        assert req.prompt == "Hello"
        assert req.config_overrides is None
        assert req.conversation_id is None

    def test_empty_agent_type_rejected(self):
        with pytest.raises(ValidationError):
            JobSubmitRequest(agent_type="", prompt="Hello")

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            JobSubmitRequest(agent_type="test", prompt="")

    def test_with_overrides(self):
        req = JobSubmitRequest(
            agent_type="test",
            prompt="Hello",
            config_overrides={"temperature": 0.5},
            conversation_id="conv-1",
        )
        assert req.config_overrides == {"temperature": 0.5}
        assert req.conversation_id == "conv-1"


@pytest.mark.unit
class TestJobSubmitResponse:
    def test_defaults(self):
        resp = JobSubmitResponse(job_id="abc-123")
        assert resp.job_id == "abc-123"
        assert resp.status == "pending"
        assert resp.status_url == ""

    def test_frozen(self):
        resp = JobSubmitResponse(job_id="abc-123")
        with pytest.raises(ValidationError):
            resp.job_id = "new-id"  # type: ignore[misc]


@pytest.mark.unit
class TestJobStatusResponse:
    def test_minimal(self):
        resp = JobStatusResponse(job_id="abc-123", status="pending")
        assert resp.job_id == "abc-123"
        assert resp.status == "pending"
        assert resp.result is None

    def test_with_result(self):
        resp = JobStatusResponse(
            job_id="abc-123",
            status="completed",
            agent_type="test",
            result={"output": "Hello"},
        )
        assert resp.result == {"output": "Hello"}
        assert resp.agent_type == "test"

    def test_with_error(self):
        resp = JobStatusResponse(
            job_id="abc-123",
            status="failed",
            error="Something went wrong",
        )
        assert resp.error == "Something went wrong"
