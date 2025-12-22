"""Pydantic v2 models for the execution layer."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    """Status of an asynchronous agent execution job."""

    PENDING = "pending"
    STARTED = "running"
    SUCCESS = "completed"
    FAILURE = "failed"
    REVOKED = "cancelled"
    UNKNOWN = "unknown"


class JobInfo(BaseModel):
    """Internal DTO describing a job's metadata."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    agent_type: str | None = None
    status: JobStatus = JobStatus.UNKNOWN
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class JobSubmitRequest(BaseModel):
    """Request body for submitting an async agent execution."""

    agent_type: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=100_000)
    config_overrides: dict[str, object] | None = None
    conversation_id: str | None = None


class JobSubmitResponse(BaseModel):
    """Response body after submitting an async job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: str = "pending"
    status_url: str = ""


class JobStatusResponse(BaseModel):
    """Response body for querying a job's status and result."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: str
    agent_type: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


# ============================================================================
# Schedule models
# ============================================================================


class ScheduleDefinition(BaseModel):
    """Defines a cron-based schedule for agent execution."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=256)
    agent_type: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=100_000)
    cron_expression: str = Field(..., min_length=1)
    enabled: bool = True
    config_overrides: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ScheduleCreateRequest(BaseModel):
    """Request body for creating a schedule."""

    name: str = Field(..., min_length=1, max_length=256)
    agent_type: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=100_000)
    cron_expression: str = Field(..., min_length=1)
    enabled: bool = True
    config_overrides: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ScheduleResponse(BaseModel):
    """Response body for a schedule."""

    model_config = ConfigDict(frozen=True)

    schedule_id: str
    name: str
    agent_type: str
    prompt: str
    cron_expression: str
    enabled: bool
    config_overrides: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


# ============================================================================
# Trigger models
# ============================================================================


class EventTrigger(BaseModel):
    """Defines an event-driven trigger for agent execution."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=256)
    agent_type: str = Field(..., min_length=1)
    prompt_template: str = Field(..., min_length=1, max_length=100_000)
    event_type: str = Field(..., min_length=1)
    config_overrides: dict[str, Any] | None = None


class TriggerCreateRequest(BaseModel):
    """Request body for registering a trigger."""

    name: str = Field(..., min_length=1, max_length=256)
    agent_type: str = Field(..., min_length=1)
    prompt_template: str = Field(..., min_length=1, max_length=100_000)
    event_type: str = Field(..., min_length=1)
    config_overrides: dict[str, Any] | None = None


class TriggerResponse(BaseModel):
    """Response body for a trigger."""

    model_config = ConfigDict(frozen=True)

    trigger_id: str
    name: str
    agent_type: str
    prompt_template: str
    event_type: str
    config_overrides: dict[str, Any] | None = None


class WebhookPayload(BaseModel):
    """Incoming webhook event payload."""

    data: dict[str, Any] = Field(default_factory=dict)


class WebhookResponse(BaseModel):
    """Response from firing a webhook event."""

    model_config = ConfigDict(frozen=True)

    event_type: str
    trigger_id: str
    job_id: str
    status: str = "submitted"
