"""Job management API routes for async agent execution."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Security

from ia_agent_fwk.api.dependencies import check_rate_limit, get_job_manager, require_api_key
from ia_agent_fwk.execution.manager import JobManager  # noqa: TC001
from ia_agent_fwk.execution.models import JobStatus, JobStatusResponse, JobSubmitRequest

router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["jobs"],
    dependencies=[Security(require_api_key), Depends(check_rate_limit)],
)


# ---------------------------------------------------------------------------
# POST /api/v1/jobs — Submit async agent execution
# ---------------------------------------------------------------------------


@router.post("", status_code=202)
async def submit_job(
    body: JobSubmitRequest,
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
) -> dict[str, Any]:
    """Submit an async agent execution job.

    Returns ``202 Accepted`` with a ``job_id``.
    """
    from ia_agent_fwk.agents.registry import AgentRegistry  # noqa: PLC0415

    AgentRegistry.get(body.agent_type)

    job_id = job_manager.submit(agent_type=body.agent_type, prompt=body.prompt)
    return {
        "job_id": job_id,
        "status": "pending",
        "status_url": f"/api/v1/jobs/{job_id}",
    }


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id} — Get job status and result
# ---------------------------------------------------------------------------


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
) -> JobStatusResponse:
    """Return the current status and result (if complete) for a job."""
    info = job_manager.get_status(job_id)

    result_data: dict[str, object] | None = None
    if info.status == JobStatus.SUCCESS:
        result_data = job_manager.get_result(job_id)

    return JobStatusResponse(
        job_id=info.job_id,
        status=info.status.value,
        agent_type=info.agent_type,
        result=result_data,
        error=info.error,
        created_at=info.created_at,
    )


# ---------------------------------------------------------------------------
# DELETE /api/v1/jobs/{job_id} — Cancel a running job
# ---------------------------------------------------------------------------


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
) -> dict[str, str]:
    """Cancel a running or pending job."""
    cancelled = job_manager.cancel(job_id)
    if cancelled:
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled successfully",
        }
    return {
        "job_id": job_id,
        "status": "already_completed",
        "message": "Job is already in a terminal state",
    }


# ---------------------------------------------------------------------------
# GET /api/v1/jobs — List recent jobs
# ---------------------------------------------------------------------------


@router.get("")
async def list_jobs(
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """List recent jobs with pagination."""
    jobs, total = job_manager.list_jobs(limit=limit, offset=offset)
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "agent_type": j.agent_type,
                "created_at": j.created_at,
            }
            for j in jobs
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
