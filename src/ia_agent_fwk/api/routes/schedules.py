"""Schedule management API routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Security

from ia_agent_fwk.api.dependencies import check_rate_limit, get_schedule_manager, require_api_key
from ia_agent_fwk.execution.exceptions import ScheduleNotFoundError
from ia_agent_fwk.execution.models import (
    ScheduleCreateRequest,
    ScheduleDefinition,
    ScheduleResponse,
)
from ia_agent_fwk.execution.scheduler import ScheduleManager  # noqa: TC001

router = APIRouter(
    prefix="/api/v1/schedules",
    tags=["schedules"],
    dependencies=[Security(require_api_key), Depends(check_rate_limit)],
)


def _to_response(schedule_id: str, definition: ScheduleDefinition) -> ScheduleResponse:
    return ScheduleResponse(
        schedule_id=schedule_id,
        name=definition.name,
        agent_type=definition.agent_type,
        prompt=definition.prompt,
        cron_expression=definition.cron_expression,
        enabled=definition.enabled,
        config_overrides=definition.config_overrides,
        metadata=definition.metadata,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/schedules — Create schedule
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_schedule(
    body: ScheduleCreateRequest,
    schedule_manager: Annotated[ScheduleManager, Depends(get_schedule_manager)],
) -> ScheduleResponse:
    """Create a new cron-based schedule."""
    definition = ScheduleDefinition(
        name=body.name,
        agent_type=body.agent_type,
        prompt=body.prompt,
        cron_expression=body.cron_expression,
        enabled=body.enabled,
        config_overrides=body.config_overrides,
        metadata=body.metadata,
    )

    schedule_id = schedule_manager.add_schedule(definition)
    return _to_response(schedule_id, definition)


# ---------------------------------------------------------------------------
# GET /api/v1/schedules — List schedules
# ---------------------------------------------------------------------------


@router.get("")
async def list_schedules(
    schedule_manager: Annotated[ScheduleManager, Depends(get_schedule_manager)],
) -> dict[str, Any]:
    """List all registered schedules."""
    schedules = schedule_manager.list_schedules()
    return {
        "schedules": [_to_response(sid, defn).model_dump() for sid, defn in schedules],
        "total": len(schedules),
    }


# ---------------------------------------------------------------------------
# GET /api/v1/schedules/{schedule_id} — Get schedule
# ---------------------------------------------------------------------------


@router.get("/{schedule_id}")
async def get_schedule(
    schedule_id: str,
    schedule_manager: Annotated[ScheduleManager, Depends(get_schedule_manager)],
) -> ScheduleResponse:
    """Return a schedule by ID."""
    definition = schedule_manager.get_schedule(schedule_id)
    if definition is None:
        msg = f"Schedule not found: {schedule_id}"
        raise ScheduleNotFoundError(msg)
    return _to_response(schedule_id, definition)


# ---------------------------------------------------------------------------
# PUT /api/v1/schedules/{schedule_id} — Update schedule
# ---------------------------------------------------------------------------


@router.put("/{schedule_id}")
async def update_schedule(
    schedule_id: str,
    body: ScheduleCreateRequest,
    schedule_manager: Annotated[ScheduleManager, Depends(get_schedule_manager)],
) -> ScheduleResponse:
    """Update an existing schedule."""
    definition = ScheduleDefinition(
        name=body.name,
        agent_type=body.agent_type,
        prompt=body.prompt,
        cron_expression=body.cron_expression,
        enabled=body.enabled,
        config_overrides=body.config_overrides,
        metadata=body.metadata,
    )
    schedule_manager.update_schedule(schedule_id, definition)
    return _to_response(schedule_id, definition)


# ---------------------------------------------------------------------------
# DELETE /api/v1/schedules/{schedule_id} — Delete schedule
# ---------------------------------------------------------------------------


@router.delete("/{schedule_id}", status_code=204)
async def delete_schedule(
    schedule_id: str,
    schedule_manager: Annotated[ScheduleManager, Depends(get_schedule_manager)],
) -> None:
    """Delete a schedule by ID."""
    removed = schedule_manager.remove_schedule(schedule_id)
    if not removed:
        msg = f"Schedule not found: {schedule_id}"
        raise ScheduleNotFoundError(msg)
