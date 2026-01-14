"""Event trigger and webhook API routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Security
from starlette.responses import JSONResponse

from ia_agent_fwk.api.dependencies import check_rate_limit, get_trigger_manager, require_api_key
from ia_agent_fwk.execution.models import (
    EventTrigger,
    TriggerCreateRequest,
    TriggerResponse,
    WebhookPayload,
    WebhookResponse,
)
from ia_agent_fwk.execution.triggers import TriggerManager  # noqa: TC001

router = APIRouter(
    prefix="/api/v1",
    tags=["triggers"],
    dependencies=[Depends(check_rate_limit)],
)


def _to_response(trigger_id: str, trigger: EventTrigger) -> TriggerResponse:
    return TriggerResponse(
        trigger_id=trigger_id,
        name=trigger.name,
        agent_type=trigger.agent_type,
        prompt_template=trigger.prompt_template,
        event_type=trigger.event_type,
        config_overrides=trigger.config_overrides,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/triggers — Register trigger
# ---------------------------------------------------------------------------


@router.post(
    "/triggers",
    status_code=201,
    dependencies=[Security(require_api_key)],
)
async def register_trigger(
    body: TriggerCreateRequest,
    trigger_manager: Annotated[TriggerManager, Depends(get_trigger_manager)],
) -> TriggerResponse:
    """Register a new event trigger."""
    trigger = EventTrigger(
        name=body.name,
        agent_type=body.agent_type,
        prompt_template=body.prompt_template,
        event_type=body.event_type,
        config_overrides=body.config_overrides,
    )
    trigger_id = trigger_manager.register_trigger(trigger)
    return _to_response(trigger_id, trigger)


# ---------------------------------------------------------------------------
# GET /api/v1/triggers — List triggers
# ---------------------------------------------------------------------------


@router.get(
    "/triggers",
    dependencies=[Security(require_api_key)],
)
async def list_triggers(
    trigger_manager: Annotated[TriggerManager, Depends(get_trigger_manager)],
) -> dict[str, Any]:
    """List all registered triggers."""
    triggers = trigger_manager.list_triggers()
    return {
        "triggers": [_to_response(tid, t).model_dump() for tid, t in triggers],
        "total": len(triggers),
    }


# ---------------------------------------------------------------------------
# DELETE /api/v1/triggers/{trigger_id} — Unregister trigger
# ---------------------------------------------------------------------------


@router.delete(
    "/triggers/{trigger_id}",
    status_code=204,
    dependencies=[Security(require_api_key)],
)
async def unregister_trigger(
    trigger_id: str,
    trigger_manager: Annotated[TriggerManager, Depends(get_trigger_manager)],
) -> None:
    """Unregister a trigger by ID."""
    removed = trigger_manager.unregister_trigger(trigger_id)
    if not removed:
        from ia_agent_fwk.execution.exceptions import TriggerNotFoundError  # noqa: PLC0415

        msg = f"Trigger not found: {trigger_id}"
        raise TriggerNotFoundError(msg)


# ---------------------------------------------------------------------------
# POST /api/v1/webhooks/{event_type} — Fire event trigger
# ---------------------------------------------------------------------------


@router.post(
    "/webhooks/{event_type}",
    dependencies=[Security(require_api_key)],
)
async def fire_webhook(
    event_type: str,
    body: WebhookPayload,
    trigger_manager: Annotated[TriggerManager, Depends(get_trigger_manager)],
) -> JSONResponse:
    """Fire an event trigger via webhook.

    Returns the job info if a matching trigger is found, or 404 otherwise.
    """
    result = trigger_manager.fire_trigger(event_type, body.data)
    if result is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "No trigger registered for event type",
                "event_type": event_type,
            },
        )
    trigger_id, job_id = result
    response = WebhookResponse(
        event_type=event_type,
        trigger_id=trigger_id,
        job_id=job_id,
    )
    return JSONResponse(status_code=200, content=response.model_dump())
