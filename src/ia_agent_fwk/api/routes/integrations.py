"""Webhook endpoints for channel integrations."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Security
from starlette.responses import JSONResponse, PlainTextResponse

from ia_agent_fwk.api.dependencies import check_rate_limit, get_settings, require_api_key
from ia_agent_fwk.config.settings import AppSettings  # noqa: TC001
from ia_agent_fwk.integrations.exceptions import ChannelConfigError, IntegrationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/integrations", tags=["integrations"])


# ------------------------------------------------------------------
# List integrations (authenticated)
# ------------------------------------------------------------------


@router.get("", dependencies=[Security(require_api_key), Depends(check_rate_limit)])
async def list_integrations(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> JSONResponse:
    """List configured integrations and their status."""
    integrations: list[dict[str, object]] = [
        {
            "channel": "slack",
            "enabled": settings.integrations.slack.enabled,
        },
        {
            "channel": "email",
            "enabled": settings.integrations.email.enabled,
        },
        {
            "channel": "whatsapp",
            "enabled": settings.integrations.whatsapp.enabled,
        },
    ]
    return JSONResponse(content={"integrations": integrations})


# ------------------------------------------------------------------
# Slack webhook
# ------------------------------------------------------------------


@router.post("/slack/webhook")
async def slack_webhook(
    request: Request,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> JSONResponse:
    """Handle inbound Slack Events API payloads.

    Handles the URL verification challenge and message events.
    """
    body: dict[str, Any] = await request.json()

    # URL verification challenge
    if body.get("type") == "url_verification":
        return JSONResponse(content={"challenge": body.get("challenge", "")})

    if not settings.integrations.slack.enabled:
        return JSONResponse(
            content={"error": "Slack integration is not enabled"},
            status_code=403,
        )

    # Process the event through the channel router
    try:
        channel_router = _get_channel_router(request)
        response_text = await channel_router.route_incoming(
            "slack",
            body,
            settings.llm,
        )
    except (ChannelConfigError, IntegrationError) as exc:
        logger.exception(
            "Slack webhook error",
            extra={
                "integration_data": {
                    "event": "webhook_error",
                    "channel": "slack",
                    "error": str(exc),
                }
            },
        )
        return JSONResponse(
            content={"error": str(exc)},
            status_code=500,
        )

    return JSONResponse(content={"ok": True, "response": response_text})


# ------------------------------------------------------------------
# Email webhook
# ------------------------------------------------------------------


@router.post("/email/webhook")
async def email_webhook(
    request: Request,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> JSONResponse:
    """Handle inbound email webhook payloads (e.g. SendGrid/Mailgun)."""
    if not settings.integrations.email.enabled:
        return JSONResponse(
            content={"error": "Email integration is not enabled"},
            status_code=403,
        )

    body: dict[str, Any] = await request.json()

    try:
        channel_router = _get_channel_router(request)
        response_text = await channel_router.route_incoming(
            "email",
            body,
            settings.llm,
        )
    except (ChannelConfigError, IntegrationError) as exc:
        logger.exception(
            "Email webhook error",
            extra={
                "integration_data": {
                    "event": "webhook_error",
                    "channel": "email",
                    "error": str(exc),
                }
            },
        )
        return JSONResponse(
            content={"error": str(exc)},
            status_code=500,
        )

    return JSONResponse(content={"ok": True, "response": response_text})


# ------------------------------------------------------------------
# WhatsApp webhook
# ------------------------------------------------------------------


@router.get("/whatsapp/webhook", response_model=None)
async def whatsapp_verify(
    request: Request,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> PlainTextResponse | JSONResponse:
    """Handle WhatsApp webhook verification challenge."""
    mode = request.query_params.get("hub.mode", "")
    token = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")

    verify_token = settings.integrations.whatsapp.verify_token
    if mode == "subscribe" and token == verify_token:
        return PlainTextResponse(content=challenge)

    return JSONResponse(content={"error": "Verification failed"}, status_code=403)


@router.post("/whatsapp/webhook")
async def whatsapp_webhook(
    request: Request,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> JSONResponse:
    """Handle inbound WhatsApp webhook payloads."""
    if not settings.integrations.whatsapp.enabled:
        return JSONResponse(
            content={"error": "WhatsApp integration is not enabled"},
            status_code=403,
        )

    body: dict[str, Any] = await request.json()

    try:
        channel_router = _get_channel_router(request)
        response_text = await channel_router.route_incoming(
            "whatsapp",
            body,
            settings.llm,
        )
    except (ChannelConfigError, IntegrationError) as exc:
        logger.exception(
            "WhatsApp webhook error",
            extra={
                "integration_data": {
                    "event": "webhook_error",
                    "channel": "whatsapp",
                    "error": str(exc),
                }
            },
        )
        return JSONResponse(
            content={"error": str(exc)},
            status_code=500,
        )

    return JSONResponse(content={"ok": True, "response": response_text})


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _get_channel_router(request: Request) -> Any:
    """Retrieve ``ChannelRouter`` from ``app.state``."""
    router_instance: Any = getattr(request.app.state, "channel_router", None)
    if router_instance is None:
        from ia_agent_fwk.integrations.router import ChannelRouter  # noqa: PLC0415

        router_instance = ChannelRouter()
        request.app.state.channel_router = router_instance
    return router_instance
