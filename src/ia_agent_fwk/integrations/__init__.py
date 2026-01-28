"""Pre-built channel integrations (Slack, Email, WhatsApp)."""

from __future__ import annotations

from ia_agent_fwk.integrations.base import ChannelIntegration
from ia_agent_fwk.integrations.exceptions import (
    ChannelConfigError,
    ChannelConnectionError,
    IntegrationError,
    MessageDeliveryError,
)
from ia_agent_fwk.integrations.models import ChannelConfig, IncomingMessage, OutgoingMessage
from ia_agent_fwk.integrations.router import ChannelRouter

__all__ = [
    "ChannelConfig",
    "ChannelConfigError",
    "ChannelConnectionError",
    "ChannelIntegration",
    "ChannelRouter",
    "IncomingMessage",
    "IntegrationError",
    "MessageDeliveryError",
    "OutgoingMessage",
]
