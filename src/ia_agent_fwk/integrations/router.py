"""Channel router -- routes incoming events to agents and sends responses.

The ``ChannelRouter`` manages registered channels and orchestrates the
incoming-event -> agent-execution -> outgoing-response flow.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ia_agent_fwk.integrations.exceptions import ChannelConfigError
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import LLMSettings
    from ia_agent_fwk.integrations.base import ChannelIntegration
    from ia_agent_fwk.integrations.models import ChannelConfig

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class ChannelRouter:
    """Routes incoming channel events to agents and sends back responses."""

    def __init__(self) -> None:
        self._channels: dict[str, ChannelIntegration] = {}
        self._configs: dict[str, ChannelConfig] = {}

    def register(self, channel: ChannelIntegration, config: ChannelConfig) -> None:
        """Register a channel integration with its configuration."""
        channel_type = channel.channel_type
        self._channels[channel_type] = channel
        self._configs[channel_type] = config
        collector = get_metrics_collector()
        collector.increment(
            "integration_channels_registered_total",
            labels={"channel": channel_type},
        )
        logger.info(
            "Registered channel integration: %s",
            channel_type,
            extra={
                "integration_data": {
                    "event": "channel_registered",
                    "channel": channel_type,
                    "agent_type": config.agent_type,
                    "total_channels": len(self._channels),
                }
            },
        )

    def get_channel(self, channel_type: str) -> ChannelIntegration | None:
        """Return a registered channel, or ``None`` if not found."""
        return self._channels.get(channel_type)

    def list_channels(self) -> list[str]:
        """Return a sorted list of registered channel type names."""
        return sorted(self._channels.keys())

    async def route_incoming(
        self,
        channel_type: str,
        raw_event: dict[str, object],
        llm_settings: LLMSettings,
    ) -> str | None:
        """Route an incoming event to the appropriate agent and return the response.

        Parameters
        ----------
        channel_type:
            The channel that received the event.
        raw_event:
            The raw webhook/event payload.
        llm_settings:
            LLM settings for agent creation.

        Returns
        -------
        str | None
            The agent's response text, or ``None`` if the event was
            ignored or the channel is not registered.

        """
        collector = get_metrics_collector()
        start = time.monotonic()

        with _tracer.start_as_current_span(
            "integration.route_incoming",
            attributes={"integration.channel": channel_type},
        ) as span:
            channel = self._channels.get(channel_type)
            config = self._configs.get(channel_type)

            if channel is None or config is None:
                collector.increment(
                    "integration_route_total",
                    labels={"channel": channel_type, "outcome": "channel_not_found"},
                )
                logger.warning(
                    "No channel registered for type: %s",
                    channel_type,
                    extra={
                        "integration_data": {
                            "event": "route_channel_not_found",
                            "channel": channel_type,
                        }
                    },
                )
                return None

            if not config.agent_type:
                msg = f"No agent_type configured for channel '{channel_type}'"
                raise ChannelConfigError(msg)

            # Parse the raw event
            incoming = await channel.process_incoming(raw_event)
            if incoming is None:
                collector.increment(
                    "integration_route_total",
                    labels={"channel": channel_type, "outcome": "ignored"},
                )
                logger.debug(
                    "Event ignored by channel '%s'",
                    channel_type,
                    extra={
                        "integration_data": {
                            "event": "route_event_ignored",
                            "channel": channel_type,
                        }
                    },
                )
                return None

            # Create and run the agent
            from ia_agent_fwk.agents.config import AgentConfig  # noqa: PLC0415
            from ia_agent_fwk.agents.factory import AgentFactory  # noqa: PLC0415

            agent_config = AgentConfig(
                name=f"{config.agent_type}-{channel_type}",
                agent_type=config.agent_type,
                provider_name=llm_settings.default_provider,
            )
            agent = AgentFactory.create(agent_config, llm_settings)
            result = await agent.run(incoming.content)

            # Send the response back through the channel
            from ia_agent_fwk.integrations.models import OutgoingMessage  # noqa: PLC0415

            outgoing = OutgoingMessage(
                channel=channel_type,
                recipient=incoming.sender,
                content=result.output,
            )
            await channel.send_message(outgoing)

            duration_ms = (time.monotonic() - start) * 1000
            span.set_attribute("integration.duration_ms", duration_ms)
            span.set_attribute("integration.sender", incoming.sender)
            span.set_attribute("integration.agent_type", config.agent_type)
            collector.increment(
                "integration_route_total",
                labels={"channel": channel_type, "outcome": "routed"},
            )
            collector.observe(
                "integration_route_duration_seconds",
                duration_ms / 1000,
            )
            logger.info(
                "Message routed: channel=%s, sender=%s, agent=%s (%.1fms)",
                channel_type,
                incoming.sender,
                config.agent_type,
                duration_ms,
                extra={
                    "integration_data": {
                        "event": "message_routed",
                        "channel": channel_type,
                        "sender": incoming.sender,
                        "agent_type": config.agent_type,
                        "duration_ms": round(duration_ms, 1),
                    }
                },
            )

            return result.output
