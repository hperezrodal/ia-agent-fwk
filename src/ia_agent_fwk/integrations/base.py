"""Channel integration abstract base class.

``ChannelIntegration`` defines the contract that every channel
(Slack, Email, WhatsApp, etc.) must satisfy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ia_agent_fwk.integrations.models import IncomingMessage, OutgoingMessage


class ChannelIntegration(ABC):
    """Abstract base class for channel integrations."""

    @property
    @abstractmethod
    def channel_type(self) -> str:
        """Return the channel type identifier (e.g. ``'slack'``)."""
        ...

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send a message through the channel.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abstractmethod
    async def process_incoming(self, raw_event: dict[str, object]) -> IncomingMessage | None:
        """Parse a raw webhook/event payload into an ``IncomingMessage``.

        Returns ``None`` if the event should be ignored (e.g. bot messages).
        """
        ...

    async def start(self) -> None:  # noqa: B027
        """Initialise the channel connection (default no-op)."""

    async def stop(self) -> None:  # noqa: B027
        """Close the channel connection (default no-op)."""

    async def health_check(self) -> bool:
        """Check whether the channel is operational.

        Returns ``True`` by default.
        """
        return True
