"""Integration exception hierarchy.

All integration-specific exceptions inherit from ``IntegrationError``.
"""

from __future__ import annotations


class IntegrationError(Exception):
    """Base exception for all integration errors."""


class ChannelConnectionError(IntegrationError):
    """Raised when a channel connection cannot be established."""


class MessageDeliveryError(IntegrationError):
    """Raised when a message cannot be delivered to the channel."""


class ChannelConfigError(IntegrationError):
    """Raised when channel configuration is invalid or missing."""
