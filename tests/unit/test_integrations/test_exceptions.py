"""Tests for integration exception hierarchy."""

from __future__ import annotations

import pytest

from ia_agent_fwk.integrations.exceptions import (
    ChannelConfigError,
    ChannelConnectionError,
    IntegrationError,
    MessageDeliveryError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    def test_integration_error_is_exception(self):
        exc = IntegrationError("base error")
        assert isinstance(exc, Exception)
        assert str(exc) == "base error"

    def test_channel_connection_error_inherits(self):
        exc = ChannelConnectionError("connection failed")
        assert isinstance(exc, IntegrationError)
        assert isinstance(exc, Exception)
        assert str(exc) == "connection failed"

    def test_message_delivery_error_inherits(self):
        exc = MessageDeliveryError("delivery failed")
        assert isinstance(exc, IntegrationError)
        assert str(exc) == "delivery failed"

    def test_channel_config_error_inherits(self):
        exc = ChannelConfigError("bad config")
        assert isinstance(exc, IntegrationError)
        assert str(exc) == "bad config"

    def test_can_catch_all_via_base(self):
        errors = [
            ChannelConnectionError("conn"),
            MessageDeliveryError("delivery"),
            ChannelConfigError("config"),
        ]
        for err in errors:
            with pytest.raises(IntegrationError):
                raise err
