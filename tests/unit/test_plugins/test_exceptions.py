"""Tests for plugin exception hierarchy."""

from __future__ import annotations

import pytest

from ia_agent_fwk.plugins.exceptions import (
    PluginConfigError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
)


@pytest.mark.unit
class TestPluginExceptions:
    def test_plugin_error_is_base_exception(self):
        assert issubclass(PluginError, Exception)

    def test_plugin_load_error_inherits_plugin_error(self):
        assert issubclass(PluginLoadError, PluginError)

    def test_plugin_config_error_inherits_plugin_error(self):
        assert issubclass(PluginConfigError, PluginError)

    def test_plugin_not_found_error_inherits_plugin_error(self):
        assert issubclass(PluginNotFoundError, PluginError)

    def test_plugin_load_error_message_and_name(self):
        msg = "load failed"
        err = PluginLoadError(msg, plugin_name="my_plugin")
        assert str(err) == "load failed"
        assert err.plugin_name == "my_plugin"

    def test_plugin_load_error_default_plugin_name(self):
        msg = "load failed"
        err = PluginLoadError(msg)
        assert err.plugin_name == ""

    def test_plugin_config_error_message(self):
        msg = "bad config"
        err = PluginConfigError(msg)
        assert str(err) == "bad config"

    def test_plugin_not_found_error_message(self):
        msg = "not found"
        err = PluginNotFoundError(msg)
        assert str(err) == "not found"

    def test_all_exceptions_are_catchable_as_plugin_error(self):
        msg = "test"
        for exc_cls in (PluginLoadError, PluginConfigError, PluginNotFoundError):
            with pytest.raises(PluginError):
                raise exc_cls(msg)
