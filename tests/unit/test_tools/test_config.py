"""Tests for tool configuration models."""

import pytest
from pydantic import ValidationError

from ia_agent_fwk.tools.config import ToolPermissionConfig, ToolsConfig


class TestToolsConfig:
    def test_defaults(self):
        cfg = ToolsConfig()
        assert cfg.default_timeout == 30.0
        assert cfg.default_permission_mode == "allow_all"
        assert cfg.builtin_tools_enabled is True
        assert cfg.max_retries == 3

    def test_custom_values(self):
        cfg = ToolsConfig(
            default_timeout=60.0,
            default_permission_mode="deny_list",
            builtin_tools_enabled=False,
            max_retries=5,
        )
        assert cfg.default_timeout == 60.0
        assert cfg.default_permission_mode == "deny_list"
        assert cfg.builtin_tools_enabled is False
        assert cfg.max_retries == 5

    def test_frozen(self):
        cfg = ToolsConfig()
        with pytest.raises(ValidationError):
            cfg.default_timeout = 99.0  # type: ignore[misc]


class TestToolPermissionConfig:
    def test_defaults(self):
        cfg = ToolPermissionConfig()
        assert cfg.mode == "allow_all"
        assert cfg.allowed == []
        assert cfg.denied == []
        assert cfg.require_confirmation == []

    def test_custom_values(self):
        cfg = ToolPermissionConfig(
            mode="allow_list",
            allowed=["tool_a", "tool_b"],
            denied=["tool_c"],
            require_confirmation=["tool_d"],
        )
        assert cfg.mode == "allow_list"
        assert cfg.allowed == ["tool_a", "tool_b"]
        assert cfg.denied == ["tool_c"]
        assert cfg.require_confirmation == ["tool_d"]

    def test_frozen(self):
        cfg = ToolPermissionConfig()
        with pytest.raises(ValidationError):
            cfg.mode = "deny_list"  # type: ignore[misc]
