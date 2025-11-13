"""Tests for ToolPermissionManager."""

import pytest

from ia_agent_fwk.tools.config import ToolPermissionConfig
from ia_agent_fwk.tools.exceptions import ToolPermissionError
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager


class TestPermissionModeEnum:
    def test_values(self):
        assert PermissionMode.allow_all.value == "allow_all"
        assert PermissionMode.allow_list.value == "allow_list"
        assert PermissionMode.deny_list.value == "deny_list"
        assert PermissionMode.require_confirmation.value == "require_confirmation"


class TestAllowAllMode:
    def test_permits_any_tool(self):
        pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        pm.check_permission("agent-1", "any_tool")  # Should not raise
        pm.check_permission("agent-2", "another_tool")

    def test_is_permitted_returns_true(self):
        pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        assert pm.is_permitted("agent-1", "any_tool") is True


class TestAllowListMode:
    def test_permits_listed_tools(self):
        cfg = ToolPermissionConfig(mode="allow_list", allowed=["calculator", "echo"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": cfg},
        )
        pm.check_permission("agent-1", "calculator")  # Should not raise
        pm.check_permission("agent-1", "echo")  # Should not raise

    def test_denies_unlisted_tools(self):
        cfg = ToolPermissionConfig(mode="allow_list", allowed=["calculator"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": cfg},
        )
        with pytest.raises(ToolPermissionError, match="not in the allow list"):
            pm.check_permission("agent-1", "http_request")

    def test_is_permitted_returns_false_for_unlisted(self):
        cfg = ToolPermissionConfig(mode="allow_list", allowed=["calculator"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": cfg},
        )
        assert pm.is_permitted("agent-1", "http_request") is False


class TestDenyListMode:
    def test_denies_listed_tools(self):
        cfg = ToolPermissionConfig(mode="deny_list", denied=["http_request"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": cfg},
        )
        with pytest.raises(ToolPermissionError, match="denied"):
            pm.check_permission("agent-1", "http_request")

    def test_permits_unlisted_tools(self):
        cfg = ToolPermissionConfig(mode="deny_list", denied=["http_request"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": cfg},
        )
        pm.check_permission("agent-1", "calculator")  # Should not raise

    def test_is_permitted(self):
        cfg = ToolPermissionConfig(mode="deny_list", denied=["http_request"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": cfg},
        )
        assert pm.is_permitted("agent-1", "http_request") is False
        assert pm.is_permitted("agent-1", "calculator") is True


class TestRequireConfirmationMode:
    def test_raises_for_listed_tools(self):
        cfg = ToolPermissionConfig(
            mode="require_confirmation",
            require_confirmation=["http_request"],
        )
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": cfg},
        )
        with pytest.raises(ToolPermissionError, match="requires human confirmation"):
            pm.check_permission("agent-1", "http_request")

    def test_permits_unlisted_tools(self):
        cfg = ToolPermissionConfig(
            mode="require_confirmation",
            require_confirmation=["http_request"],
        )
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": cfg},
        )
        pm.check_permission("agent-1", "calculator")  # Should not raise


class TestDefaultModeFallback:
    def test_falls_back_to_default_mode(self):
        pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        # Agent not in agent_permissions, should use default
        pm.check_permission("unknown-agent", "any_tool")

    def test_falls_back_to_deny_all_default(self):
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_list,
        )
        # Default mode is allow_list with empty allowed list
        with pytest.raises(ToolPermissionError):
            pm.check_permission("unknown-agent", "any_tool")
