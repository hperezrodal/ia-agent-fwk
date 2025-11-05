"""Tests for ToolRegistry."""

import pytest

from ia_agent_fwk.tools.config import ToolPermissionConfig
from ia_agent_fwk.tools.exceptions import ToolNotFoundError, ToolValidationError
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
from ia_agent_fwk.tools.registry import ToolRegistry

from .conftest import DummyTool


class TestToolRegistryRegister:
    def test_register_tool(self):
        registry = ToolRegistry()
        tool = DummyTool("test_tool")
        registry.register(tool)
        assert registry.has("test_tool")

    def test_duplicate_name_raises(self):
        registry = ToolRegistry()
        registry.register(DummyTool("tool_a"))
        with pytest.raises(ToolValidationError, match="already registered"):
            registry.register(DummyTool("tool_a"))

    def test_duplicate_name_with_replace(self):
        registry = ToolRegistry()
        tool1 = DummyTool("tool_a")
        tool2 = DummyTool("tool_a")
        registry.register(tool1)
        registry.register(tool2, replace=True)
        assert registry.get("tool_a") is tool2


class TestToolRegistryGet:
    def test_get_existing_tool(self):
        registry = ToolRegistry()
        tool = DummyTool("my_tool")
        registry.register(tool)
        assert registry.get("my_tool") is tool

    def test_get_missing_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError, match="not found"):
            registry.get("nonexistent")


class TestToolRegistryList:
    def test_list_all(self, sample_registry):
        tools = sample_registry.list()
        assert len(tools) == 3

    def test_list_with_matching_tag(self, sample_registry):
        tools = sample_registry.list(tags=["tag1"])
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_c"}

    def test_list_with_no_matching_tag(self, sample_registry):
        tools = sample_registry.list(tags=["nonexistent"])
        assert len(tools) == 0

    def test_list_with_multiple_tags(self, sample_registry):
        tools = sample_registry.list(tags=["tag2", "tag3"])
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}

    def test_list_empty_registry(self):
        registry = ToolRegistry()
        assert registry.list() == []


class TestToolRegistryHas:
    def test_has_existing(self):
        registry = ToolRegistry()
        registry.register(DummyTool("test"))
        assert registry.has("test") is True

    def test_has_missing(self):
        registry = ToolRegistry()
        assert registry.has("nonexistent") is False


class TestToolRegistryRemove:
    def test_remove_existing(self):
        registry = ToolRegistry()
        registry.register(DummyTool("tool_a"))
        registry.remove("tool_a")
        assert not registry.has("tool_a")

    def test_remove_missing_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError, match="not found"):
            registry.remove("nonexistent")


class TestToolRegistryOpenaiSchemas:
    def test_schema_format(self, sample_registry):
        schemas = sample_registry.openai_schemas()
        assert len(schemas) == 3
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_schema_contains_correct_names(self, sample_registry):
        schemas = sample_registry.openai_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert names == {"tool_a", "tool_b", "tool_c"}

    def test_parameters_is_json_schema(self, sample_registry):
        schemas = sample_registry.openai_schemas()
        for schema in schemas:
            params = schema["function"]["parameters"]
            assert "properties" in params
            assert "type" in params

    def test_filtered_by_permissions(self, sample_registry):
        perm_cfg = ToolPermissionConfig(mode="allow_list", allowed=["tool_a"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"agent-1": perm_cfg},
        )
        schemas = sample_registry.openai_schemas(agent_id="agent-1", permission_manager=pm)
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "tool_a"

    def test_no_filter_when_no_agent_id(self, sample_registry):
        pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        schemas = sample_registry.openai_schemas(permission_manager=pm)
        assert len(schemas) == 3
