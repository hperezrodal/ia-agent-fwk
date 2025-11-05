"""Tests for Tool ABC and ToolContext dataclass."""

import pytest
from pydantic import BaseModel

from ia_agent_fwk.tools.base import Tool, ToolContext

from .conftest import DummyInput, DummyOutput, DummyTool


class TestToolABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Tool()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        tool = DummyTool()
        assert tool.name == "dummy"
        assert tool.description == "A dummy tool named dummy"

    def test_input_schema_is_pydantic_model(self):
        tool = DummyTool()
        assert issubclass(tool.input_schema, BaseModel)

    def test_output_schema_is_pydantic_model(self):
        tool = DummyTool()
        assert issubclass(tool.output_schema, BaseModel)

    def test_tags_default_empty(self):
        tool = DummyTool()
        assert tool.tags == []

    def test_tags_with_values(self):
        tool = DummyTool(tool_tags=["tag1", "tag2"])
        assert tool.tags == ["tag1", "tag2"]

    async def test_execute_is_async(self):
        tool = DummyTool()
        context = ToolContext(execution_id="test-001")
        inp = DummyInput(value="hello")
        result = await tool.execute(inp, context)
        assert isinstance(result, DummyOutput)
        assert result.result == "echo:hello"

    def test_custom_name(self):
        tool = DummyTool(tool_name="custom")
        assert tool.name == "custom"


class TestToolContext:
    def test_creation_with_defaults(self):
        ctx = ToolContext(execution_id="test-001")
        assert ctx.execution_id == "test-001"
        assert ctx.agent_id == ""
        assert ctx.timeout == 30.0
        assert ctx.metadata == {}

    def test_creation_with_all_fields(self):
        ctx = ToolContext(
            execution_id="test-002",
            agent_id="agent-1",
            timeout=60.0,
            metadata={"key": "value"},
        )
        assert ctx.execution_id == "test-002"
        assert ctx.agent_id == "agent-1"
        assert ctx.timeout == 60.0
        assert ctx.metadata == {"key": "value"}

    def test_metadata_default_factory(self):
        ctx1 = ToolContext(execution_id="t1")
        ctx2 = ToolContext(execution_id="t2")
        # Ensure default_factory creates separate dicts
        ctx1.metadata["key"] = "val"
        assert "key" not in ctx2.metadata
