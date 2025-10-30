"""Tests for ToolExecutor protocol, ToolResult, and NoOpToolExecutor."""

from __future__ import annotations

import pytest

from ia_agent_fwk.agents.protocols import NoOpToolExecutor, ToolExecutor, ToolResult
from ia_agent_fwk.llm.models import ToolCall


class TestToolResult:
    def test_creation(self):
        result = ToolResult(output="data", tool_call_id="tc-1")
        assert result.output == "data"
        assert result.tool_call_id == "tc-1"
        assert result.error is None

    def test_with_error(self):
        result = ToolResult(output="", tool_call_id="tc-2", error="failed")
        assert result.error == "failed"


class TestToolExecutorProtocol:
    def test_noop_satisfies_protocol(self):
        executor = NoOpToolExecutor()
        assert isinstance(executor, ToolExecutor)


class TestNoOpToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self):
        executor = NoOpToolExecutor()
        tool_call = ToolCall(id="tc-1", name="search", arguments="{}")
        result = await executor.execute(tool_call)
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_execute_error_contains_tool_name(self):
        executor = NoOpToolExecutor()
        tool_call = ToolCall(id="tc-2", name="calculate", arguments='{"x": 1}')
        result = await executor.execute(tool_call)
        assert result.error is not None
        assert "calculate" in result.error
        assert "Epic 4" in result.error

    @pytest.mark.asyncio
    async def test_execute_preserves_tool_call_id(self):
        executor = NoOpToolExecutor()
        tool_call = ToolCall(id="unique-id-42", name="test", arguments="{}")
        result = await executor.execute(tool_call)
        assert result.tool_call_id == "unique-id-42"

    @pytest.mark.asyncio
    async def test_execute_output_is_empty(self):
        executor = NoOpToolExecutor()
        tool_call = ToolCall(id="tc-3", name="test", arguments="{}")
        result = await executor.execute(tool_call)
        assert result.output == ""
