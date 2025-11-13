"""Tests for DefaultToolExecutor."""

import json

import pytest

from ia_agent_fwk.agents.protocols import ToolExecutor, ToolResult
from ia_agent_fwk.llm.models import ToolCall
from ia_agent_fwk.tools.config import ToolPermissionConfig
from ia_agent_fwk.tools.executor import DefaultToolExecutor
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
from ia_agent_fwk.tools.registry import ToolRegistry

from .conftest import (
    AdderTool,
    BadOutputTool,
    DummyTool,
    ErrorTool,
    SlowTool,
    ToolExecutionErrorTool,
)


@pytest.fixture
def executor_with_tools():
    registry = ToolRegistry()
    registry.register(DummyTool("dummy"))
    registry.register(AdderTool())
    registry.register(ErrorTool())
    registry.register(SlowTool(sleep_seconds=5.0))
    registry.register(ToolExecutionErrorTool())
    registry.register(BadOutputTool())
    pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
    return DefaultToolExecutor(
        registry=registry,
        permission_manager=pm,
        agent_id="test-agent",
        default_timeout=2.0,
    )


class TestProtocolCompliance:
    def test_isinstance_check(self, executor_with_tools):
        assert isinstance(executor_with_tools, ToolExecutor)


class TestFullPipelineSuccess:
    async def test_dummy_tool_success(self, executor_with_tools):
        tc = ToolCall(id="tc-1", name="dummy", arguments='{"value": "hello"}')
        result = await executor_with_tools.execute(tc)
        assert isinstance(result, ToolResult)
        assert result.tool_call_id == "tc-1"
        assert result.error is None
        data = json.loads(result.output)
        assert data["result"] == "echo:hello"

    async def test_adder_tool_success(self, executor_with_tools):
        tc = ToolCall(id="tc-2", name="adder", arguments='{"x": 3, "y": 7}')
        result = await executor_with_tools.execute(tc)
        assert result.error is None
        data = json.loads(result.output)
        assert data["sum"] == 10


class TestMissingTool:
    async def test_missing_tool_returns_error(self, executor_with_tools):
        tc = ToolCall(id="tc-3", name="nonexistent", arguments="{}")
        result = await executor_with_tools.execute(tc)
        assert result.error is not None
        assert "not found" in result.error
        assert result.tool_call_id == "tc-3"


class TestPermissionDenied:
    async def test_denied_tool_returns_error(self):
        registry = ToolRegistry()
        registry.register(DummyTool("dummy"))
        cfg = ToolPermissionConfig(mode="allow_list", allowed=["other_tool"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"test-agent": cfg},
        )
        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=pm,
            agent_id="test-agent",
        )
        tc = ToolCall(id="tc-4", name="dummy", arguments="{}")
        result = await executor.execute(tc)
        assert result.error is not None
        assert "not in the allow list" in result.error


class TestInputValidationFailure:
    async def test_wrong_argument_types(self, executor_with_tools):
        tc = ToolCall(id="tc-5", name="adder", arguments='{"x": "not_int", "y": 2}')
        result = await executor_with_tools.execute(tc)
        assert result.error is not None
        assert "Validation error" in result.error

    async def test_missing_required_field(self, executor_with_tools):
        tc = ToolCall(id="tc-6", name="adder", arguments='{"x": 1}')
        result = await executor_with_tools.execute(tc)
        assert result.error is not None
        assert "Validation error" in result.error

    async def test_invalid_json_arguments(self, executor_with_tools):
        tc = ToolCall(id="tc-7", name="dummy", arguments="not valid json")
        result = await executor_with_tools.execute(tc)
        assert result.error is not None
        assert "Validation error" in result.error


class TestOutputValidationFailure:
    async def test_wrong_output_type(self, executor_with_tools):
        tc = ToolCall(id="tc-8", name="bad_output_tool", arguments='{"value": "test"}')
        result = await executor_with_tools.execute(tc)
        assert result.error is not None
        assert "Validation error" in result.error


class TestTimeout:
    async def test_slow_tool_times_out(self, executor_with_tools):
        tc = ToolCall(id="tc-9", name="slow_tool", arguments='{"value": "test"}')
        result = await executor_with_tools.execute(tc)
        assert result.error is not None
        assert "timed out" in result.error

    async def test_timeout_returns_within_reasonable_time(self, executor_with_tools):
        import time

        tc = ToolCall(id="tc-10", name="slow_tool", arguments='{"value": "test"}')
        start = time.monotonic()
        result = await executor_with_tools.execute(tc)
        elapsed = time.monotonic() - start
        assert result.error is not None
        # Should return within ~3s (2s timeout + overhead)
        assert elapsed < 4.0


class TestExceptionWrapping:
    async def test_runtime_error_wrapped(self, executor_with_tools):
        tc = ToolCall(id="tc-11", name="error_tool", arguments='{"value": "test"}')
        result = await executor_with_tools.execute(tc)
        assert result.error is not None
        assert "execution failed" in result.error
        assert result.tool_call_id == "tc-11"

    async def test_tool_execution_error_wrapped(self, executor_with_tools):
        tc = ToolCall(id="tc-12", name="exec_error_tool", arguments='{"value": "test"}')
        result = await executor_with_tools.execute(tc)
        assert result.error is not None
        assert "custom execution error" in result.error


class TestLogging:
    async def test_success_logged(self, executor_with_tools, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            tc = ToolCall(id="tc-13", name="dummy", arguments='{"value": "test"}')
            await executor_with_tools.execute(tc)
        assert any("dummy" in r.message and "success" in r.message for r in caplog.records)

    async def test_error_logged(self, executor_with_tools, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            tc = ToolCall(id="tc-14", name="nonexistent", arguments="{}")
            await executor_with_tools.execute(tc)
        assert any("nonexistent" in r.message for r in caplog.records)
