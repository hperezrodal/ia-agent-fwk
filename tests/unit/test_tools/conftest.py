"""Shared fixtures for tool system tests."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from ia_agent_fwk.tools.base import Tool, ToolContext
from ia_agent_fwk.tools.exceptions import ToolExecutionError
from ia_agent_fwk.tools.executor import DefaultToolExecutor
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
from ia_agent_fwk.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Input/Output models for test tools
# ---------------------------------------------------------------------------


class DummyInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: str = "default"


class DummyOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    result: str


class NumericInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int
    y: int


class NumericOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    sum: int


# ---------------------------------------------------------------------------
# Test tool implementations
# ---------------------------------------------------------------------------


class DummyTool(Tool):
    """Minimal tool that echoes its input."""

    def __init__(self, tool_name: str = "dummy", tool_tags: list[str] | None = None):
        self._name = tool_name
        self._tags = tool_tags or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"A dummy tool named {self._name}"

    @property
    def input_schema(self) -> type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return DummyOutput

    @property
    def tags(self) -> list[str]:
        return self._tags

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        assert isinstance(validated_input, DummyInput)
        return DummyOutput(result=f"echo:{validated_input.value}")


class ErrorTool(Tool):
    """Tool that always raises an error."""

    @property
    def name(self) -> str:
        return "error_tool"

    @property
    def description(self) -> str:
        return "A tool that always fails"

    @property
    def input_schema(self) -> type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return DummyOutput

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        msg = "Intentional error for testing"
        raise RuntimeError(msg)


class SlowTool(Tool):
    """Tool that sleeps for a configurable duration."""

    def __init__(self, sleep_seconds: float = 5.0):
        self._sleep = sleep_seconds

    @property
    def name(self) -> str:
        return "slow_tool"

    @property
    def description(self) -> str:
        return "A tool that takes a long time"

    @property
    def input_schema(self) -> type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return DummyOutput

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        await asyncio.sleep(self._sleep)
        return DummyOutput(result="slow_done")


class ToolExecutionErrorTool(Tool):
    """Tool that raises ToolExecutionError."""

    @property
    def name(self) -> str:
        return "exec_error_tool"

    @property
    def description(self) -> str:
        return "A tool that raises ToolExecutionError"

    @property
    def input_schema(self) -> type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return DummyOutput

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        msg = "custom execution error"
        raise ToolExecutionError(msg, tool_name="exec_error_tool")


class BadOutputTool(Tool):
    """Tool that returns wrong output type."""

    @property
    def name(self) -> str:
        return "bad_output_tool"

    @property
    def description(self) -> str:
        return "A tool that returns the wrong output type"

    @property
    def input_schema(self) -> type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return NumericOutput

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        # Returns DummyOutput when NumericOutput is expected
        return DummyOutput(result="wrong_type")


class AdderTool(Tool):
    """Tool that adds two numbers."""

    @property
    def name(self) -> str:
        return "adder"

    @property
    def description(self) -> str:
        return "Add two numbers"

    @property
    def input_schema(self) -> type[BaseModel]:
        return NumericInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return NumericOutput

    @property
    def tags(self) -> list[str]:
        return ["math"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        assert isinstance(validated_input, NumericInput)
        return NumericOutput(sum=validated_input.x + validated_input.y)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_tool():
    return DummyTool()


@pytest.fixture
def error_tool():
    return ErrorTool()


@pytest.fixture
def slow_tool():
    return SlowTool(sleep_seconds=5.0)


@pytest.fixture
def sample_tool_context():
    return ToolContext(execution_id="test-exec-001", agent_id="test-agent", timeout=30.0)


@pytest.fixture
def sample_registry():
    registry = ToolRegistry()
    registry.register(DummyTool("tool_a", ["tag1", "tag2"]))
    registry.register(DummyTool("tool_b", ["tag2", "tag3"]))
    registry.register(DummyTool("tool_c", ["tag1"]))
    return registry


@pytest.fixture
def sample_permission_manager():
    return ToolPermissionManager(default_mode=PermissionMode.allow_all)


@pytest.fixture
def sample_executor(sample_registry, sample_permission_manager):
    return DefaultToolExecutor(
        registry=sample_registry,
        permission_manager=sample_permission_manager,
        agent_id="test-agent",
        default_timeout=5.0,
    )
