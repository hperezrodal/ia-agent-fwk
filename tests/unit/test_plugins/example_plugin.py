"""Example plugin for testing the plugin system.

Provides an ``ExamplePlugin`` with a single ``EchoTool`` that
returns its input as output.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ia_agent_fwk.plugins.base import Plugin
from ia_agent_fwk.plugins.models import PluginManifest
from ia_agent_fwk.tools.base import Tool, ToolContext


class EchoInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    message: str = "hello"


class EchoOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    echo: str


class EchoTool(Tool):
    """Simple tool that echoes its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the input message"

    @property
    def input_schema(self) -> type[BaseModel]:
        return EchoInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return EchoOutput

    @property
    def tags(self) -> list[str]:
        return ["example"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        assert isinstance(validated_input, EchoInput)
        return EchoOutput(echo=validated_input.message)


class ExamplePlugin(Plugin):
    """Example plugin providing an EchoTool."""

    def __init__(self) -> None:
        self._loaded = False
        self._unloaded = False

    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            name="example",
            version="1.0.0",
            description="An example plugin for testing",
            author="Test Author",
            tools=["echo"],
            entry_point="tests.unit.test_plugins.example_plugin:ExamplePlugin",
        )

    def get_tools(self) -> list[Tool]:
        return [EchoTool()]

    async def on_load(self) -> None:
        self._loaded = True

    async def on_unload(self) -> None:
        self._unloaded = True


class FailingPlugin(Plugin):
    """Plugin that fails on load for testing error handling."""

    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            name="failing",
            version="0.0.1",
            description="A plugin that fails on load",
        )

    def get_tools(self) -> list[Tool]:
        return []

    async def on_load(self) -> None:
        msg = "Intentional load failure"
        raise RuntimeError(msg)


class MultiToolPlugin(Plugin):
    """Plugin that provides multiple tools for testing."""

    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            name="multi",
            version="2.0.0",
            description="A plugin with multiple tools",
            tools=["echo1", "echo2"],
        )

    def get_tools(self) -> list[Tool]:
        class Echo1(EchoTool):
            @property
            def name(self) -> str:
                return "echo1"

        class Echo2(EchoTool):
            @property
            def name(self) -> str:
                return "echo2"

        return [Echo1(), Echo2()]
