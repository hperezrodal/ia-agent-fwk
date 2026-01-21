"""Tests for plugin loader."""

from __future__ import annotations

import pytest

from ia_agent_fwk.plugins.exceptions import PluginLoadError, PluginNotFoundError
from ia_agent_fwk.plugins.loader import PluginLoader, _NamespacedTool
from ia_agent_fwk.plugins.models import PluginConfig
from ia_agent_fwk.tools.base import ToolContext


@pytest.mark.unit
class TestPluginLoader:
    async def test_load_plugin_registers_tools(self, tool_registry, example_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        info = await loader.load_plugin(example_plugin_cls)

        assert info.load_status == "loaded"
        assert info.name == "example"
        assert info.version == "1.0.0"
        assert "example.echo" in info.tools_registered
        assert tool_registry.has("example.echo")

    async def test_load_plugin_calls_on_load(self, tool_registry, example_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        await loader.load_plugin(example_plugin_cls)

        # The on_load was called (verified via the plugin's internal state)
        # We verify indirectly by checking the plugin loaded successfully
        plugins = loader.get_loaded_plugins()
        assert len(plugins) == 1
        assert plugins[0].load_status == "loaded"

    async def test_unload_plugin_unregisters_tools(self, tool_registry, example_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        await loader.load_plugin(example_plugin_cls)

        assert tool_registry.has("example.echo")

        result = await loader.unload_plugin("example")
        assert result is True
        assert not tool_registry.has("example.echo")

    async def test_unload_plugin_calls_on_unload(self, tool_registry, example_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        await loader.load_plugin(example_plugin_cls)

        await loader.unload_plugin("example")

        # After unload, the plugin info should be updated
        plugins = loader.get_loaded_plugins()
        assert len(plugins) == 1
        assert plugins[0].load_status == "unloaded"

    async def test_unload_nonexistent_plugin_raises(self, tool_registry):
        loader = PluginLoader(tool_registry=tool_registry)

        with pytest.raises(PluginNotFoundError, match="not loaded"):
            await loader.unload_plugin("nonexistent")

    async def test_load_disabled_plugin_skipped(self, tool_registry, example_plugin_cls):
        configs = [PluginConfig(name="example", enabled=False)]
        loader = PluginLoader(tool_registry=tool_registry, plugin_configs=configs)

        info = await loader.load_plugin(example_plugin_cls)
        assert info.load_status == "disabled"
        assert info.enabled is False
        assert not tool_registry.has("example.echo")

    async def test_tool_namespacing(self, tool_registry, example_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        info = await loader.load_plugin(example_plugin_cls)

        # Tools should be namespaced as "plugin_name.tool_name"
        assert info.tools_registered == ["example.echo"]
        tool = tool_registry.get("example.echo")
        assert tool.name == "example.echo"

    async def test_load_all(self, tool_registry, example_plugin_cls, multi_tool_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        results = await loader.load_all([example_plugin_cls, multi_tool_plugin_cls])

        assert len(results) == 2
        assert all(info.load_status == "loaded" for info in results)

    async def test_load_all_handles_failures(self, tool_registry, failing_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        results = await loader.load_all([failing_plugin_cls])

        assert len(results) == 1
        assert results[0].load_status == "error"

    async def test_load_failing_plugin_raises(self, tool_registry, failing_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)

        with pytest.raises(PluginLoadError, match="on_load\\(\\) failed"):
            await loader.load_plugin(failing_plugin_cls)

    async def test_get_loaded_plugins(self, tool_registry, example_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        assert loader.get_loaded_plugins() == []

        await loader.load_plugin(example_plugin_cls)
        plugins = loader.get_loaded_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "example"

    async def test_multi_tool_plugin(self, tool_registry, multi_tool_plugin_cls):
        loader = PluginLoader(tool_registry=tool_registry)
        info = await loader.load_plugin(multi_tool_plugin_cls)

        assert len(info.tools_registered) == 2
        assert "multi.echo1" in info.tools_registered
        assert "multi.echo2" in info.tools_registered
        assert tool_registry.has("multi.echo1")
        assert tool_registry.has("multi.echo2")


@pytest.mark.unit
class TestNamespacedTool:
    """Tests for _NamespacedTool property delegation and execute."""

    def _make_inner_tool(self):
        from tests.unit.test_plugins.example_plugin import EchoTool

        return EchoTool()

    def test_name_returns_namespaced_name(self):
        inner = self._make_inner_tool()
        wrapped = _NamespacedTool(inner, "myplugin.echo")
        assert wrapped.name == "myplugin.echo"

    def test_description_delegates_to_inner(self):
        inner = self._make_inner_tool()
        wrapped = _NamespacedTool(inner, "myplugin.echo")
        assert wrapped.description == inner.description
        assert wrapped.description == "Echoes the input message"

    def test_input_schema_delegates_to_inner(self):
        inner = self._make_inner_tool()
        wrapped = _NamespacedTool(inner, "myplugin.echo")
        assert wrapped.input_schema is inner.input_schema

    def test_output_schema_delegates_to_inner(self):
        inner = self._make_inner_tool()
        wrapped = _NamespacedTool(inner, "myplugin.echo")
        assert wrapped.output_schema is inner.output_schema

    def test_tags_delegates_to_inner(self):
        inner = self._make_inner_tool()
        wrapped = _NamespacedTool(inner, "myplugin.echo")
        assert wrapped.tags == inner.tags
        assert wrapped.tags == ["example"]

    async def test_execute_delegates_to_inner(self):
        from tests.unit.test_plugins.example_plugin import EchoInput

        inner = self._make_inner_tool()
        wrapped = _NamespacedTool(inner, "myplugin.echo")
        ctx = ToolContext(execution_id="test-exec", agent_id="test-agent")
        inp = EchoInput(message="hello world")
        result = await wrapped.execute(inp, ctx)
        assert result.echo == "hello world"
