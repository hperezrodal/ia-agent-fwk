"""Integration tests for the tool system."""

import asyncio
import json
import time

from ia_agent_fwk.agents.protocols import ToolExecutor, ToolResult
from ia_agent_fwk.llm.models import ToolCall
from ia_agent_fwk.tools.builtin import register_builtin_tools
from ia_agent_fwk.tools.builtin.calculator import CalculatorTool
from ia_agent_fwk.tools.config import ToolPermissionConfig, ToolsConfig
from ia_agent_fwk.tools.executor import DefaultToolExecutor
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
from ia_agent_fwk.tools.registry import ToolRegistry

from .conftest import DummyTool


class TestEndToEndExecutorPipeline:
    """Test the full path: ToolCall -> DefaultToolExecutor -> ToolRegistry -> Tool -> ToolResult."""

    async def test_calculator_end_to_end(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=pm,
            agent_id="test-agent",
        )

        tc = ToolCall(id="tc1", name="calculator", arguments='{"expression": "2 + 2"}')
        result = await executor.execute(tc)

        assert isinstance(result, ToolResult)
        assert result.error is None
        data = json.loads(result.output)
        assert data["result"] == 4.0
        assert result.tool_call_id == "tc1"

    async def test_complex_calculator_expression(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=pm,
        )

        tc = ToolCall(
            id="tc2",
            name="calculator",
            arguments='{"expression": "2 ** 10 + 3 * (4 - 1)"}',
        )
        result = await executor.execute(tc)

        assert result.error is None
        data = json.loads(result.output)
        assert data["result"] == 1033.0


class TestExecutorWithPermissions:
    async def test_permission_denied_returns_error(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        cfg = ToolPermissionConfig(mode="allow_list", allowed=["echo"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"test-agent": cfg},
        )
        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=pm,
            agent_id="test-agent",
        )

        tc = ToolCall(id="tc3", name="calculator", arguments='{"expression": "1+1"}')
        result = await executor.execute(tc)

        assert result.error is not None
        assert "not in the allow list" in result.error

    async def test_deny_list_permission(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(DummyTool("dummy"))

        cfg = ToolPermissionConfig(mode="deny_list", denied=["calculator"])
        pm = ToolPermissionManager(
            default_mode=PermissionMode.allow_all,
            agent_permissions={"test-agent": cfg},
        )
        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=pm,
            agent_id="test-agent",
        )

        # Calculator should be denied
        tc1 = ToolCall(id="tc4", name="calculator", arguments='{"expression": "1+1"}')
        result1 = await executor.execute(tc1)
        assert result1.error is not None

        # Dummy should be allowed
        tc2 = ToolCall(id="tc5", name="dummy", arguments='{"value": "test"}')
        result2 = await executor.execute(tc2)
        assert result2.error is None


class TestBuiltinToolRegistration:
    def test_register_builtin_tools(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)

        # Should have 24 built-in tools (7 core + 4 support + 4 document + 2 doc loader + 4 finance + 1 rag + 2 calendar)
        tools = registry.list()
        assert len(tools) == 24

        expected_names = {
            # Core tools
            "calculator",
            "file_reader",
            "http_request",
            "web_scraper",
            "database_query",
            "current_time",
            "echo",
            # Customer support tools
            "ticket_lookup",
            "faq_search",
            "escalation",
            "response_draft",
            # Document processor tools
            "text_extractor",
            "section_identifier",
            "entity_extractor",
            "summarizer",
            # Document loader tools
            "list_documents",
            "load_document",
            # Finance tools
            "financial_data_lookup",
            "ratio_calculator",
            "anomaly_detector",
            "report_generator",
            # RAG tools
            "rag_search",
            # Calendar tools
            "email_parser",
            "event_validator",
        }
        actual_names = {t.name for t in tools}
        assert actual_names == expected_names

    def test_openai_schemas_for_builtins(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)
        schemas = registry.openai_schemas()

        assert len(schemas) == 24
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func


class TestProtocolCompliance:
    def test_executor_satisfies_protocol(self):
        registry = ToolRegistry()
        pm = ToolPermissionManager()
        executor = DefaultToolExecutor(registry=registry, permission_manager=pm)
        assert isinstance(executor, ToolExecutor)


class TestConcurrency:
    async def test_concurrent_executions(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=pm,
        )

        async def run_calculation(i):
            tc = ToolCall(
                id=f"tc-{i}",
                name="calculator",
                arguments=json.dumps({"expression": f"{i} + {i}"}),
            )
            return await executor.execute(tc)

        # Run 10 concurrent calculations
        results = await asyncio.gather(*[run_calculation(i) for i in range(10)])

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result.error is None
            data = json.loads(result.output)
            assert data["result"] == float(i + i)


class TestPerformance:
    def test_registry_registration_performance(self):
        registry = ToolRegistry()
        start = time.monotonic()
        for i in range(100):
            registry.register(DummyTool(f"tool_{i}"))
        elapsed = time.monotonic() - start
        # NFR-001: < 10ms per tool, so < 1s for 100 tools
        assert elapsed < 1.0

    def test_registry_lookup_performance(self):
        registry = ToolRegistry()
        for i in range(50):
            registry.register(DummyTool(f"tool_{i}"))

        start = time.monotonic()
        for _ in range(1000):
            registry.get("tool_25")
        elapsed_per_lookup = (time.monotonic() - start) / 1000
        # NFR-002: < 1ms per lookup
        assert elapsed_per_lookup < 0.001

    async def test_executor_pipeline_overhead(self):
        registry = ToolRegistry()
        registry.register(DummyTool("dummy"))
        pm = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=pm,
        )

        tc = ToolCall(id="perf-1", name="dummy", arguments='{"value": "test"}')

        # Warm up
        await executor.execute(tc)

        # Measure
        times = []
        for _ in range(10):
            start = time.monotonic()
            await executor.execute(tc)
            times.append((time.monotonic() - start) * 1000)

        avg_ms = sum(times) / len(times)
        # NFR-004: < 10ms overhead
        assert avg_ms < 10.0


class TestConfigIntegration:
    def test_tools_config_defaults(self):
        cfg = ToolsConfig()
        assert cfg.default_timeout == 30.0
        assert cfg.default_permission_mode == "allow_all"
        assert cfg.builtin_tools_enabled is True

    def test_tools_config_with_custom_values(self):
        cfg = ToolsConfig(
            default_timeout=60.0,
            default_permission_mode="deny_list",
            builtin_tools_enabled=False,
        )
        assert cfg.default_timeout == 60.0
        assert cfg.default_permission_mode == "deny_list"
        assert cfg.builtin_tools_enabled is False
