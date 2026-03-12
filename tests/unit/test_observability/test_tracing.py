"""Tests for the observability tracing module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ia_agent_fwk.config.settings import TracingSettings
from ia_agent_fwk.observability.exceptions import TracingConfigError
from ia_agent_fwk.observability.tracing import TracingManager, get_tracer, traced


@pytest.mark.unit
class TestTracingManager:
    """Tests for TracingManager."""

    def test_setup_disabled(self, tracing_settings_disabled):
        """When tracing is disabled, setup is a no-op."""
        manager = TracingManager(tracing_settings_disabled)
        manager.setup()
        assert manager.provider is None

    def test_setup_console_exporter(self, tracing_settings_console):
        """Console exporter creates a TracerProvider."""
        manager = TracingManager(tracing_settings_console)
        manager.setup()
        assert manager.provider is not None
        manager.shutdown()

    def test_setup_invalid_exporter(self):
        """Invalid exporter raises TracingConfigError."""
        settings = TracingSettings(enabled=True, exporter="invalid_exporter")
        manager = TracingManager(settings)
        with pytest.raises(TracingConfigError, match="Unsupported tracing exporter"):
            manager.setup()

    def test_shutdown_without_setup(self, tracing_settings_disabled):
        """Shutdown without setup is a no-op."""
        manager = TracingManager(tracing_settings_disabled)
        manager.shutdown()  # Should not raise

    def test_shutdown_after_setup(self, tracing_settings_console):
        """Shutdown after setup flushes the provider."""
        manager = TracingManager(tracing_settings_console)
        manager.setup()
        manager.shutdown()
        # Provider still exists but is shut down
        assert manager.provider is not None

    def test_setup_otlp_missing_dependency(self):
        """OTLP exporter raises TracingConfigError when dependency is missing."""
        settings = TracingSettings(enabled=True, exporter="otlp")
        manager = TracingManager(settings)
        with (
            patch.dict("sys.modules", {"opentelemetry.exporter.otlp.proto.grpc.trace_exporter": None}),
            pytest.raises(TracingConfigError, match="opentelemetry-exporter-otlp"),
        ):
            manager.setup()


@pytest.mark.unit
class TestGetTracer:
    """Tests for get_tracer()."""

    def test_returns_tracer(self):
        """get_tracer returns a Tracer instance."""
        tracer = get_tracer("test.module")
        assert tracer is not None

    def test_different_names_return_tracers(self):
        """Different module names return (possibly different) tracers."""
        t1 = get_tracer("module.a")
        t2 = get_tracer("module.b")
        assert t1 is not None
        assert t2 is not None


@pytest.mark.unit
class TestTracedDecorator:
    """Tests for the @traced decorator."""

    async def test_traced_success(self):
        """@traced wraps a function and returns its result."""

        @traced("test_span")
        async def my_func(x: int) -> int:
            return x * 2

        result = await my_func(5)
        assert result == 10

    async def test_traced_default_name(self):
        """@traced without name uses module.qualname."""

        @traced()
        async def another_func() -> str:
            return "hello"

        result = await another_func()
        assert result == "hello"

    async def test_traced_with_attributes(self):
        """@traced with static attributes does not raise."""

        @traced("attr_span", attributes={"agent_type": "test", "tool_name": "calc"})
        async def func_with_attrs() -> int:
            return 42

        result = await func_with_attrs()
        assert result == 42

    async def test_traced_exception_propagation(self):
        """@traced re-raises exceptions from the wrapped function."""

        @traced("error_span")
        async def failing_func() -> None:
            msg = "test error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="test error"):
            await failing_func()

    async def test_traced_preserves_function_name(self):
        """@traced preserves the wrapped function's name."""

        @traced("span")
        async def original_name() -> None:
            pass

        assert original_name.__name__ == "original_name"
