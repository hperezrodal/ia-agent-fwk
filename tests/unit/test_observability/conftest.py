"""Shared fixtures for observability unit tests."""

from __future__ import annotations

import pytest

from ia_agent_fwk.config.settings import (
    LoggingSettings,
    MetricsSettings,
    ObservabilitySettings,
    PromptLoggingSettings,
    TracingSettings,
)
from ia_agent_fwk.observability.metrics import MetricsCollector
from ia_agent_fwk.observability.prompt_log import PromptLogger


@pytest.fixture
def observability_settings() -> ObservabilitySettings:
    """Default test observability settings."""
    return ObservabilitySettings()


@pytest.fixture
def logging_settings() -> LoggingSettings:
    """Default test logging settings."""
    return LoggingSettings()


@pytest.fixture
def tracing_settings_disabled() -> TracingSettings:
    """Tracing settings with tracing disabled."""
    return TracingSettings(enabled=False)


@pytest.fixture
def tracing_settings_console() -> TracingSettings:
    """Tracing settings with console exporter."""
    return TracingSettings(enabled=True, exporter="console", service_name="test-service")


@pytest.fixture
def metrics_settings() -> MetricsSettings:
    """Default test metrics settings."""
    return MetricsSettings()


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Fresh MetricsCollector instance for testing."""
    return MetricsCollector()


@pytest.fixture
def prompt_logging_settings() -> PromptLoggingSettings:
    """Default test prompt logging settings."""
    return PromptLoggingSettings()


@pytest.fixture
def prompt_logger(prompt_logging_settings) -> PromptLogger:
    """PromptLogger with default settings."""
    return PromptLogger(prompt_logging_settings)
