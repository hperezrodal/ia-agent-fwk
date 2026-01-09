"""Tests for the observability prompt_log module."""

from __future__ import annotations

import logging

import pytest

from ia_agent_fwk.config.settings import PromptLoggingSettings
from ia_agent_fwk.observability.prompt_log import PromptLogger


@pytest.mark.unit
class TestPromptLogger:
    """Tests for PromptLogger."""

    def test_enabled_by_default(self, prompt_logger):
        """PromptLogger is enabled by default."""
        assert prompt_logger.enabled is True

    def test_disabled_when_setting_false(self):
        """PromptLogger respects enabled=False."""
        settings = PromptLoggingSettings(enabled=False)
        logger = PromptLogger(settings)
        assert logger.enabled is False

    def test_log_prompt_emits_record(self, prompt_logger, caplog):
        """log_prompt emits a log record with prompt data."""
        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.prompts"):
            prompt_logger.log_prompt(
                provider="openai",
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                response="Hi there!",
                duration_ms=150.0,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "openai" in record.message
        assert "gpt-4o" in record.message
        assert hasattr(record, "prompt_data")
        assert record.prompt_data["provider"] == "openai"
        assert record.prompt_data["model"] == "gpt-4o"

    def test_log_prompt_disabled_no_output(self, caplog):
        """log_prompt is silent when disabled."""
        settings = PromptLoggingSettings(enabled=False)
        logger = PromptLogger(settings)

        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.prompts"):
            logger.log_prompt(
                provider="openai",
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                response="Hi",
                duration_ms=100.0,
            )

        assert len(caplog.records) == 0

    def test_log_prompt_without_inputs(self, caplog):
        """log_prompt respects log_inputs=False."""
        settings = PromptLoggingSettings(log_inputs=False)
        logger = PromptLogger(settings)

        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.prompts"):
            logger.log_prompt(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Secret"}],
                response="Answer",
                duration_ms=200.0,
            )

        assert len(caplog.records) == 1
        assert "messages" not in caplog.records[0].prompt_data

    def test_log_prompt_without_outputs(self, caplog):
        """log_prompt respects log_outputs=False."""
        settings = PromptLoggingSettings(log_outputs=False)
        logger = PromptLogger(settings)

        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.prompts"):
            logger.log_prompt(
                provider="openai",
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                response="Hello",
                duration_ms=100.0,
            )

        assert len(caplog.records) == 1
        assert "response" not in caplog.records[0].prompt_data

    def test_log_prompt_without_latency(self, caplog):
        """log_prompt respects log_latency=False."""
        settings = PromptLoggingSettings(log_latency=False)
        logger = PromptLogger(settings)

        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.prompts"):
            logger.log_prompt(
                provider="openai",
                model="gpt-4o",
                messages=[],
                response="",
                duration_ms=100.0,
            )

        assert len(caplog.records) == 1
        assert "duration_ms" not in caplog.records[0].prompt_data

    def test_log_prompt_without_tokens(self, caplog):
        """log_prompt respects log_tokens=False."""
        settings = PromptLoggingSettings(log_tokens=False)
        logger = PromptLogger(settings)

        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.prompts"):
            logger.log_prompt(
                provider="openai",
                model="gpt-4o",
                messages=[],
                response="",
                duration_ms=100.0,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        assert len(caplog.records) == 1
        assert "usage" not in caplog.records[0].prompt_data

    def test_log_prompt_with_no_usage(self, prompt_logger, caplog):
        """log_prompt works when usage is None."""
        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.prompts"):
            prompt_logger.log_prompt(
                provider="ollama",
                model="llama3",
                messages=[{"role": "user", "content": "test"}],
                response="response",
                duration_ms=50.0,
                usage=None,
            )

        assert len(caplog.records) == 1
        assert "usage" not in caplog.records[0].prompt_data
