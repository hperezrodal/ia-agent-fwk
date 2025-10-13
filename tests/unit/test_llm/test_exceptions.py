"""Tests for the LLM exception hierarchy."""

from __future__ import annotations

import pytest

from ia_agent_fwk.llm.exceptions import (
    CircuitOpenError,
    LLMAuthenticationError,
    LLMConfigError,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamError,
    LLMTimeoutError,
)


class TestExceptionHierarchy:
    def test_base_is_exception(self):
        assert issubclass(LLMProviderError, Exception)

    @pytest.mark.parametrize(
        "cls",
        [
            LLMStreamError,
            CircuitOpenError,
            LLMConfigError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
        ],
    )
    def test_all_inherit_from_base(self, cls):
        assert issubclass(cls, LLMProviderError)

    def test_stream_error_partial_content(self):
        exc = LLMStreamError("broken", partial_content="partial")
        assert exc.partial_content == "partial"
        assert "broken" in str(exc)

    def test_stream_error_no_partial_content(self):
        exc = LLMStreamError("broken")
        assert exc.partial_content is None

    def test_circuit_open_error_attributes(self):
        exc = CircuitOpenError("openai", 30.0)
        assert exc.provider_name == "openai"
        assert exc.recovery_timeout == 30.0
        assert "openai" in str(exc)
        assert "30.0" in str(exc)

    def test_base_error_catches_all(self):
        msg = "auth failed"
        with pytest.raises(LLMProviderError):
            raise LLMAuthenticationError(msg)

    def test_config_error_message(self):
        exc = LLMConfigError("bad config")
        assert str(exc) == "bad config"
