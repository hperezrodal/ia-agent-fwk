"""Tests for the retry decorator / wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ia_agent_fwk.config.settings import RetrySettings
from ia_agent_fwk.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ia_agent_fwk.llm.retry import is_retryable, with_retry


class TestIsRetryable:
    def test_connection_error(self):
        assert is_retryable(ConnectionError("fail")) is True

    def test_timeout_error(self):
        assert is_retryable(TimeoutError("timeout")) is True

    def test_llm_timeout(self):
        assert is_retryable(LLMTimeoutError("t")) is True

    def test_llm_rate_limit(self):
        assert is_retryable(LLMRateLimitError("r")) is True

    def test_auth_error_not_retryable(self):
        assert is_retryable(LLMAuthenticationError("auth")) is False

    def test_status_code_429(self):
        exc = Exception("rate limited")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert is_retryable(exc) is True

    def test_status_code_500(self):
        exc = Exception("server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert is_retryable(exc) is True

    def test_status_code_400(self):
        exc = Exception("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert is_retryable(exc) is False

    def test_status_code_401(self):
        exc = Exception("unauthorized")
        exc.status_code = 401  # type: ignore[attr-defined]
        assert is_retryable(exc) is False

    def test_os_error(self):
        assert is_retryable(OSError("conn refused")) is True


class TestWithRetry:
    @pytest.fixture
    def fast_retry(self):
        return RetrySettings(max_attempts=3, backoff_base=0.01, backoff_max=0.01)

    async def test_success_first_try(self, fast_retry):
        fn = AsyncMock(return_value="ok")
        result = await with_retry(fn, retry_settings=fast_retry)
        assert result == "ok"
        assert fn.await_count == 1

    @patch("ia_agent_fwk.llm.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_success_after_retries(self, mock_sleep, fast_retry):
        fn = AsyncMock(side_effect=[ConnectionError("fail"), ConnectionError("fail"), "ok"])
        result = await with_retry(fn, retry_settings=fast_retry)
        assert result == "ok"
        assert fn.await_count == 3
        assert mock_sleep.await_count == 2

    @patch("ia_agent_fwk.llm.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_max_attempts_exhausted(self, mock_sleep, fast_retry):
        fn = AsyncMock(side_effect=ConnectionError("fail"))
        with pytest.raises(LLMProviderError, match="fail"):
            await with_retry(fn, retry_settings=fast_retry)
        assert fn.await_count == 3

    async def test_non_retryable_raises_immediately(self, fast_retry):
        fn = AsyncMock(side_effect=LLMAuthenticationError("auth"))
        with pytest.raises(LLMAuthenticationError):
            await with_retry(fn, retry_settings=fast_retry)
        assert fn.await_count == 1

    @patch("ia_agent_fwk.llm.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_after_header(self, mock_sleep, fast_retry):
        class RateLimitExc(Exception):
            status_code = 429

            class response:
                headers = {"retry-after": "1.5"}

        fn = AsyncMock(side_effect=[RateLimitExc(), "ok"])
        result = await with_retry(fn, retry_settings=fast_retry)
        assert result == "ok"
        # The sleep should have been called with the retry-after value (capped).
        mock_sleep.assert_called_once()
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay == pytest.approx(0.01, abs=0.01)  # capped by backoff_max

    @patch("ia_agent_fwk.llm.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_wraps_non_llm_exception(self, mock_sleep, fast_retry):
        fn = AsyncMock(side_effect=ConnectionError("conn"))
        with pytest.raises(LLMProviderError) as exc_info:
            await with_retry(fn, retry_settings=fast_retry)
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    async def test_single_attempt_setting(self):
        settings = RetrySettings(max_attempts=1)
        fn = AsyncMock(side_effect=ConnectionError("fail"))
        with pytest.raises(LLMProviderError):
            await with_retry(fn, retry_settings=settings)
        assert fn.await_count == 1
