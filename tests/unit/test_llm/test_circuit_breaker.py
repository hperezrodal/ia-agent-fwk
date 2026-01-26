"""Tests for the circuit breaker state machine."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ia_agent_fwk.config.settings import CircuitBreakerSettings
from ia_agent_fwk.llm.circuit_breaker import CircuitBreaker, CircuitState
from ia_agent_fwk.llm.exceptions import CircuitOpenError


@pytest.fixture
def cb_settings() -> CircuitBreakerSettings:
    return CircuitBreakerSettings(enabled=True, failure_threshold=3, recovery_timeout=10.0)


@pytest.fixture
def cb(cb_settings) -> CircuitBreaker:
    return CircuitBreaker(provider_name="test", settings=cb_settings)


class TestCircuitBreaker:
    async def test_initial_state_is_closed(self, cb):
        assert cb.state == CircuitState.CLOSED

    async def test_success_keeps_closed(self, cb):
        fn = AsyncMock(return_value="ok")
        result = await cb.call(fn)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    async def test_failures_below_threshold_stay_closed(self, cb):
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fn)
        assert cb.state == CircuitState.CLOSED

    async def test_failures_at_threshold_opens_circuit(self, cb):
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fn)
        assert cb.state == CircuitState.OPEN

    async def test_open_circuit_raises_immediately(self, cb):
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fn)

        clean_fn = AsyncMock(return_value="ok")
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(clean_fn)
        assert exc_info.value.provider_name == "test"
        clean_fn.assert_not_awaited()

    @patch("ia_agent_fwk.llm.circuit_breaker.time.monotonic")
    async def test_recovery_timeout_transitions_to_half_open(self, mock_time, cb):
        mock_time.return_value = 100.0
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fn)
        assert cb.state == CircuitState.OPEN

        # Advance past recovery timeout.
        mock_time.return_value = 111.0
        ok_fn = AsyncMock(return_value="recovered")
        result = await cb.call(ok_fn)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    @patch("ia_agent_fwk.llm.circuit_breaker.time.monotonic")
    async def test_half_open_failure_reopens(self, mock_time, cb):
        mock_time.return_value = 100.0
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fn)

        mock_time.return_value = 111.0
        fail_fn = AsyncMock(side_effect=RuntimeError("still broken"))
        with pytest.raises(RuntimeError):
            await cb.call(fail_fn)
        assert cb.state == CircuitState.OPEN

    async def test_success_resets_failure_count(self, cb):
        fn_fail = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await cb.call(fn_fail)
        with pytest.raises(RuntimeError):
            await cb.call(fn_fail)

        fn_ok = AsyncMock(return_value="ok")
        await cb.call(fn_ok)
        assert cb._failure_count == 0

    async def test_disabled_circuit_breaker(self):
        settings = CircuitBreakerSettings(enabled=False, failure_threshold=1)
        cb = CircuitBreaker(provider_name="test", settings=settings)

        fn = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(10):
            with pytest.raises(RuntimeError):
                await cb.call(fn)
        # Should never open.
        assert cb.state == CircuitState.CLOSED

    async def test_reset(self, cb):
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fn)
        assert cb.state == CircuitState.OPEN

        await cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
