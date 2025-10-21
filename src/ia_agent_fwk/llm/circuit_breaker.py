"""Per-provider circuit breaker with CLOSED / OPEN / HALF_OPEN states.

The circuit breaker prevents cascading failures by short-circuiting calls
to a provider that is consistently failing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ia_agent_fwk.config.settings import CircuitBreakerSettings

from ia_agent_fwk.llm.exceptions import CircuitOpenError
from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Async-safe circuit breaker for an LLM provider.

    Parameters
    ----------
    provider_name:
        Logical name used in log/error messages.
    settings:
        Configurable thresholds and timeouts.

    """

    def __init__(
        self,
        provider_name: str,
        settings: CircuitBreakerSettings,
    ) -> None:
        self.provider_name = provider_name
        self.settings = settings
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (read-only)."""
        return self._state

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute *fn* through the circuit breaker.

        If the circuit breaker is disabled (``settings.enabled is False``),
        *fn* is called directly without any state tracking.
        """
        if not self.settings.enabled:
            return await fn(*args, **kwargs)

        async with self._lock:
            self._maybe_transition_to_half_open()

            if self._state is CircuitState.OPEN:
                collector = get_metrics_collector()
                collector.increment(
                    "llm_circuit_breaker_rejected_total",
                    labels={"provider": self.provider_name},
                )
                raise CircuitOpenError(
                    self.provider_name,
                    self.settings.recovery_timeout,
                )

            # CLOSED or HALF_OPEN -- allow the call through.
            is_half_open = self._state is CircuitState.HALF_OPEN

        # Execute outside the lock so concurrent calls are not serialised.
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            async with self._lock:
                self._record_failure(is_half_open=is_half_open)
            raise

        # Success path.
        async with self._lock:
            self._record_success(is_half_open=is_half_open)

        return result

    async def reset(self) -> None:
        """Manually reset the circuit to CLOSED."""
        async with self._lock:
            self._transition(CircuitState.CLOSED)
            self._failure_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """If OPEN and the recovery timeout has elapsed, move to HALF_OPEN."""
        if self._state is not CircuitState.OPEN:
            return
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self.settings.recovery_timeout:
            self._transition(CircuitState.HALF_OPEN)

    def _record_failure(self, *, is_half_open: bool) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if is_half_open:
            # Trial request failed -- re-open.
            self._transition(CircuitState.OPEN)
        elif self._failure_count >= self.settings.failure_threshold:
            self._transition(CircuitState.OPEN)

    def _record_success(self, *, is_half_open: bool) -> None:
        if is_half_open:
            self._transition(CircuitState.CLOSED)
        self._failure_count = 0

    def _transition(self, new_state: CircuitState) -> None:
        if new_state != self._state:
            old_state = self._state
            collector = get_metrics_collector()
            collector.increment(
                "llm_circuit_breaker_transitions_total",
                labels={
                    "provider": self.provider_name,
                    "from_state": old_state.value,
                    "to_state": new_state.value,
                },
            )
            logger.warning(
                "Circuit breaker [%s] %s -> %s",
                self.provider_name,
                old_state,
                new_state,
                extra={
                    "llm_data": {
                        "event": "circuit_breaker_transition",
                        "provider": self.provider_name,
                        "from_state": old_state.value,
                        "to_state": new_state.value,
                        "failure_count": self._failure_count,
                    }
                },
            )
            self._state = new_state
