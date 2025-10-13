"""LLM provider exception hierarchy.

All LLM-related exceptions inherit from ``LLMProviderError`` which itself
inherits from the built-in ``Exception``.
"""

from __future__ import annotations


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""


class LLMStreamError(LLMProviderError):
    """Error during an LLM streaming response.

    Attributes
    ----------
    partial_content:
        Any content received before the error occurred.

    """

    def __init__(self, message: str, *, partial_content: str | None = None) -> None:
        super().__init__(message)
        self.partial_content = partial_content


class CircuitOpenError(LLMProviderError):
    """Raised when a circuit breaker is in the OPEN state.

    Attributes
    ----------
    provider_name:
        Name of the provider whose circuit is open.
    recovery_timeout:
        Seconds until the circuit transitions to HALF_OPEN.

    """

    def __init__(self, provider_name: str, recovery_timeout: float) -> None:
        super().__init__(
            f"Circuit breaker for provider '{provider_name}' is OPEN. Recovery in {recovery_timeout:.1f}s."
        )
        self.provider_name = provider_name
        self.recovery_timeout = recovery_timeout


class LLMConfigError(LLMProviderError):
    """Raised for LLM configuration errors (missing keys, unknown providers)."""


class LLMAuthenticationError(LLMProviderError):
    """Raised when authentication with an LLM provider fails (HTTP 401/403)."""


class LLMRateLimitError(LLMProviderError):
    """Raised when an LLM provider returns a rate-limit error (HTTP 429)."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM request times out."""
