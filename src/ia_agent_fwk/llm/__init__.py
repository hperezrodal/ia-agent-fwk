"""LLM Provider Layer -- public API.

All public types are re-exported from this package so that consumers can
write ``from ia_agent_fwk.llm import LLMProvider, Message, ...``.
"""

from ia_agent_fwk.llm.base import LLMProvider
from ia_agent_fwk.llm.circuit_breaker import CircuitBreaker, CircuitState
from ia_agent_fwk.llm.cost import CostEstimator
from ia_agent_fwk.llm.exceptions import (
    CircuitOpenError,
    LLMAuthenticationError,
    LLMConfigError,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamError,
    LLMTimeoutError,
)
from ia_agent_fwk.llm.factory import LLMProviderFactory
from ia_agent_fwk.llm.models import (
    ChatResponse,
    CompletionResponse,
    FinishReason,
    HealthStatus,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)

__all__ = [
    "ChatResponse",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "CompletionResponse",
    "CostEstimator",
    "FinishReason",
    "HealthStatus",
    "LLMAuthenticationError",
    "LLMConfigError",
    "LLMProvider",
    "LLMProviderError",
    "LLMProviderFactory",
    "LLMRateLimitError",
    "LLMStreamError",
    "LLMTimeoutError",
    "Message",
    "StreamChunk",
    "TokenUsage",
    "ToolCall",
]
