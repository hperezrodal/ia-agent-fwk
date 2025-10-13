"""Abstract base class for LLM providers.

All concrete providers (OpenAI, Anthropic, Ollama, ...) extend
``LLMProvider`` and implement every abstract method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ia_agent_fwk.config.settings import LLMProviderSettings
    from ia_agent_fwk.llm.models import (
        ChatResponse,
        CompletionResponse,
        HealthStatus,
        Message,
        StreamChunk,
    )

_EMBED_NOT_IMPLEMENTED_MSG = (
    "LLMProvider.embed() is not implemented. Use EmbeddingProvider for embedding tasks (see Epic 7)."
)


class LLMProvider(ABC):
    """Abstract LLM provider interface.

    Parameters
    ----------
    settings:
        Provider-specific configuration.
    provider_name:
        Logical name of this provider (e.g. ``"openai"``).

    """

    def __init__(self, settings: LLMProviderSettings, provider_name: str) -> None:
        self.settings = settings
        self.provider_name = provider_name

    # ------------------------------------------------------------------
    # Abstract methods (must be overridden)
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Generate a text completion for *prompt*."""
        ...

    @abstractmethod
    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        """Generate a chat completion for *messages*."""
        ...

    @abstractmethod
    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion, yielding ``StreamChunk`` objects."""
        ...
        # Unreachable, but required to type the method as an async generator:
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count the number of tokens in *text*."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check provider connectivity and return a ``HealthStatus``."""
        ...

    # ------------------------------------------------------------------
    # Concrete methods (may be overridden)
    # ------------------------------------------------------------------

    def format_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool schemas to the provider's native format.

        The default implementation returns OpenAI-format schemas as-is.
        Providers that use a different format (e.g. Anthropic) should
        override this method.

        Parameters
        ----------
        schemas:
            Tool schemas in OpenAI function-calling format.

        Returns
        -------
        list[dict[str, Any]]
            Tool schemas in the provider's native format.

        """
        return schemas

    async def close(self) -> None:  # noqa: B027
        """Release resources held by this provider (default: no-op)."""

    async def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embeddings.

        .. note::

           Embedding is handled by the separate ``EmbeddingProvider`` ABC
           (Epic 7).  This stub exists to satisfy interface completeness.
        """
        raise NotImplementedError(_EMBED_NOT_IMPLEMENTED_MSG)
