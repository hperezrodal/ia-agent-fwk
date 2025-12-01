"""Abstract base class for embedding providers.

All concrete embedding providers extend ``EmbeddingProvider`` and implement
the ``embed``, ``dimension``, and ``max_tokens`` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Subclasses must implement ``embed``, ``dimension``, and ``max_tokens``.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors, one per input text.

        """
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider/model."""
        ...

    @abstractmethod
    def max_tokens(self) -> int:
        """Return the maximum input tokens per text for this provider/model."""
        ...

    async def close(self) -> None:  # noqa: B027
        """Release any resources held by the provider. Default: no-op."""
