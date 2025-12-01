"""OpenAI embedding provider using the embeddings API."""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

import openai

from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.exceptions import MemoryConfigError, MemoryStoreError
from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using the embeddings API.

    Parameters
    ----------
    api_key:
        OpenAI API key. If empty, reads from ``OPENAI_API_KEY`` env var.
    model:
        Embedding model name. Default: ``"text-embedding-3-small"``.

    """

    MODELS: ClassVar[dict[str, dict[str, int]]] = {
        "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8191},
        "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8191},
    }

    def __init__(self, api_key: str = "", model: str = "text-embedding-3-small") -> None:
        if model not in self.MODELS:
            msg = f"Unknown OpenAI embedding model: {model!r}. Supported: {', '.join(sorted(self.MODELS))}"
            raise MemoryConfigError(msg)

        self._model = model
        self._model_info = self.MODELS[model]

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client: openai.AsyncOpenAI = openai.AsyncOpenAI(**client_kwargs)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts via OpenAI API.

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors, one per input text.

        """
        if not texts:
            return []

        collector = get_metrics_collector()
        t0 = time.monotonic()
        try:
            response = await self._client.embeddings.create(
                input=texts,
                model=self._model,
            )
        except Exception as exc:
            collector.increment("rag_embedding_requests_total", labels={"provider": "openai", "status": "failure"})
            msg = f"OpenAI embedding generation failed: {exc}"
            raise MemoryStoreError(msg) from exc

        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("rag_embedding_requests_total", labels={"provider": "openai", "status": "success"})
        collector.observe("rag_embedding_duration_seconds", duration_ms / 1000, labels={"provider": "openai"})
        collector.observe("rag_embedding_texts_count", len(texts), labels={"provider": "openai"})

        # Sort by index to ensure correct ordering
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]

    def dimension(self) -> int:
        """Return the embedding dimension for the configured model."""
        return self._model_info["dimension"]

    def max_tokens(self) -> int:
        """Return the maximum input tokens for the configured model."""
        return self._model_info["max_tokens"]

    async def close(self) -> None:
        """Close the underlying OpenAI client."""
        await self._client.close()
