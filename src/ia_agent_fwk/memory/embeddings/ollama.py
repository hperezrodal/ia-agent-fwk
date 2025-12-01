"""Ollama embedding provider using the local embeddings API."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import httpx

from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.exceptions import MemoryStoreError
from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# Ollama embedding models and their dimensions.
_MODELS: dict[str, dict[str, int]] = {
    "nomic-embed-text": {"dimension": 768, "max_tokens": 8192},
    "mxbai-embed-large": {"dimension": 1024, "max_tokens": 512},
    "all-minilm": {"dimension": 384, "max_tokens": 256},
    "snowflake-arctic-embed": {"dimension": 1024, "max_tokens": 512},
}


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider using the local ``/api/embed`` endpoint.

    Parameters
    ----------
    base_url:
        Ollama server URL. Default: ``"http://localhost:11434"``.
    model:
        Embedding model name. Default: ``"nomic-embed-text"``.

    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._model_info = _MODELS.get(model, {"dimension": 768, "max_tokens": 8192})
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(120.0),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts via Ollama API.

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
            payload: dict[str, Any] = {
                "model": self._model,
                "input": texts,
            }
            resp = await self._client.post("/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
            duration_ms = (time.monotonic() - t0) * 1000
            collector.increment("rag_embedding_requests_total", labels={"provider": "ollama", "status": "success"})
            collector.observe("rag_embedding_duration_seconds", duration_ms / 1000, labels={"provider": "ollama"})
            collector.observe("rag_embedding_texts_count", len(texts), labels={"provider": "ollama"})
            return cast("list[list[float]]", data["embeddings"])
        except Exception as exc:
            collector.increment("rag_embedding_requests_total", labels={"provider": "ollama", "status": "failure"})
            msg = f"Ollama embedding generation failed: {exc}"
            raise MemoryStoreError(msg) from exc

    def dimension(self) -> int:
        """Return the embedding dimension for the configured model."""
        return self._model_info["dimension"]

    def max_tokens(self) -> int:
        """Return the maximum input tokens for the configured model."""
        return self._model_info["max_tokens"]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
