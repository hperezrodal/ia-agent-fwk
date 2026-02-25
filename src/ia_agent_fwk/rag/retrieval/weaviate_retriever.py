"""Weaviate-based retriever using WeaviateMemoryBackend for vector search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.rag.exceptions import RetrievalError
from ia_agent_fwk.rag.models import Chunk, RetrievalResult
from ia_agent_fwk.rag.retrieval.base import Retriever

if TYPE_CHECKING:
    from ia_agent_fwk.memory.backends.weaviate_backend import WeaviateMemoryBackend
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class WeaviateRetriever(Retriever):
    """Retrieve chunks by embedding the query and searching a Weaviate backend.

    Parameters
    ----------
    backend:
        A ``WeaviateMemoryBackend`` instance for vector storage and retrieval.
    embedding_provider:
        An ``EmbeddingProvider`` to embed the query text.
    metadata_filter:
        Optional metadata filter applied to every retrieval call.

    """

    def __init__(
        self,
        backend: WeaviateMemoryBackend,
        embedding_provider: EmbeddingProvider,
        metadata_filter: dict[str, Any] | None = None,
    ) -> None:
        self._backend = backend
        self._embedding_provider = embedding_provider
        self._metadata_filter = metadata_filter

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Embed *query* and search the Weaviate backend for similar chunks."""
        try:
            # Merge constructor-level metadata_filter with per-call filters.
            # Per-call filters take precedence on key conflicts.
            merged_filter: dict[str, Any] = {}
            if self._metadata_filter:
                merged_filter.update(self._metadata_filter)
            if filters:
                merged_filter.update(filters)

            kwargs: dict[str, Any] = {}
            if merged_filter:
                kwargs["metadata_filter"] = merged_filter
            results = await self._backend.search(query, top_k=top_k, **kwargs)
        except Exception as exc:
            msg = f"Weaviate retrieval failed: {exc}"
            raise RetrievalError(msg) from exc

        retrieval_results: list[RetrievalResult] = []
        for result in results:
            chunk = Chunk(
                content=str(result.value) if result.value is not None else "",
                metadata=result.metadata or {},
                source=result.key,
            )
            retrieval_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=result.score,
                    metadata=result.metadata or {},
                )
            )

        return retrieval_results
