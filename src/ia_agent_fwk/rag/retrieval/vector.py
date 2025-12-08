"""Vector-based retriever using a MemoryBackend and EmbeddingProvider."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.rag.exceptions import RetrievalError
from ia_agent_fwk.rag.models import Chunk, RetrievalResult
from ia_agent_fwk.rag.retrieval.base import Retriever

if TYPE_CHECKING:
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    """Retrieve chunks by embedding the query and searching a vector backend.

    Parameters
    ----------
    backend:
        A ``MemoryBackend`` instance (typically a vector store like pgvector
        or Qdrant).
    embedding_provider:
        An ``EmbeddingProvider`` to embed the query text.

    """

    def __init__(
        self,
        backend: MemoryBackend,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._backend = backend
        self._embedding_provider = embedding_provider

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Embed *query* and search the vector backend for similar chunks."""
        collector = get_metrics_collector()
        t0 = time.monotonic()
        try:
            kwargs: dict[str, Any] = {}
            if filters and hasattr(self._backend, "search"):
                # Pass metadata_filter if the backend's search method supports it
                kwargs["metadata_filter"] = filters
            results = await self._backend.search(query, top_k=top_k, **kwargs)
        except TypeError:
            # Backend doesn't accept metadata_filter; fall back without it
            results = await self._backend.search(query, top_k=top_k)
        except Exception as exc:
            collector.increment("rag_retrieval_total", labels={"strategy": "vector", "status": "failure"})
            msg = f"Vector retrieval failed: {exc}"
            raise RetrievalError(msg) from exc

        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("rag_retrieval_total", labels={"strategy": "vector", "status": "success"})
        collector.observe("rag_retrieval_duration_seconds", duration_ms / 1000, labels={"strategy": "vector"})
        collector.observe("rag_retrieval_results_count", len(results), labels={"strategy": "vector"})

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
