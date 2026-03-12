"""RAG store adapter for chunk storage and retrieval."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.rag.exceptions import EmbeddingError

if TYPE_CHECKING:
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
    from ia_agent_fwk.memory.models import MemoryResult
    from ia_agent_fwk.rag.models import Chunk

logger = logging.getLogger(__name__)


class RAGStore:
    """Adapt chunk storage operations to vector memory backends.

    Provides batch storage, search, and document deletion over
    the underlying MemoryBackend.
    """

    def __init__(
        self,
        memory_backend: MemoryBackend,
        embedding_provider: EmbeddingProvider,
        *,
        collection: str = "rag_chunks",
    ) -> None:
        self._backend = memory_backend
        self._embedding = embedding_provider
        self._collection = collection

    async def store_chunks(
        self,
        chunks: list[Chunk],
        document_id: str,
    ) -> int:
        """Store a batch of chunks with their embeddings.

        Returns the number of chunks stored.
        """
        collector = get_metrics_collector()
        if not chunks:
            return 0

        # Batch embed
        t0 = time.monotonic()
        texts = [c.content for c in chunks]
        try:
            embeddings = await self._embedding.embed(texts)
        except Exception as exc:
            collector.increment("rag_store_errors_total", labels={"operation": "embed"})
            msg = f"Failed to embed {len(texts)} chunks: {exc}"
            raise EmbeddingError(msg) from exc

        stored = 0
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            key = f"{document_id}:{chunk.chunk_index}"
            metadata: dict[str, Any] = {
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "source": chunk.source,
                "collection": self._collection,
                "embedding": embedding,
                **chunk.metadata,
            }
            await self._backend.store(
                key=key,
                value=chunk.content,
                metadata=metadata,
            )
            stored += 1

        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("rag_store_chunks_total", value=stored)
        collector.observe("rag_store_duration_seconds", duration_ms / 1000)

        logger.info(
            "Stored %d chunks for document '%s' in collection '%s' (%.0fms)",
            stored,
            document_id,
            self._collection,
            duration_ms,
            extra={
                "rag_data": {
                    "event": "chunks_stored",
                    "document_id": document_id,
                    "chunks_stored": stored,
                    "collection": self._collection,
                    "duration_ms": round(duration_ms, 1),
                }
            },
        )
        return stored

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryResult]:
        """Search for relevant chunks by semantic similarity."""
        collector = get_metrics_collector()
        t0 = time.monotonic()
        results = await self._backend.search(  # type: ignore[call-arg]
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("rag_store_search_total")
        collector.observe("rag_store_search_duration_seconds", duration_ms / 1000)
        return results

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.

        Returns the number of chunks deleted.
        """
        # Search for all chunks with this document_id
        results = await self._backend.search(  # type: ignore[call-arg]
            query="",
            top_k=1000,
            metadata_filter={"document_id": document_id},
        )

        deleted = 0
        for result in results:
            key = result.key if hasattr(result, "key") else f"{document_id}:{deleted}"
            try:
                await self._backend.delete(key)
                deleted += 1
            except Exception:  # noqa: BLE001
                logger.warning("Failed to delete chunk %s", key)

        collector = get_metrics_collector()
        collector.increment("rag_store_delete_total")
        collector.observe("rag_store_chunks_deleted", deleted)
        logger.info("Deleted %d chunks for document '%s'", deleted, document_id)
        return deleted
