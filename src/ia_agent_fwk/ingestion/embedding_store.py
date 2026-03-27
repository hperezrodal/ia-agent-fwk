"""Embedding + Storage — encapsulates embed + store in one step.

Takes ProcessedChunks from the chunking pipeline and:
1. Generates dense embeddings (via any embedding provider)
2. Generates sparse BM25 vectors (via fastembed)
3. Stores both in Qdrant with full content + metadata

The embedding provider is injected, so it works with Ollama, OpenAI,
or any provider that implements embed(texts) → list[list[float]].

Usage:
    store = EmbeddingStore(
        embedding_provider=OllamaEmbeddingProvider(model="nomic-embed-text"),
        qdrant_url="http://localhost:6333",
        collection_name="my_collection",
        dense_dim=768,
    )
    stored = await store.store_chunks(chunks, file_name="doc.pdf")
    results = store.search("query text", top_k=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ia_agent_fwk.ingestion.hybrid_store import HybridSearchResult, HybridStore, SearchMode
from ia_agent_fwk.ingestion.models import ProcessedChunk

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for any embedding provider (Ollama, OpenAI, etc.)."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class StoreConfig:
    """Configuration for embedding + storage."""

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "ia_agent_fwk_rag"
    dense_dim: int = 768
    embedding_model: str = ""  # stored in collection for auto-detection
    max_embed_chars: int = 4000  # truncate content for embedding (token limits)
    batch_size: int = 50  # chunks per embedding batch call


class EmbeddingStore:
    """Encapsulates embed + store: ProcessedChunks → Qdrant.

    Parameters
    ----------
    embedding_provider:
        Any object with async embed(texts) → list[list[float]].
    config:
        Storage configuration.

    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: StoreConfig | None = None,
    ) -> None:
        self._provider = embedding_provider
        self._config = config or StoreConfig()
        self._store = HybridStore(
            url=self._config.qdrant_url,
            collection_name=self._config.collection_name,
            dense_dim=self._config.dense_dim,
            embedding_model=self._config.embedding_model,
        )

    async def store_chunks(
        self,
        chunks: list[ProcessedChunk],
        file_name: str = "unknown",
    ) -> int:
        """Embed and store chunks. Returns number of chunks stored.

        1. Truncate content for embedding (full content stored in payload)
        2. Batch-embed with dense provider
        3. Upsert to Qdrant with dense + sparse (BM25) vectors
        """
        if not chunks:
            return 0

        contents = [c.content for c in chunks]
        chunk_ids = [f"{file_name}:chunk:{i}" for i in range(len(chunks))]
        metadatas = [c.metadata for c in chunks]

        # Truncate for embedding only — full content goes to payload
        max_chars = self._config.max_embed_chars
        embed_contents = [c[:max_chars] for c in contents]

        # Batch embed with provider
        dense_embeddings = await self._batch_embed(embed_contents)

        # Upsert to Qdrant (dense + sparse)
        self._store.upsert_batch(chunk_ids, contents, dense_embeddings, metadatas)

        logger.info("Stored %d chunks for %s", len(chunks), file_name)
        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        mode: SearchMode | None = None,
    ) -> list[HybridSearchResult]:
        """Search — handles dual encoding internally.

        Parameters
        ----------
        query:
            Natural language query string.
        top_k:
            Number of results.
        filters:
            Metadata filters (exact match, server-side).
        mode:
            SearchMode.HYBRID (default), DENSE (semantic only), or SPARSE (keyword only).

        The caller just passes the query string. No need to pre-embed.

        """
        dense_embedding = (await self._provider.embed([query]))[0]
        return self._store.search(
            query=query,
            dense_embedding=dense_embedding,
            top_k=top_k,
            filters=filters,
            mode=mode,
        )

    def delete_collection(self) -> None:
        """Delete the underlying Qdrant collection."""
        self._store.delete_collection()

    def collection_info(self) -> dict[str, Any]:
        """Get collection stats."""
        return self._store.collection_info()

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to respect provider limits."""
        all_embeddings: list[list[float]] = []
        batch_size = self._config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self._provider.embed(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings
