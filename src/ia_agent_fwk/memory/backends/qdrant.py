"""Qdrant vector database memory backend.

Stores text content with vector embeddings for semantic similarity search
using the Qdrant vector database via ``qdrant-client`` async API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.exceptions import MemoryConfigError, MemoryRetrieveError, MemoryStoreError
from ia_agent_fwk.memory.models import MemoryResult
from ia_agent_fwk.observability.metrics import get_metrics_collector

try:
    import qdrant_client
except ImportError:  # pragma: no cover
    qdrant_client = None  # type: ignore[assignment,unused-ignore]

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class QdrantMemoryBackend(MemoryBackend):
    """Qdrant vector database memory backend.

    Parameters
    ----------
    url:
        Qdrant server URL (e.g., ``"http://localhost:6333"``).
    embedding_provider:
        Provider for generating embeddings.
    collection_name:
        Qdrant collection name.
    embedding_dimensions:
        Vector dimension. Must match ``embedding_provider.dimension()``.
    agent_namespace:
        Namespace for multi-agent isolation (stored as payload field).
    api_key:
        Optional Qdrant API key (for Qdrant Cloud).
    distance:
        Distance metric. Default: ``"Cosine"``.

    """

    def __init__(  # noqa: PLR0913
        self,
        url: str,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "agent_memory",
        embedding_dimensions: int = 1536,
        agent_namespace: str = "default",
        api_key: str = "",
        distance: str = "Cosine",
    ) -> None:
        if embedding_provider.dimension() != embedding_dimensions:
            msg = (
                f"Embedding provider dimension ({embedding_provider.dimension()}) "
                f"does not match configured embedding_dimensions ({embedding_dimensions})"
            )
            raise MemoryConfigError(msg)

        self._url = url
        self._embedding_provider = embedding_provider
        self._collection_name = collection_name
        self._embedding_dimensions = embedding_dimensions
        self._agent_namespace = agent_namespace
        self._api_key = api_key
        self._distance = distance
        self._client: AsyncQdrantClient | None = None
        self._collection_ready = False

    @property
    def backend_type(self) -> str:
        """Return ``'qdrant'``."""
        return "qdrant"

    def _get_client(self) -> AsyncQdrantClient:
        """Get or create the Qdrant async client."""
        if self._client is None:
            if qdrant_client is None:  # pragma: no cover
                msg = "qdrant-client is required for QdrantMemoryBackend"
                raise MemoryConfigError(msg)

            kwargs: dict[str, Any] = {"url": self._url}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = qdrant_client.AsyncQdrantClient(**kwargs)
        return self._client

    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        if self._collection_ready:
            return

        models = qdrant_client.models

        client = self._get_client()

        distance_map: dict[str, Any] = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }
        dist = distance_map.get(self._distance, models.Distance.COSINE)

        try:
            collections = await client.get_collections()
            existing = {c.name for c in collections.collections}

            if self._collection_name not in existing:
                await client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._embedding_dimensions,
                        distance=dist,
                    ),
                )
        except Exception as exc:
            msg = f"Failed to ensure Qdrant collection: {exc}"
            raise MemoryStoreError(msg) from exc

        self._collection_ready = True

    @staticmethod
    def _key_to_uuid(key: str, namespace: str) -> str:
        """Generate a deterministic UUID from key + namespace."""
        combined = f"{namespace}:{key}"
        return str(uuid.UUID(hashlib.md5(combined.encode()).hexdigest()))  # noqa: S324

    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Store a value with auto-generated embedding.

        If ``metadata`` contains an ``"embedding"`` key, the pre-computed
        vector is used instead of calling the embedding provider.
        """
        await self._ensure_collection()
        client = self._get_client()

        models = qdrant_client.models

        meta = dict(metadata) if metadata else {}

        # Allow pre-computed embeddings via metadata
        embedding: list[float] | None = meta.pop("embedding", None)
        if embedding is None:
            text_to_embed = value if isinstance(value, str) else json.dumps(value)
            vectors = await self._embedding_provider.embed([text_to_embed])
            embedding = vectors[0]

        value_str = value if isinstance(value, str) else json.dumps(value)
        point_id = self._key_to_uuid(key, self._agent_namespace)

        payload = {
            "key": key,
            "value": value_str,
            "agent_namespace": self._agent_namespace,
            "metadata": meta,
        }

        point = models.PointStruct(id=point_id, vector=embedding, payload=payload)

        collector = get_metrics_collector()
        t0 = time.monotonic()
        try:
            await client.upsert(
                collection_name=self._collection_name,
                points=[point],
            )
            duration_ms = (time.monotonic() - t0) * 1000
            collector.increment("qdrant_operations_total", labels={"operation": "store", "status": "success"})
            collector.observe("qdrant_operation_duration_seconds", duration_ms / 1000, labels={"operation": "store"})
        except Exception as exc:
            collector.increment("qdrant_operations_total", labels={"operation": "store", "status": "failure"})
            msg = f"Failed to store key {key!r}: {exc}"
            raise MemoryStoreError(msg) from exc

    async def retrieve(self, key: str) -> Any | None:
        """Retrieve a value by key using scroll with key filter."""
        await self._ensure_collection()
        client = self._get_client()

        models = qdrant_client.models

        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="key", match=models.MatchValue(value=key)),
                models.FieldCondition(key="agent_namespace", match=models.MatchValue(value=self._agent_namespace)),
            ]
        )

        try:
            results, _offset = await client.scroll(
                collection_name=self._collection_name,
                scroll_filter=query_filter,
                limit=1,
            )
        except Exception as exc:
            msg = f"Failed to retrieve key {key!r}: {exc}"
            raise MemoryRetrieveError(msg) from exc

        if not results:
            return None

        payload = results[0].payload
        if payload is None:
            return None
        return payload.get("value")

    async def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryResult]:
        """Perform vector similarity search.

        Parameters
        ----------
        query:
            Text to search for (will be embedded).
        top_k:
            Maximum number of results.
        **kwargs:
            Optional ``score_threshold`` (float) and ``metadata_filter`` (dict).

        """
        await self._ensure_collection()
        client = self._get_client()

        models = qdrant_client.models

        score_threshold: float = kwargs.get("score_threshold", 0.0)
        metadata_filter: dict[str, Any] | None = kwargs.get("metadata_filter")

        vectors = await self._embedding_provider.embed([query])
        query_vector = vectors[0]

        # Build filter conditions
        must_conditions = [
            models.FieldCondition(key="agent_namespace", match=models.MatchValue(value=self._agent_namespace)),
        ]

        if metadata_filter:
            for filter_key, filter_value in metadata_filter.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{filter_key}",
                        match=models.MatchValue(value=filter_value),
                    )
                )

        query_filter = models.Filter(must=must_conditions)

        query_kwargs: dict[str, Any] = {
            "collection_name": self._collection_name,
            "query": query_vector,
            "query_filter": query_filter,
            "limit": top_k,
            "with_payload": True,
        }

        if score_threshold > 0:
            query_kwargs["score_threshold"] = score_threshold

        collector = get_metrics_collector()
        t0 = time.monotonic()
        try:
            response = await client.query_points(**query_kwargs)
        except Exception as exc:
            collector.increment("qdrant_operations_total", labels={"operation": "search", "status": "failure"})
            msg = f"Failed to search: {exc}"
            raise MemoryRetrieveError(msg) from exc

        duration_ms = (time.monotonic() - t0) * 1000
        results: list[MemoryResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                MemoryResult(
                    key=payload.get("key", ""),
                    value=payload.get("value", ""),
                    score=point.score,
                    metadata=payload.get("metadata"),
                )
            )
        collector.increment("qdrant_operations_total", labels={"operation": "search", "status": "success"})
        collector.observe("qdrant_operation_duration_seconds", duration_ms / 1000, labels={"operation": "search"})
        collector.observe("qdrant_search_results_count", len(results))
        return results

    async def delete(self, key: str) -> bool:
        """Delete an entry by key."""
        await self._ensure_collection()
        client = self._get_client()

        models = qdrant_client.models

        # Check if the point exists first
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="key", match=models.MatchValue(value=key)),
                models.FieldCondition(key="agent_namespace", match=models.MatchValue(value=self._agent_namespace)),
            ]
        )

        try:
            results, _offset = await client.scroll(
                collection_name=self._collection_name,
                scroll_filter=query_filter,
                limit=1,
            )

            if not results:
                return False

            point_ids = [r.id for r in results]
            await client.delete(
                collection_name=self._collection_name,
                points_selector=point_ids,
            )
        except Exception as exc:
            msg = f"Failed to delete key {key!r}: {exc}"
            raise MemoryStoreError(msg) from exc

        return True

    async def clear(self) -> None:
        """Remove all entries for the configured namespace."""
        await self._ensure_collection()
        client = self._get_client()

        models = qdrant_client.models

        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="agent_namespace", match=models.MatchValue(value=self._agent_namespace)),
            ]
        )

        try:
            await client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
            )
        except Exception as exc:
            msg = f"Failed to clear collection: {exc}"
            raise MemoryStoreError(msg) from exc

    async def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        collector = get_metrics_collector()
        try:
            client = self._get_client()
            collections = await client.get_collections()
            _ = collections.collections  # verify response is valid
        except Exception:
            collector.increment("qdrant_health_checks_total", labels={"status": "failure"})
            logger.exception("Qdrant health check failed")
            return False
        collector.increment("qdrant_health_checks_total", labels={"status": "success"})
        return True

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._collection_ready = False
