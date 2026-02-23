"""Weaviate vector database memory backend.

Stores text content with vector embeddings for semantic similarity search
using the Weaviate vector database via ``weaviate-client`` v4 API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.exceptions import MemoryConfigError, MemoryRetrieveError, MemoryStoreError
from ia_agent_fwk.memory.models import MemoryResult

try:
    import weaviate
except ImportError:  # pragma: no cover
    weaviate = None  # type: ignore[assignment,unused-ignore]

if TYPE_CHECKING:
    from weaviate import WeaviateClient

    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class WeaviateMemoryBackend(MemoryBackend):
    """Weaviate vector database memory backend.

    Parameters
    ----------
    url:
        Weaviate server URL (e.g., ``"http://localhost:8080"``).
    embedding_provider:
        Provider for generating embeddings.
    collection_name:
        Weaviate collection name.
    embedding_dimensions:
        Vector dimension. Must match ``embedding_provider.dimension()``.
    agent_namespace:
        Namespace for multi-agent isolation (stored as a property).
    api_key:
        Optional Weaviate API key.

    """

    def __init__(  # noqa: PLR0913
        self,
        url: str,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "AgentMemory",
        embedding_dimensions: int = 1536,
        agent_namespace: str = "default",
        api_key: str = "",
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
        self._client: WeaviateClient | None = None
        self._collection_ready = False

    @property
    def backend_type(self) -> str:
        """Return ``'weaviate'``."""
        return "weaviate"

    def _get_client(self) -> WeaviateClient:
        """Get or create the Weaviate client."""
        if self._client is None:
            if weaviate is None:  # pragma: no cover
                msg = "weaviate-client is required for WeaviateMemoryBackend"
                raise MemoryConfigError(msg)

            if self._api_key:
                self._client = weaviate.connect_to_custom(
                    http_host=self._url.replace("http://", "").replace("https://", "").split(":")[0],
                    http_port=int(self._url.rstrip("/").split(":")[-1]) if ":" in self._url.split("//")[-1] else 8080,
                    http_secure=self._url.startswith("https"),
                    grpc_host=self._url.replace("http://", "").replace("https://", "").split(":")[0],
                    grpc_port=50051,
                    grpc_secure=self._url.startswith("https"),
                    auth_credentials=weaviate.auth.AuthApiKey(self._api_key),
                )
            else:
                self._client = weaviate.connect_to_local(
                    host=self._url.replace("http://", "").replace("https://", "").split(":")[0],
                    port=int(self._url.rstrip("/").split(":")[-1]) if ":" in self._url.split("//")[-1] else 8080,
                )
        return self._client

    def _ensure_collection(self) -> None:
        """Create the Weaviate collection if it does not exist."""
        if self._collection_ready:
            return

        client = self._get_client()

        try:
            if not client.collections.exists(self._collection_name):
                client.collections.create(
                    name=self._collection_name,
                    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                )
        except Exception as exc:
            msg = f"Failed to ensure Weaviate collection: {exc}"
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
        self._ensure_collection()
        client = self._get_client()

        meta = dict(metadata) if metadata else {}

        # Allow pre-computed embeddings via metadata
        embedding: list[float] | None = meta.pop("embedding", None)
        if embedding is None:
            text_to_embed = value if isinstance(value, str) else json.dumps(value)
            vectors = await self._embedding_provider.embed([text_to_embed])
            embedding = vectors[0]

        value_str = value if isinstance(value, str) else json.dumps(value)
        obj_uuid = self._key_to_uuid(key, self._agent_namespace)

        properties = {
            "key": key,
            "value": value_str,
            "agent_namespace": self._agent_namespace,
            "metadata_json": json.dumps(meta),
        }

        collection = client.collections.get(self._collection_name)

        try:
            existing = collection.query.fetch_object_by_id(obj_uuid)
            if existing is not None:
                collection.data.update(
                    uuid=obj_uuid,
                    properties=properties,
                    vector=embedding,
                )
            else:
                collection.data.insert(
                    uuid=obj_uuid,
                    properties=properties,
                    vector=embedding,
                )
        except Exception as exc:
            msg = f"Failed to store key {key!r}: {exc}"
            raise MemoryStoreError(msg) from exc

    async def retrieve(self, key: str) -> Any | None:
        """Retrieve a value by key using filter."""
        self._ensure_collection()
        client = self._get_client()

        collection = client.collections.get(self._collection_name)

        try:
            response = collection.query.fetch_objects(
                filters=(
                    weaviate.classes.query.Filter.by_property("key").equal(key)
                    & weaviate.classes.query.Filter.by_property("agent_namespace").equal(self._agent_namespace)
                ),
                limit=1,
            )
        except Exception as exc:
            msg = f"Failed to retrieve key {key!r}: {exc}"
            raise MemoryRetrieveError(msg) from exc

        if not response.objects:
            return None

        props = response.objects[0].properties
        return props.get("value")

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
        self._ensure_collection()
        client = self._get_client()

        score_threshold: float = kwargs.get("score_threshold", 0.0)
        metadata_filter: dict[str, Any] | None = kwargs.get("metadata_filter")

        vectors = await self._embedding_provider.embed([query])
        query_vector = vectors[0]

        collection = client.collections.get(self._collection_name)

        # Build filter for namespace
        filters = weaviate.classes.query.Filter.by_property("agent_namespace").equal(self._agent_namespace)

        if metadata_filter:
            for filter_key, filter_value in metadata_filter.items():
                filters = filters & weaviate.classes.query.Filter.by_property(filter_key).equal(filter_value)

        try:
            response = collection.query.near_vector(
                near_vector=query_vector,
                filters=filters,
                limit=top_k,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            )
        except Exception as exc:
            msg = f"Failed to search: {exc}"
            raise MemoryRetrieveError(msg) from exc

        results: list[MemoryResult] = []
        for obj in response.objects:
            props = obj.properties
            # Weaviate returns distance; convert to similarity score (1 - distance for cosine)
            distance = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 0.0
            score = 1.0 - distance

            if score_threshold > 0 and score < score_threshold:
                continue

            meta: dict[str, Any] = {}
            metadata_json = props.get("metadata_json", "{}")
            if isinstance(metadata_json, str):
                try:
                    meta = json.loads(metadata_json)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            results.append(
                MemoryResult(
                    key=props.get("key", ""),
                    value=props.get("value", ""),
                    score=score,
                    metadata=meta,
                )
            )
        return results

    async def delete(self, key: str) -> bool:
        """Delete an entry by key."""
        self._ensure_collection()
        client = self._get_client()

        collection = client.collections.get(self._collection_name)

        try:
            response = collection.query.fetch_objects(
                filters=(
                    weaviate.classes.query.Filter.by_property("key").equal(key)
                    & weaviate.classes.query.Filter.by_property("agent_namespace").equal(self._agent_namespace)
                ),
                limit=1,
            )

            if not response.objects:
                return False

            obj_uuid = response.objects[0].uuid
            collection.data.delete_by_id(obj_uuid)
        except Exception as exc:
            msg = f"Failed to delete key {key!r}: {exc}"
            raise MemoryStoreError(msg) from exc

        return True

    async def clear(self) -> None:
        """Remove all entries for the configured namespace."""
        self._ensure_collection()
        client = self._get_client()

        collection = client.collections.get(self._collection_name)

        try:
            collection.data.delete_many(
                where=weaviate.classes.query.Filter.by_property("agent_namespace").equal(self._agent_namespace),
            )
        except Exception as exc:
            msg = f"Failed to clear collection: {exc}"
            raise MemoryStoreError(msg) from exc

    async def health_check(self) -> bool:
        """Check Weaviate connectivity."""
        try:
            client = self._get_client()
            return bool(client.is_ready())
        except Exception:
            logger.exception("Weaviate health check failed")
            return False

    async def close(self) -> None:
        """Close the Weaviate client."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection_ready = False
