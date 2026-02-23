"""Tests for WeaviateMemoryBackend with mocked weaviate client."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ia_agent_fwk.memory.backends.weaviate_backend import WeaviateMemoryBackend
from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.exceptions import MemoryConfigError, MemoryRetrieveError, MemoryStoreError


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider that returns deterministic vectors."""

    def __init__(self, dimension: int = 1536) -> None:
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]

    def dimension(self) -> int:
        return self._dimension

    def max_tokens(self) -> int:
        return 8191


def _make_backend(
    embedding_provider: EmbeddingProvider | None = None,
    **kwargs: Any,
) -> WeaviateMemoryBackend:
    provider = embedding_provider or MockEmbeddingProvider()
    defaults: dict[str, Any] = {
        "url": "http://localhost:8080",
        "embedding_provider": provider,
        "collection_name": "TestMemory",
        "embedding_dimensions": 1536,
        "agent_namespace": "test",
    }
    defaults.update(kwargs)
    return WeaviateMemoryBackend(**defaults)


def _mock_weaviate_module() -> MagicMock:
    """Create a mock weaviate module with necessary nested classes."""
    mock_mod = MagicMock()

    # Mock Filter class with chainable methods
    mock_filter_instance = MagicMock()
    mock_filter_instance.__and__ = MagicMock(return_value=mock_filter_instance)
    mock_filter_by_property = MagicMock()
    mock_filter_by_property.equal = MagicMock(return_value=mock_filter_instance)
    mock_mod.classes.query.Filter.by_property = MagicMock(return_value=mock_filter_by_property)

    # Mock MetadataQuery
    mock_mod.classes.query.MetadataQuery = MagicMock(return_value=MagicMock())

    # Mock Configure.Vectorizer.none()
    mock_mod.classes.config.Configure.Vectorizer.none = MagicMock(return_value=MagicMock())

    # Mock auth
    mock_mod.auth.AuthApiKey = MagicMock(return_value=MagicMock())

    return mock_mod


def _mock_collection() -> MagicMock:
    """Create a mock Weaviate collection object."""
    collection = MagicMock()
    collection.query = MagicMock()
    collection.data = MagicMock()
    return collection


@pytest.mark.unit
class TestWeaviateMemoryBackend:
    def test_backend_type(self):
        backend = _make_backend()
        assert backend.backend_type == "weaviate"

    def test_dimension_mismatch_raises(self):
        provider = MockEmbeddingProvider(dimension=768)
        with pytest.raises(MemoryConfigError, match="does not match"):
            _make_backend(embedding_provider=provider, embedding_dimensions=1536)

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_store_generates_embedding(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes
        mock_weaviate_mod.classes.config = _mock_weaviate_module().classes.config

        provider = MockEmbeddingProvider()
        backend = _make_backend(embedding_provider=provider)

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        # Mock fetch_object_by_id to return None (object doesn't exist)
        mock_collection.query.fetch_object_by_id.return_value = None

        backend._client = mock_client

        await backend.store("key1", "hello world")
        mock_collection.data.insert.assert_called_once()

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_store_with_precomputed_embedding(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes
        mock_weaviate_mod.classes.config = _mock_weaviate_module().classes.config

        provider = MockEmbeddingProvider()
        # Track embed calls
        original_embed = provider.embed
        embed_called = False

        async def tracking_embed(texts):
            nonlocal embed_called
            embed_called = True
            return await original_embed(texts)

        provider.embed = tracking_embed  # type: ignore[method-assign]
        backend = _make_backend(embedding_provider=provider)

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True
        mock_collection.query.fetch_object_by_id.return_value = None

        backend._client = mock_client

        await backend.store("key1", "hello", metadata={"embedding": [0.5] * 1536})
        assert not embed_called

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_store_updates_existing(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes
        mock_weaviate_mod.classes.config = _mock_weaviate_module().classes.config

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        # Object already exists
        mock_existing = MagicMock()
        mock_collection.query.fetch_object_by_id.return_value = mock_existing

        backend._client = mock_client

        await backend.store("key1", "updated value")
        mock_collection.data.update.assert_called_once()

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_store_error_raises(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes
        mock_weaviate_mod.classes.config = _mock_weaviate_module().classes.config

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        # Both fetch_object_by_id and insert raise
        mock_collection.query.fetch_object_by_id.side_effect = RuntimeError("connection error")
        mock_collection.data.insert.side_effect = RuntimeError("connection error")

        backend._client = mock_client

        with pytest.raises(MemoryStoreError, match="Failed to store"):
            await backend.store("key1", "value")

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_retrieve_existing(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        mock_obj = MagicMock()
        mock_obj.properties = {"key": "key1", "value": "hello world", "agent_namespace": "test", "metadata_json": "{}"}
        mock_response = MagicMock()
        mock_response.objects = [mock_obj]
        mock_collection.query.fetch_objects.return_value = mock_response

        backend._client = mock_client

        result = await backend.retrieve("key1")
        assert result == "hello world"

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_retrieve_nonexistent(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        mock_response = MagicMock()
        mock_response.objects = []
        mock_collection.query.fetch_objects.return_value = mock_response

        backend._client = mock_client

        result = await backend.retrieve("nonexistent")
        assert result is None

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_retrieve_error_raises(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True
        mock_collection.query.fetch_objects.side_effect = RuntimeError("connection error")

        backend._client = mock_client

        with pytest.raises(MemoryRetrieveError, match="Failed to retrieve"):
            await backend.retrieve("key1")

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_search_returns_results(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        # Create mock search results
        obj1 = MagicMock()
        obj1.properties = {"key": "key1", "value": "hello", "metadata_json": json.dumps({"tag": "test"})}
        obj1.metadata = MagicMock()
        obj1.metadata.distance = 0.05  # score = 0.95

        obj2 = MagicMock()
        obj2.properties = {"key": "key2", "value": "world", "metadata_json": "{}"}
        obj2.metadata = MagicMock()
        obj2.metadata.distance = 0.2  # score = 0.80

        mock_response = MagicMock()
        mock_response.objects = [obj1, obj2]
        mock_collection.query.near_vector.return_value = mock_response

        backend._client = mock_client

        results = await backend.search("hello", top_k=5)
        assert len(results) == 2
        assert results[0].key == "key1"
        assert results[0].score == pytest.approx(0.95)
        assert results[0].metadata == {"tag": "test"}

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_search_with_score_threshold(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        obj1 = MagicMock()
        obj1.properties = {"key": "key1", "value": "hello", "metadata_json": "{}"}
        obj1.metadata = MagicMock()
        obj1.metadata.distance = 0.05  # score = 0.95

        obj2 = MagicMock()
        obj2.properties = {"key": "key2", "value": "world", "metadata_json": "{}"}
        obj2.metadata = MagicMock()
        obj2.metadata.distance = 0.5  # score = 0.50

        mock_response = MagicMock()
        mock_response.objects = [obj1, obj2]
        mock_collection.query.near_vector.return_value = mock_response

        backend._client = mock_client

        results = await backend.search("hello", top_k=5, score_threshold=0.7)
        assert len(results) == 1
        assert results[0].key == "key1"

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_search_error_raises(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True
        mock_collection.query.near_vector.side_effect = RuntimeError("search error")

        backend._client = mock_client

        with pytest.raises(MemoryRetrieveError, match="Failed to search"):
            await backend.search("hello")

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_delete_existing(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        mock_obj = MagicMock()
        mock_obj.uuid = "some-uuid"
        mock_response = MagicMock()
        mock_response.objects = [mock_obj]
        mock_collection.query.fetch_objects.return_value = mock_response

        backend._client = mock_client

        result = await backend.delete("key1")
        assert result is True
        mock_collection.data.delete_by_id.assert_called_once_with("some-uuid")

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_delete_nonexistent(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        mock_response = MagicMock()
        mock_response.objects = []
        mock_collection.query.fetch_objects.return_value = mock_response

        backend._client = mock_client

        result = await backend.delete("nonexistent")
        assert result is False

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_delete_error_raises(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True
        mock_collection.query.fetch_objects.side_effect = RuntimeError("connection error")

        backend._client = mock_client

        with pytest.raises(MemoryStoreError, match="Failed to delete"):
            await backend.delete("key1")

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_clear(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        backend._client = mock_client

        await backend.clear()
        mock_collection.data.delete_many.assert_called_once()

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_clear_error_raises(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True
        mock_collection.data.delete_many.side_effect = RuntimeError("clear error")

        backend._client = mock_client

        with pytest.raises(MemoryStoreError, match="Failed to clear"):
            await backend.clear()

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_health_check_healthy(self, mock_weaviate_mod):
        backend = _make_backend()

        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        backend._client = mock_client

        result = await backend.health_check()
        assert result is True

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_health_check_unhealthy(self, mock_weaviate_mod):
        backend = _make_backend()

        mock_client = MagicMock()
        mock_client.is_ready.side_effect = RuntimeError("connection failed")
        backend._client = mock_client

        result = await backend.health_check()
        assert result is False

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_close(self, mock_weaviate_mod):
        backend = _make_backend()

        mock_client = MagicMock()
        backend._client = mock_client

        await backend.close()
        mock_client.close.assert_called_once()
        assert backend._client is None
        assert backend._collection_ready is False

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_close_when_no_client(self, mock_weaviate_mod):
        backend = _make_backend()
        backend._client = None

        # Should not raise
        await backend.close()

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_collection_autocreation(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes
        mock_weaviate_mod.classes.config = _mock_weaviate_module().classes.config

        backend = _make_backend()

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.return_value = MagicMock()

        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection

        mock_response = MagicMock()
        mock_response.objects = []
        mock_collection.query.fetch_objects.return_value = mock_response

        backend._client = mock_client

        await backend.retrieve("key1")
        mock_client.collections.create.assert_called_once()

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_collection_already_exists(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True

        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection

        mock_response = MagicMock()
        mock_response.objects = []
        mock_collection.query.fetch_objects.return_value = mock_response

        backend._client = mock_client

        await backend.retrieve("key1")
        mock_client.collections.create.assert_not_called()

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_ensure_collection_error(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_client.collections.exists.side_effect = RuntimeError("connection error")

        backend._client = mock_client

        with pytest.raises(MemoryStoreError, match="Failed to ensure Weaviate collection"):
            await backend.retrieve("key1")

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_store_json_value(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes
        mock_weaviate_mod.classes.config = _mock_weaviate_module().classes.config

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True
        mock_collection.query.fetch_object_by_id.return_value = None

        backend._client = mock_client

        value = {"nested": "data", "count": 42}
        await backend.store("key1", value)

        call_kwargs = mock_collection.data.insert.call_args
        assert call_kwargs is not None
        stored_value = call_kwargs.kwargs.get("properties", call_kwargs[1].get("properties", {}))
        assert stored_value["value"] == json.dumps(value)

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_search_with_metadata_filter(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        mock_response = MagicMock()
        mock_response.objects = []
        mock_collection.query.near_vector.return_value = mock_response

        backend._client = mock_client

        results = await backend.search("hello", top_k=5, metadata_filter={"category": "test"})
        assert len(results) == 0
        mock_collection.query.near_vector.assert_called_once()

    def test_key_to_uuid_deterministic(self):
        uuid1 = WeaviateMemoryBackend._key_to_uuid("key1", "ns1")
        uuid2 = WeaviateMemoryBackend._key_to_uuid("key1", "ns1")
        uuid3 = WeaviateMemoryBackend._key_to_uuid("key2", "ns1")
        assert uuid1 == uuid2
        assert uuid1 != uuid3

    @patch("ia_agent_fwk.memory.backends.weaviate_backend.weaviate")
    async def test_search_handles_invalid_metadata_json(self, mock_weaviate_mod):
        mock_weaviate_mod.classes = _mock_weaviate_module().classes

        backend = _make_backend()

        mock_client = MagicMock()
        mock_collection = _mock_collection()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True

        obj = MagicMock()
        obj.properties = {"key": "key1", "value": "hello", "metadata_json": "invalid json{"}
        obj.metadata = MagicMock()
        obj.metadata.distance = 0.1

        mock_response = MagicMock()
        mock_response.objects = [obj]
        mock_collection.query.near_vector.return_value = mock_response

        backend._client = mock_client

        results = await backend.search("hello")
        assert len(results) == 1
        assert results[0].metadata == {}
