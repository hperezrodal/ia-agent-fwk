"""Tests for QdrantMemoryBackend with mocked qdrant client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.memory.backends.qdrant import QdrantMemoryBackend
from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.exceptions import MemoryConfigError


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
) -> QdrantMemoryBackend:
    provider = embedding_provider or MockEmbeddingProvider()
    defaults: dict[str, Any] = {
        "url": "http://localhost:6333",
        "embedding_provider": provider,
        "collection_name": "test_memory",
        "embedding_dimensions": 1536,
        "agent_namespace": "test",
    }
    defaults.update(kwargs)
    return QdrantMemoryBackend(**defaults)


@pytest.mark.unit
class TestQdrantMemoryBackend:
    def test_backend_type(self):
        backend = _make_backend()
        assert backend.backend_type == "qdrant"

    def test_dimension_mismatch_raises(self):
        provider = MockEmbeddingProvider(dimension=768)
        with pytest.raises(MemoryConfigError, match="does not match"):
            _make_backend(embedding_provider=provider, embedding_dimensions=1536)

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_store_generates_embedding(self, mock_qdrant_mod):
        provider = MockEmbeddingProvider()
        provider.embed = AsyncMock(return_value=[[0.1] * 1536])  # type: ignore[method-assign]
        backend = _make_backend(embedding_provider=provider)

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client

        # Mock collection check
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.create_collection = AsyncMock()
        mock_client.upsert = AsyncMock()

        # Inject the mock client
        backend._client = mock_client

        await backend.store("key1", "hello world")
        provider.embed.assert_awaited_once()
        mock_client.upsert.assert_awaited_once()

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_store_with_precomputed_embedding(self, mock_qdrant_mod):
        provider = MockEmbeddingProvider()
        provider.embed = AsyncMock(return_value=[[0.1] * 1536])  # type: ignore[method-assign]
        backend = _make_backend(embedding_provider=provider)

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client

        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.create_collection = AsyncMock()
        mock_client.upsert = AsyncMock()

        backend._client = mock_client

        await backend.store("key1", "hello", metadata={"embedding": [0.5] * 1536})
        provider.embed.assert_not_awaited()

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_retrieve_existing(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="test_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)

        mock_point = MagicMock()
        mock_point.payload = {"key": "key1", "value": "hello world", "agent_namespace": "test", "metadata": {}}
        mock_client.scroll = AsyncMock(return_value=([mock_point], None))

        result = await backend.retrieve("key1")
        assert result == "hello world"

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_retrieve_nonexistent(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="test_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.scroll = AsyncMock(return_value=([], None))

        result = await backend.retrieve("nonexistent")
        assert result is None

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_search_returns_results(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="test_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)

        hit1 = MagicMock()
        hit1.score = 0.95
        hit1.payload = {"key": "key1", "value": "hello", "metadata": {"tag": "test"}}

        hit2 = MagicMock()
        hit2.score = 0.80
        hit2.payload = {"key": "key2", "value": "world", "metadata": {}}

        mock_response = MagicMock()
        mock_response.points = [hit1, hit2]
        mock_client.query_points = AsyncMock(return_value=mock_response)

        results = await backend.search("hello", top_k=5)
        assert len(results) == 2
        assert results[0].key == "key1"
        assert results[0].score == 0.95

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_delete_existing(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="test_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)

        mock_point = MagicMock()
        mock_point.id = "some-uuid"
        mock_client.scroll = AsyncMock(return_value=([mock_point], None))
        mock_client.delete = AsyncMock()

        result = await backend.delete("key1")
        assert result is True
        mock_client.delete.assert_awaited_once()

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_delete_nonexistent(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="test_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.scroll = AsyncMock(return_value=([], None))

        result = await backend.delete("nonexistent")
        assert result is False

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_clear(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="test_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.delete = AsyncMock()

        await backend.clear()
        mock_client.delete.assert_awaited_once()

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_health_check(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections = AsyncMock(return_value=mock_collections)

        result = await backend.health_check()
        assert result is True

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_close(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        await backend.close()
        mock_client.close.assert_awaited_once()

    @patch("ia_agent_fwk.memory.backends.qdrant.qdrant_client")
    async def test_collection_autocreation(self, mock_qdrant_mod):
        backend = _make_backend()

        mock_client = AsyncMock()
        mock_qdrant_mod.AsyncQdrantClient.return_value = mock_client
        backend._client = mock_client

        # Collection does not exist
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.create_collection = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        await backend.retrieve("key1")
        mock_client.create_collection.assert_awaited_once()
