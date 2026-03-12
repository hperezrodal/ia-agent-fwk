"""Tests for PgVectorMemoryBackend with mocked asyncpg."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.memory.backends.pgvector import PgVectorMemoryBackend
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
) -> PgVectorMemoryBackend:
    provider = embedding_provider or MockEmbeddingProvider()
    defaults: dict[str, Any] = {
        "database_url": "postgresql://test:test@localhost:5432/test",
        "embedding_provider": provider,
        "collection_name": "test_vectors",
        "embedding_dimensions": 1536,
        "agent_namespace": "test",
    }
    defaults.update(kwargs)
    return PgVectorMemoryBackend(**defaults)


def _setup_mock_pool(mock_asyncpg: MagicMock) -> tuple[MagicMock, AsyncMock]:
    """Create a properly mocked asyncpg pool with async context manager support."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = mock_ctx
    mock_pool.close = AsyncMock()

    mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
    return mock_pool, mock_conn


@pytest.mark.unit
class TestPgVectorMemoryBackend:
    def test_backend_type(self):
        backend = _make_backend()
        assert backend.backend_type == "pgvector"

    def test_dimension_mismatch_raises(self):
        provider = MockEmbeddingProvider(dimension=768)
        with pytest.raises(MemoryConfigError, match="does not match"):
            _make_backend(embedding_provider=provider, embedding_dimensions=1536)

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_store_and_retrieve(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        # Store
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        await backend.store("key1", "hello world")

        # Verify embed was called (indirectly, since we use mock provider)
        assert mock_conn.execute.await_count > 0

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_store_generates_embedding(self, mock_asyncpg):
        provider = MockEmbeddingProvider()
        provider.embed = AsyncMock(return_value=[[0.1] * 1536])  # type: ignore[method-assign]
        backend = _make_backend(embedding_provider=provider)

        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")

        await backend.store("key1", "hello world")
        provider.embed.assert_awaited_once()

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_store_with_precomputed_embedding(self, mock_asyncpg):
        provider = MockEmbeddingProvider()
        provider.embed = AsyncMock(return_value=[[0.1] * 1536])  # type: ignore[method-assign]
        backend = _make_backend(embedding_provider=provider)

        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")

        await backend.store("key1", "hello", metadata={"embedding": [0.5] * 1536})
        provider.embed.assert_not_awaited()

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_retrieve_existing(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.fetchrow = AsyncMock(return_value={"value": "hello world"})
        result = await backend.retrieve("key1")
        assert result == "hello world"

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_retrieve_nonexistent(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.fetchrow = AsyncMock(return_value=None)
        result = await backend.retrieve("nonexistent")
        assert result is None

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_search_returns_results(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.fetch = AsyncMock(
            return_value=[
                {"key": "key1", "value": "hello", "score": 0.95, "metadata": "{}"},
                {"key": "key2", "value": "world", "score": 0.80, "metadata": "{}"},
            ]
        )

        results = await backend.search("hello", top_k=5)
        assert len(results) == 2
        assert results[0].key == "key1"
        assert results[0].score == 0.95

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_delete_existing(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.execute = AsyncMock(return_value="DELETE 1")
        result = await backend.delete("key1")
        assert result is True

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_delete_nonexistent(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.execute = AsyncMock(return_value="DELETE 0")
        result = await backend.delete("nonexistent")
        assert result is False

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_clear(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        await backend.clear()
        # Verify execute was called (for table creation + clear)
        assert mock_conn.execute.await_count > 0

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_health_check(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, _mock_conn = _setup_mock_pool(mock_asyncpg)

        result = await backend.health_check()
        assert result is True

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_close(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, _mock_conn = _setup_mock_pool(mock_asyncpg)

        # Initialize the pool
        await backend.health_check()
        await backend.close()
        mock_pool.close.assert_awaited_once()

    @patch("ia_agent_fwk.memory.backends.pgvector.asyncpg")
    async def test_table_autocreation(self, mock_asyncpg):
        backend = _make_backend()
        _mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.fetchrow = AsyncMock(return_value=None)

        # First operation triggers table creation
        await backend.retrieve("key1")

        # Verify CREATE TABLE and CREATE INDEX calls
        execute_calls = [str(c) for c in mock_conn.execute.call_args_list]
        call_texts = " ".join(execute_calls)
        assert "CREATE TABLE" in call_texts or "CREATE EXTENSION" in call_texts

    def test_vector_to_sql(self):
        result = PgVectorMemoryBackend._vector_to_sql([0.1, 0.2, 0.3])
        assert result == "[0.1,0.2,0.3]"
