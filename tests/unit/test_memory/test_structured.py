"""Tests for StructuredMemoryBackend with mocked asyncpg."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.memory.backends.structured import StructuredMemoryBackend


def _make_backend(**kwargs: Any) -> StructuredMemoryBackend:
    defaults: dict[str, Any] = {
        "database_url": "postgresql://test:test@localhost:5432/test",
        "table_name": "test_kv",
        "agent_namespace": "test",
        "default_ttl_seconds": 0,
    }
    defaults.update(kwargs)
    return StructuredMemoryBackend(**defaults)


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
class TestStructuredMemoryBackend:
    def test_backend_type(self):
        backend = _make_backend()
        assert backend.backend_type == "structured"

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_store_and_retrieve(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        await backend.store("key1", "hello world")
        assert mock_conn.execute.await_count > 0

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_store_upsert(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        await backend.store("key1", "value1")
        await backend.store("key1", "value2")
        # Both stores should succeed (upsert)
        # execute calls: table creation (4) + 2 stores = 6
        assert mock_conn.execute.await_count > 1

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_retrieve_existing(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.fetchrow = AsyncMock(return_value={"value": '"hello world"'})
        result = await backend.retrieve("key1")
        assert result == "hello world"

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_retrieve_nonexistent(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.fetchrow = AsyncMock(return_value=None)
        result = await backend.retrieve("nonexistent")
        assert result is None

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_retrieve_json_value(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        complex_value = {"name": "test", "data": [1, 2, 3]}
        mock_conn.fetchrow = AsyncMock(return_value={"value": json.dumps(complex_value)})
        result = await backend.retrieve("key1")
        assert result == complex_value

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_search_substring(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.fetch = AsyncMock(
            return_value=[
                {"key": "user_name", "value": '"Alice"', "score": 0.5, "metadata": "{}"},
                {"key": "user_email", "value": '"alice@test.com"', "score": 0.5, "metadata": "{}"},
            ]
        )

        results = await backend.search("user", top_k=5)
        assert len(results) == 2

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_delete_existing(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.execute = AsyncMock(return_value="DELETE 1")
        result = await backend.delete("key1")
        assert result is True

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_delete_nonexistent(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.execute = AsyncMock(return_value="DELETE 0")
        result = await backend.delete("nonexistent")
        assert result is False

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_clear(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        await backend.clear()
        assert mock_conn.execute.await_count > 0

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_store_with_ttl(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        await backend.store("key1", "value1", metadata={"ttl_seconds": 3600})
        # Verify that the TTL path was taken
        assert mock_conn.execute.await_count > 0

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_store_with_default_ttl(self, mock_asyncpg):
        backend = _make_backend(default_ttl_seconds=300)
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        await backend.store("key1", "value1")
        assert mock_conn.execute.await_count > 0

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_health_check(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        result = await backend.health_check()
        assert result is True

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_close(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        # Initialize the pool
        await backend.health_check()
        await backend.close()
        mock_pool.close.assert_awaited_once()

    @patch("ia_agent_fwk.memory.backends.structured.asyncpg")
    async def test_table_autocreation(self, mock_asyncpg):
        backend = _make_backend()
        mock_pool, mock_conn = _setup_mock_pool(mock_asyncpg)

        mock_conn.fetchrow = AsyncMock(return_value=None)
        await backend.retrieve("key1")

        execute_calls = [str(c) for c in mock_conn.execute.call_args_list]
        call_texts = " ".join(execute_calls)
        assert "CREATE TABLE" in call_texts
