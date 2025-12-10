"""Tests for RAGStore adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ia_agent_fwk.rag.exceptions import EmbeddingError
from ia_agent_fwk.rag.models import Chunk
from ia_agent_fwk.rag.store import RAGStore


def _make_mock_backend() -> AsyncMock:
    """Create a mock MemoryBackend."""
    backend = AsyncMock()
    backend.store = AsyncMock()
    backend.search = AsyncMock(return_value=[])
    backend.delete = AsyncMock(return_value=True)
    return backend


def _make_mock_embedding_provider(dimension: int = 4) -> AsyncMock:
    """Create a mock EmbeddingProvider that returns fixed embeddings."""
    provider = AsyncMock()
    provider.embed = AsyncMock(
        side_effect=lambda texts: [[0.1] * dimension for _ in texts],
    )
    return provider


def _make_chunks(count: int = 3) -> list[Chunk]:
    """Create a list of test chunks."""
    return [
        Chunk(
            content=f"chunk content {i}",
            metadata={"extra": f"val_{i}"},
            chunk_index=i,
            source="test.txt",
        )
        for i in range(count)
    ]


@pytest.mark.unit
class TestRAGStoreStoreChunks:
    async def test_store_chunks_stores_all_with_metadata(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        store = RAGStore(backend, embedding, collection="test_col")
        chunks = _make_chunks(3)

        result = await store.store_chunks(chunks, document_id="doc-1")

        assert result == 3
        assert backend.store.call_count == 3

        # Verify metadata for first chunk
        first_call = backend.store.call_args_list[0]
        assert first_call.kwargs["key"] == "doc-1:0"
        assert first_call.kwargs["value"] == "chunk content 0"
        meta = first_call.kwargs["metadata"]
        assert meta["document_id"] == "doc-1"
        assert meta["chunk_index"] == 0
        assert meta["source"] == "test.txt"
        assert meta["collection"] == "test_col"
        assert meta["extra"] == "val_0"
        assert meta["embedding"] == [0.1, 0.1, 0.1, 0.1]

    async def test_store_chunks_with_empty_list_returns_zero(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        store = RAGStore(backend, embedding)

        result = await store.store_chunks([], document_id="doc-1")

        assert result == 0
        embedding.embed.assert_not_called()
        backend.store.assert_not_called()

    async def test_store_chunks_calls_embed_with_all_texts(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        store = RAGStore(backend, embedding)
        chunks = _make_chunks(2)

        await store.store_chunks(chunks, document_id="doc-1")

        embedding.embed.assert_called_once_with(
            ["chunk content 0", "chunk content 1"],
        )

    async def test_store_chunks_raises_embedding_error_on_failure(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        embedding.embed = AsyncMock(side_effect=RuntimeError("API down"))
        store = RAGStore(backend, embedding)
        chunks = _make_chunks(1)

        with pytest.raises(EmbeddingError, match="Failed to embed 1 chunks"):
            await store.store_chunks(chunks, document_id="doc-1")

    async def test_store_chunks_uses_correct_keys(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        store = RAGStore(backend, embedding)
        chunks = _make_chunks(3)

        await store.store_chunks(chunks, document_id="my-doc")

        keys = [call.kwargs["key"] for call in backend.store.call_args_list]
        assert keys == ["my-doc:0", "my-doc:1", "my-doc:2"]


@pytest.mark.unit
class TestRAGStoreSearch:
    async def test_search_delegates_to_backend(self) -> None:
        backend = _make_mock_backend()
        sentinel = MagicMock()
        backend.search = AsyncMock(return_value=[sentinel])
        embedding = _make_mock_embedding_provider()
        store = RAGStore(backend, embedding)

        results = await store.search("test query", top_k=3)

        assert results == [sentinel]
        backend.search.assert_called_once_with(
            query="test query",
            top_k=3,
            metadata_filter=None,
        )

    async def test_search_passes_metadata_filter(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        store = RAGStore(backend, embedding)
        filt: dict[str, Any] = {"document_id": "doc-1"}

        await store.search("query", top_k=5, metadata_filter=filt)

        backend.search.assert_called_once_with(
            query="query",
            top_k=5,
            metadata_filter=filt,
        )

    async def test_search_returns_empty_list(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        store = RAGStore(backend, embedding)

        results = await store.search("nothing")

        assert results == []


@pytest.mark.unit
class TestRAGStoreDeleteDocument:
    async def test_delete_document_deletes_found_chunks(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()

        # Simulate search returning 2 results
        result_1 = MagicMock()
        result_1.key = "doc-1:0"
        result_2 = MagicMock()
        result_2.key = "doc-1:1"
        backend.search = AsyncMock(return_value=[result_1, result_2])

        store = RAGStore(backend, embedding)
        deleted = await store.delete_document("doc-1")

        assert deleted == 2
        assert backend.delete.call_count == 2
        backend.delete.assert_any_call("doc-1:0")
        backend.delete.assert_any_call("doc-1:1")

    async def test_delete_document_no_chunks(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        backend.search = AsyncMock(return_value=[])

        store = RAGStore(backend, embedding)
        deleted = await store.delete_document("doc-1")

        assert deleted == 0
        backend.delete.assert_not_called()

    async def test_delete_document_handles_delete_failure(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()

        result_1 = MagicMock()
        result_1.key = "doc-1:0"
        result_2 = MagicMock()
        result_2.key = "doc-1:1"
        backend.search = AsyncMock(return_value=[result_1, result_2])
        # First delete succeeds, second raises
        backend.delete = AsyncMock(side_effect=[True, RuntimeError("fail")])

        store = RAGStore(backend, embedding)
        deleted = await store.delete_document("doc-1")

        # Only the first one counted as deleted
        assert deleted == 1

    async def test_delete_document_searches_with_metadata_filter(self) -> None:
        backend = _make_mock_backend()
        embedding = _make_mock_embedding_provider()
        store = RAGStore(backend, embedding)

        await store.delete_document("my-doc")

        backend.search.assert_called_once_with(
            query="",
            top_k=1000,
            metadata_filter={"document_id": "my-doc"},
        )
