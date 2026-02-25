"""Tests for WeaviateRetriever with mocked backend."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.models import MemoryResult
from ia_agent_fwk.rag.exceptions import RetrievalError
from ia_agent_fwk.rag.retrieval.weaviate_retriever import WeaviateRetriever


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for testing."""

    DIMENSION = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10) / 10.0] * self.DIMENSION for t in texts]

    def dimension(self) -> int:
        return self.DIMENSION

    def max_tokens(self) -> int:
        return 8191


def _make_mock_backend() -> MagicMock:
    """Create a mock WeaviateMemoryBackend."""
    backend = MagicMock()
    backend.search = AsyncMock(return_value=[])
    return backend


@pytest.mark.unit
class TestWeaviateRetriever:
    async def test_retrieve_returns_results(self):
        mock_backend = _make_mock_backend()
        mock_backend.search = AsyncMock(
            return_value=[
                MemoryResult(key="doc:0", value="relevant content", score=0.95, metadata={"source": "a.txt"}),
                MemoryResult(key="doc:1", value="other content", score=0.80, metadata={"source": "b.txt"}),
            ]
        )

        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(backend=mock_backend, embedding_provider=provider)
        results = await retriever.retrieve("relevant", top_k=5)

        assert len(results) == 2
        assert results[0].chunk.content == "relevant content"
        assert results[0].score == 0.95
        assert results[0].chunk.source == "doc:0"
        assert results[0].metadata == {"source": "a.txt"}

    async def test_retrieve_empty_results(self):
        mock_backend = _make_mock_backend()
        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(backend=mock_backend, embedding_provider=provider)
        results = await retriever.retrieve("xyz_no_match", top_k=5)
        assert len(results) == 0

    async def test_retrieve_passes_top_k(self):
        mock_backend = _make_mock_backend()
        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(backend=mock_backend, embedding_provider=provider)
        await retriever.retrieve("query", top_k=3)
        mock_backend.search.assert_awaited_once_with("query", top_k=3)

    async def test_retrieve_with_metadata_filter(self):
        mock_backend = _make_mock_backend()
        provider = FakeEmbeddingProvider()
        metadata_filter: dict[str, Any] = {"category": "docs"}
        retriever = WeaviateRetriever(
            backend=mock_backend,
            embedding_provider=provider,
            metadata_filter=metadata_filter,
        )
        await retriever.retrieve("query", top_k=5)
        mock_backend.search.assert_awaited_once_with("query", top_k=5, metadata_filter=metadata_filter)

    async def test_retrieve_error_raises_retrieval_error(self):
        mock_backend = _make_mock_backend()
        mock_backend.search = AsyncMock(side_effect=RuntimeError("search failed"))
        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(backend=mock_backend, embedding_provider=provider)

        with pytest.raises(RetrievalError, match="Weaviate retrieval failed"):
            await retriever.retrieve("query")

    async def test_retrieve_handles_none_value(self):
        mock_backend = _make_mock_backend()
        mock_backend.search = AsyncMock(
            return_value=[
                MemoryResult(key="doc:0", value=None, score=0.9, metadata=None),
            ]
        )

        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(backend=mock_backend, embedding_provider=provider)
        results = await retriever.retrieve("query")

        assert len(results) == 1
        assert results[0].chunk.content == ""
        assert results[0].metadata == {}

    async def test_retrieve_preserves_chunk_metadata(self):
        mock_backend = _make_mock_backend()
        mock_backend.search = AsyncMock(
            return_value=[
                MemoryResult(
                    key="doc:0",
                    value="content here",
                    score=0.88,
                    metadata={"source": "file.pdf", "page": 3},
                ),
            ]
        )

        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(backend=mock_backend, embedding_provider=provider)
        results = await retriever.retrieve("query")

        assert len(results) == 1
        assert results[0].chunk.metadata == {"source": "file.pdf", "page": 3}
        assert results[0].chunk.source == "doc:0"

    async def test_retrieve_merges_filters_with_constructor_metadata_filter(self):
        """Per-call filters are merged with constructor-level metadata_filter."""
        mock_backend = _make_mock_backend()
        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(
            backend=mock_backend,
            embedding_provider=provider,
            metadata_filter={"category": "docs"},
        )
        await retriever.retrieve("query", top_k=5, filters={"author": "alice"})
        mock_backend.search.assert_awaited_once_with(
            "query",
            top_k=5,
            metadata_filter={"category": "docs", "author": "alice"},
        )

    async def test_retrieve_call_filters_override_constructor_on_conflict(self):
        """Call-time filters take precedence over constructor metadata_filter."""
        mock_backend = _make_mock_backend()
        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(
            backend=mock_backend,
            embedding_provider=provider,
            metadata_filter={"category": "docs"},
        )
        await retriever.retrieve("query", top_k=5, filters={"category": "reports"})
        mock_backend.search.assert_awaited_once_with(
            "query",
            top_k=5,
            metadata_filter={"category": "reports"},
        )

    async def test_retrieve_with_only_call_filters(self):
        """Filters work even when constructor metadata_filter is None."""
        mock_backend = _make_mock_backend()
        provider = FakeEmbeddingProvider()
        retriever = WeaviateRetriever(backend=mock_backend, embedding_provider=provider)
        await retriever.retrieve("query", top_k=5, filters={"tag": "important"})
        mock_backend.search.assert_awaited_once_with(
            "query",
            top_k=5,
            metadata_filter={"tag": "important"},
        )

    async def test_retrieve_accepts_filters_signature(self):
        """The retrieve method has the filters parameter in its signature."""
        import inspect

        sig = inspect.signature(WeaviateRetriever.retrieve)
        assert "filters" in sig.parameters
        param = sig.parameters["filters"]
        assert param.default is None
