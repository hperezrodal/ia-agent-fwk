"""Tests for the RAG pipeline orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ia_agent_fwk.config.settings import RAGSettings
from ia_agent_fwk.rag.chunkers.fixed import FixedSizeChunker
from ia_agent_fwk.rag.factory import RAGPipelineFactory
from ia_agent_fwk.rag.loaders.registry import LoaderRegistry
from ia_agent_fwk.rag.models import IngestionResult, QueryResult
from ia_agent_fwk.rag.pipeline import RAGPipeline
from ia_agent_fwk.rag.retrieval.vector import VectorRetriever

from .conftest import FakeEmbeddingProvider

if TYPE_CHECKING:
    from pathlib import Path

    from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend


def _build_pipeline(backend: InMemoryBackend) -> RAGPipeline:
    """Build a pipeline with test components."""
    provider = FakeEmbeddingProvider()
    return RAGPipeline(
        loader_registry=LoaderRegistry(),
        chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=10),
        embedding_provider=provider,
        memory_backend=backend,
        retriever=VectorRetriever(backend=backend, embedding_provider=provider),
    )


@pytest.mark.unit
class TestRAGPipeline:
    async def test_ingest_returns_ingestion_result(self, tmp_text_file: Path, mock_memory_backend: InMemoryBackend):
        pipeline = _build_pipeline(mock_memory_backend)
        result = await pipeline.ingest(tmp_text_file)
        assert isinstance(result, IngestionResult)
        assert result.chunk_count >= 1
        assert result.document_id == tmp_text_file.name
        assert result.duration_ms >= 0.0

    async def test_ingest_batch_returns_list_of_ingestion_results(
        self, tmp_text_file: Path, tmp_md_file: Path, mock_memory_backend: InMemoryBackend
    ):
        pipeline = _build_pipeline(mock_memory_backend)
        results = await pipeline.ingest_batch([tmp_text_file, tmp_md_file])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, IngestionResult) for r in results)
        total = sum(r.chunk_count for r in results)
        assert total >= 2

    async def test_query_returns_query_result(self, tmp_text_file: Path, mock_memory_backend: InMemoryBackend):
        pipeline = _build_pipeline(mock_memory_backend)
        await pipeline.ingest(tmp_text_file)
        qr = await pipeline.query("sample", top_k=3)
        assert isinstance(qr, QueryResult)
        assert isinstance(qr.results, list)
        # With InMemoryBackend substring search, "sample" should match stored chunks
        assert len(qr.results) >= 1
        assert isinstance(qr.context, str)
        assert qr.retrieval_ms >= 0.0
        assert qr.total_ms >= qr.retrieval_ms

    async def test_query_with_context(self, tmp_text_file: Path, mock_memory_backend: InMemoryBackend):
        pipeline = _build_pipeline(mock_memory_backend)
        await pipeline.ingest(tmp_text_file)
        ctx = await pipeline.query_with_context("sample", top_k=3)
        assert isinstance(ctx, str)
        assert len(ctx) > 0


@pytest.mark.unit
class TestRAGPipelineErrorPaths:
    """Test error handling at each pipeline stage (F-018)."""

    async def test_ingest_wraps_load_error(self, mock_memory_backend: InMemoryBackend):
        """Non-DocumentLoadError during loading is wrapped in DocumentLoadError."""
        from unittest.mock import AsyncMock

        from ia_agent_fwk.rag.exceptions import DocumentLoadError

        pipeline = _build_pipeline(mock_memory_backend)
        pipeline._loader_registry.load = AsyncMock(side_effect=RuntimeError("disk error"))  # type: ignore[method-assign]

        with pytest.raises(DocumentLoadError, match="disk error"):
            await pipeline.ingest("/fake/file.txt")

    async def test_ingest_wraps_embedding_error(self, tmp_text_file: Path, mock_memory_backend: InMemoryBackend):
        """Embedding failure is wrapped in EmbeddingError."""
        from ia_agent_fwk.rag.exceptions import EmbeddingError

        pipeline = _build_pipeline(mock_memory_backend)
        pipeline._embedding_provider.embed = lambda _texts: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("API timeout")
        )

        with pytest.raises(EmbeddingError, match="API timeout"):
            await pipeline.ingest(tmp_text_file)

    async def test_query_wraps_retrieval_error(self, tmp_text_file: Path, mock_memory_backend: InMemoryBackend):
        """Non-RetrievalError during query is wrapped in RetrievalError."""
        from unittest.mock import AsyncMock

        from ia_agent_fwk.rag.exceptions import RetrievalError

        pipeline = _build_pipeline(mock_memory_backend)
        await pipeline.ingest(tmp_text_file)
        pipeline._retriever.retrieve = AsyncMock(side_effect=RuntimeError("backend down"))  # type: ignore[method-assign]

        with pytest.raises(RetrievalError, match="backend down"):
            await pipeline.query("test")


@pytest.mark.unit
class TestRAGPipelineFactory:
    def test_pipeline_factory_creates_pipeline(self, mock_memory_backend: InMemoryBackend):
        settings = RAGSettings()
        provider = FakeEmbeddingProvider()
        pipeline = RAGPipelineFactory.create(settings, mock_memory_backend, provider)
        assert isinstance(pipeline, RAGPipeline)
