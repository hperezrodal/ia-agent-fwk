"""RAG pipeline factory for configuration-driven construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ia_agent_fwk.rag.chunkers.factory import ChunkerFactory
from ia_agent_fwk.rag.loaders.registry import LoaderRegistry
from ia_agent_fwk.rag.pipeline import RAGPipeline
from ia_agent_fwk.rag.retrieval.vector import VectorRetriever

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import RAGSettings
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider


class RAGPipelineFactory:
    """Create a ``RAGPipeline`` from settings objects."""

    @staticmethod
    def create(
        settings: RAGSettings,
        memory_backend: MemoryBackend,
        embedding_provider: EmbeddingProvider,
    ) -> RAGPipeline:
        """Instantiate and return a fully-wired ``RAGPipeline``.

        Parameters
        ----------
        settings:
            RAG configuration section.
        memory_backend:
            The vector memory backend for chunk storage / search.
        embedding_provider:
            The embedding provider for vector generation.

        """
        loader_registry = LoaderRegistry()
        chunker = ChunkerFactory.create(
            strategy=settings.chunking.strategy,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
        retriever = VectorRetriever(
            backend=memory_backend,
            embedding_provider=embedding_provider,
        )
        return RAGPipeline(
            loader_registry=loader_registry,
            chunker=chunker,
            embedding_provider=embedding_provider,
            memory_backend=memory_backend,
            retriever=retriever,
        )
