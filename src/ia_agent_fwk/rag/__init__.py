"""RAG (Retrieval-Augmented Generation) pipeline module.

Public API re-exports for convenient imports::

    from ia_agent_fwk.rag import RAGPipeline, Document, Chunk, RetrievalResult
"""

from __future__ import annotations

from ia_agent_fwk.rag.chunkers.base import Chunker
from ia_agent_fwk.rag.chunkers.factory import ChunkerFactory
from ia_agent_fwk.rag.chunkers.fixed import FixedSizeChunker
from ia_agent_fwk.rag.chunkers.recursive import RecursiveChunker
from ia_agent_fwk.rag.exceptions import (
    ChunkingError,
    DocumentLoadError,
    EmbeddingError,
    RAGError,
    RetrievalError,
)
from ia_agent_fwk.rag.factory import RAGPipelineFactory
from ia_agent_fwk.rag.loaders.base import FileLoader
from ia_agent_fwk.rag.loaders.registry import LoaderRegistry
from ia_agent_fwk.rag.models import Chunk, Document, IngestionResult, QueryResult, RetrievalResult
from ia_agent_fwk.rag.pipeline import RAGPipeline
from ia_agent_fwk.rag.retrieval.base import Retriever
from ia_agent_fwk.rag.retrieval.context import ContextAssembler
from ia_agent_fwk.rag.retrieval.vector import VectorRetriever
from ia_agent_fwk.rag.retrieval.weaviate_retriever import WeaviateRetriever
from ia_agent_fwk.rag.store import RAGStore

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkerFactory",
    "ChunkingError",
    "ContextAssembler",
    "Document",
    "DocumentLoadError",
    "EmbeddingError",
    "FileLoader",
    "FixedSizeChunker",
    "IngestionResult",
    "LoaderRegistry",
    "QueryResult",
    "RAGError",
    "RAGPipeline",
    "RAGPipelineFactory",
    "RAGStore",
    "RecursiveChunker",
    "RetrievalError",
    "RetrievalResult",
    "Retriever",
    "VectorRetriever",
    "WeaviateRetriever",
]
