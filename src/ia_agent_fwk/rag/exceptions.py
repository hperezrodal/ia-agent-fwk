"""RAG pipeline exception hierarchy.

All RAG-specific exceptions inherit from ``RAGError`` which itself
inherits from the built-in ``Exception``.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""


class DocumentLoadError(RAGError):
    """Raised when loading a document file fails."""


class ChunkingError(RAGError):
    """Raised when chunking a document fails."""


class EmbeddingError(RAGError):
    """Raised when generating embeddings fails."""


class RetrievalError(RAGError):
    """Raised when retrieval / vector search fails."""
