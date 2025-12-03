"""Tests for RAG exception hierarchy."""

from __future__ import annotations

import pytest

from ia_agent_fwk.rag.exceptions import (
    ChunkingError,
    DocumentLoadError,
    EmbeddingError,
    RAGError,
    RetrievalError,
)


@pytest.mark.unit
class TestRAGExceptions:
    def test_rag_error_is_exception(self):
        assert issubclass(RAGError, Exception)

    def test_document_load_error_is_rag_error(self):
        assert issubclass(DocumentLoadError, RAGError)

    def test_chunking_error_is_rag_error(self):
        assert issubclass(ChunkingError, RAGError)

    def test_embedding_error_is_rag_error(self):
        assert issubclass(EmbeddingError, RAGError)

    def test_retrieval_error_is_rag_error(self):
        assert issubclass(RetrievalError, RAGError)

    def test_rag_error_message(self):
        err = RAGError("something broke")
        assert str(err) == "something broke"

    def test_document_load_error_message(self):
        err = DocumentLoadError("file not found")
        assert str(err) == "file not found"

    def test_can_catch_rag_error_for_subclasses(self):
        msg = "test"
        with pytest.raises(RAGError):
            raise DocumentLoadError(msg)

        with pytest.raises(RAGError):
            raise ChunkingError(msg)

        with pytest.raises(RAGError):
            raise EmbeddingError(msg)

        with pytest.raises(RAGError):
            raise RetrievalError(msg)
