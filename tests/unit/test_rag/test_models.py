"""Tests for RAG data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ia_agent_fwk.rag.models import Chunk, Document, IngestionResult, QueryResult, RetrievalResult


@pytest.mark.unit
class TestDocument:
    def test_create_document(self):
        doc = Document(content="hello", source="test.txt", doc_type="text")
        assert doc.content == "hello"
        assert doc.source == "test.txt"
        assert doc.doc_type == "text"
        assert doc.metadata == {}

    def test_document_with_metadata(self):
        doc = Document(content="hi", metadata={"key": "val"}, source="a.txt", doc_type="text")
        assert doc.metadata == {"key": "val"}

    def test_document_is_frozen(self):
        doc = Document(content="hi")
        with pytest.raises(ValidationError):
            doc.content = "bye"  # type: ignore[misc]


@pytest.mark.unit
class TestChunk:
    def test_create_chunk(self):
        chunk = Chunk(content="piece", chunk_index=0, source="test.txt")
        assert chunk.content == "piece"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None

    def test_chunk_with_embedding(self):
        chunk = Chunk(content="piece", chunk_index=0, embedding=[0.1, 0.2, 0.3])
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_chunk_is_frozen(self):
        chunk = Chunk(content="piece", chunk_index=0)
        with pytest.raises(ValidationError):
            chunk.content = "new"  # type: ignore[misc]

    def test_chunk_start_end_char_defaults(self):
        chunk = Chunk(content="piece", chunk_index=0)
        assert chunk.start_char == 0
        assert chunk.end_char == 0

    def test_chunk_start_end_char_explicit(self):
        chunk = Chunk(content="hello", chunk_index=0, start_char=10, end_char=15)
        assert chunk.start_char == 10
        assert chunk.end_char == 15

    def test_chunk_token_count_default(self):
        chunk = Chunk(content="piece", chunk_index=0)
        assert chunk.token_count is None

    def test_chunk_token_count_explicit(self):
        chunk = Chunk(content="piece", chunk_index=0, token_count=42)
        assert chunk.token_count == 42

    def test_chunk_document_id_default(self):
        chunk = Chunk(content="piece", chunk_index=0)
        assert chunk.document_id == ""

    def test_chunk_document_id_explicit(self):
        chunk = Chunk(content="piece", chunk_index=0, document_id="report.pdf")
        assert chunk.document_id == "report.pdf"

    def test_chunk_all_new_fields(self):
        chunk = Chunk(
            content="hello world",
            chunk_index=2,
            source="doc.txt",
            start_char=100,
            end_char=111,
            token_count=3,
            document_id="doc.txt",
        )
        assert chunk.start_char == 100
        assert chunk.end_char == 111
        assert chunk.token_count == 3
        assert chunk.document_id == "doc.txt"

    def test_retrieval_result_exposes_document_id(self):
        """F-009: document_id is accessible via result.chunk.document_id."""
        chunk = Chunk(content="data", chunk_index=0, document_id="report.pdf")
        result = RetrievalResult(chunk=chunk, score=0.9)
        assert result.chunk.document_id == "report.pdf"


@pytest.mark.unit
class TestRetrievalResult:
    def test_create_retrieval_result(self):
        chunk = Chunk(content="data", chunk_index=0, source="doc.txt")
        result = RetrievalResult(chunk=chunk, score=0.95)
        assert result.chunk.content == "data"
        assert result.score == 0.95
        assert result.metadata == {}

    def test_retrieval_result_with_metadata(self):
        chunk = Chunk(content="data", chunk_index=1)
        result = RetrievalResult(chunk=chunk, score=0.8, metadata={"doc_id": "abc"})
        assert result.metadata == {"doc_id": "abc"}

    def test_retrieval_result_is_frozen(self):
        chunk = Chunk(content="data", chunk_index=0)
        result = RetrievalResult(chunk=chunk, score=0.5)
        with pytest.raises(ValidationError):
            result.score = 0.9  # type: ignore[misc]


@pytest.mark.unit
class TestIngestionResult:
    def test_create_ingestion_result(self):
        result = IngestionResult(document_id="doc.txt", chunk_count=5, duration_ms=123.4)
        assert result.document_id == "doc.txt"
        assert result.chunk_count == 5
        assert result.duration_ms == 123.4
        assert result.metadata == {}

    def test_ingestion_result_with_metadata(self):
        result = IngestionResult(
            document_id="doc.pdf",
            chunk_count=10,
            duration_ms=200.0,
            metadata={"source": "upload"},
        )
        assert result.metadata == {"source": "upload"}

    def test_ingestion_result_is_frozen(self):
        result = IngestionResult(document_id="doc.txt", chunk_count=3, duration_ms=50.0)
        with pytest.raises(ValidationError):
            result.chunk_count = 10  # type: ignore[misc]


@pytest.mark.unit
class TestQueryResult:
    def test_create_query_result(self):
        chunk = Chunk(content="data", chunk_index=0, source="doc.txt")
        rr = RetrievalResult(chunk=chunk, score=0.9)
        qr = QueryResult(results=[rr], context="assembled context")
        assert len(qr.results) == 1
        assert qr.context == "assembled context"
        assert qr.query_embedding_ms == 0.0
        assert qr.retrieval_ms == 0.0
        assert qr.total_ms == 0.0

    def test_query_result_with_timing(self):
        qr = QueryResult(
            results=[],
            context="",
            query_embedding_ms=10.5,
            retrieval_ms=25.3,
            total_ms=35.8,
        )
        assert qr.query_embedding_ms == 10.5
        assert qr.retrieval_ms == 25.3
        assert qr.total_ms == 35.8

    def test_query_result_is_frozen(self):
        qr = QueryResult(results=[], context="ctx")
        with pytest.raises(ValidationError):
            qr.context = "new"  # type: ignore[misc]

    def test_query_result_empty_results(self):
        qr = QueryResult(results=[], context="")
        assert qr.results == []
        assert qr.context == ""
