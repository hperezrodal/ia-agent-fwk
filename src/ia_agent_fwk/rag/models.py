"""Pydantic v2 models for the RAG pipeline.

These models form the public data contract for document ingestion,
chunking, and retrieval.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A loaded document ready for chunking."""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    doc_type: str = ""


class Chunk(BaseModel):
    """A text chunk produced by the chunking stage."""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0
    source: str = ""
    embedding: list[float] | None = None
    start_char: int = 0
    end_char: int = 0
    token_count: int | None = None
    document_id: str = ""


class RetrievalResult(BaseModel):
    """A single retrieval result with chunk content and score."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Result of ingesting a single document into the RAG pipeline."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    chunk_count: int
    duration_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Result of a RAG query including timing information."""

    model_config = ConfigDict(frozen=True)

    results: list[RetrievalResult]
    context: str
    query_embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    total_ms: float = 0.0
