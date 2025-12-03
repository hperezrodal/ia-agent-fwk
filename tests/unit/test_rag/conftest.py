"""Shared fixtures for RAG unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend
from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.rag.models import Chunk, Document

if TYPE_CHECKING:
    from pathlib import Path


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for testing."""

    DIMENSION = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return fixed-dimension vectors based on text length."""
        return [[float(len(t) % 10) / 10.0] * self.DIMENSION for t in texts]

    def dimension(self) -> int:
        return self.DIMENSION

    def max_tokens(self) -> int:
        return 8191


@pytest.fixture
def sample_document() -> Document:
    return Document(
        content="Hello world. This is a test document with some content for chunking.",
        metadata={"filename": "test.txt"},
        source="test.txt",
        doc_type="text",
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(content="chunk one", metadata={}, chunk_index=0, source="test.txt"),
        Chunk(content="chunk two", metadata={}, chunk_index=1, source="test.txt"),
        Chunk(content="chunk three", metadata={}, chunk_index=2, source="test.txt"),
    ]


@pytest.fixture
def mock_embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def mock_memory_backend() -> InMemoryBackend:
    return InMemoryBackend(max_items=1000)


@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("This is sample text content for testing.", encoding="utf-8")
    return f


@pytest.fixture
def tmp_md_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.md"
    f.write_text("# Heading\n\nSome **bold** and *italic* text.", encoding="utf-8")
    return f


@pytest.fixture
def tmp_html_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.html"
    f.write_text(
        "<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p></body></html>",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def tmp_docx_file(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx", reason="python-docx not installed")

    f = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_paragraph("First paragraph of the document.")
    doc.add_paragraph("Second paragraph with more content.")
    doc.save(str(f))
    return f
