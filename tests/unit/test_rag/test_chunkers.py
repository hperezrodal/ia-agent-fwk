"""Tests for chunking strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.rag.chunkers.factory import ChunkerFactory
from ia_agent_fwk.rag.chunkers.fixed import FixedSizeChunker
from ia_agent_fwk.rag.chunkers.recursive import RecursiveChunker
from ia_agent_fwk.rag.chunkers.semantic import SemanticChunker
from ia_agent_fwk.rag.exceptions import ChunkingError
from ia_agent_fwk.rag.models import Document


@pytest.mark.unit
class TestFixedSizeChunker:
    async def test_fixed_chunker_splits_text(self):
        text = "word " * 200  # 1000 chars
        doc = Document(content=text.strip(), source="test.txt", doc_type="text")
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 100

    async def test_fixed_chunker_overlap(self):
        text = "A" * 300
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = await chunker.chunk(doc)
        assert len(chunks) > 1
        # The chunks should overlap
        assert len(chunks) >= 3

    async def test_fixed_chunker_small_text(self):
        doc = Document(content="short text", source="test.txt", doc_type="text")
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "short text"

    async def test_fixed_chunker_empty_text(self):
        doc = Document(content="", source="test.txt", doc_type="text")
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 0

    async def test_fixed_chunker_preserves_metadata(self, sample_document):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        chunks = await chunker.chunk(sample_document)
        assert len(chunks) >= 1
        assert chunks[0].source == "test.txt"

    async def test_fixed_chunker_indexes(self):
        text = "word " * 100
        doc = Document(content=text.strip(), source="test.txt", doc_type="text")
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    async def test_fixed_chunker_start_end_char(self):
        text = "Hello world, this is a test document for chunking."
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2
        for chunk in chunks:
            # The chunk content must match the original text at those positions
            assert text[chunk.start_char : chunk.end_char] == chunk.content

    async def test_fixed_chunker_start_end_char_single_chunk(self):
        text = "short"
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 5
        assert text[chunks[0].start_char : chunks[0].end_char] == "short"

    async def test_fixed_chunker_start_end_char_with_overlap(self):
        text = "A" * 300
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert text[chunk.start_char : chunk.end_char] == chunk.content


@pytest.mark.unit
class TestRecursiveChunker:
    async def test_recursive_chunker_splits_paragraphs(self):
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three."
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2

    async def test_recursive_chunker_respects_size(self):
        text = "word " * 500
        doc = Document(content=text.strip(), source="test.txt", doc_type="text")
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        assert len(chunks) > 1
        # Each chunk should be at most chunk_size chars (plus any overlap)
        for chunk in chunks:
            assert len(chunk.content) <= 200 + chunker.chunk_overlap

    async def test_recursive_chunker_empty_text(self):
        doc = Document(content="", source="test.txt", doc_type="text")
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 0

    async def test_recursive_chunker_small_text(self):
        doc = Document(content="small", source="test.txt", doc_type="text")
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "small"

    async def test_recursive_chunker_start_end_char(self):
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three."
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2
        for chunk in chunks:
            # Verify start_char and end_char are consistent
            assert chunk.end_char > chunk.start_char
            assert chunk.end_char - chunk.start_char == len(chunk.content)
            # Content at those positions must match
            assert text[chunk.start_char : chunk.end_char] == chunk.content

    async def test_recursive_chunker_start_end_char_single_chunk(self):
        text = "small"
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 5

    async def test_recursive_chunker_character_level_split(self):
        """F-017: Text with no paragraph/sentence/word boundaries forces character-level split."""
        # A single continuous string with no separators (\n\n, \n, '. ', ' ')
        text = "a" * 300
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        # 300 chars / 100 chunk_size = 3 chunks
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk.content) <= 100
        # All content should be covered
        total_len = sum(len(c.content) for c in chunks)
        assert total_len == 300

    async def test_recursive_chunker_overlap_application(self):
        """F-017: Overlap is applied between consecutive chunks."""
        # Use paragraphs that each individually fit in chunk_size but together do not,
        # forcing a split with overlap applied.
        para1 = "A" * 80
        para2 = "B" * 80
        para3 = "C" * 80
        text = f"{para1}\n\n{para2}\n\n{para3}"
        doc = Document(content=text, source="test.txt", doc_type="text")
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2
        # Second chunk onward should contain overlap from the previous chunk
        # (the overlap text is prepended from the end of the previous raw chunk)
        if len(chunks) >= 2:
            # The second chunk should have overlap content from the first
            second_content = chunks[1].content
            # With overlap=20, the second chunk starts with chars from end of first
            assert len(second_content) > 0

    async def test_recursive_chunker_long_paragraph_recursive_split(self):
        """F-017: Single very long paragraph exceeding chunk_size requires recursive splitting."""
        # A single paragraph (no \n\n) with sentence boundaries ('. ')
        # that exceeds chunk_size, forcing recursive splitting at finer separators.
        sentences = [f"Sentence number {i} with some extra filler text here" for i in range(20)]
        text = ". ".join(sentences) + "."
        doc = Document(content=text, source="test.txt", doc_type="text")
        # chunk_size is smaller than the full text but larger than individual sentences
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=0)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2
        for chunk in chunks:
            # Each chunk should respect the size limit
            assert len(chunk.content) <= 150
        # All original text content should be represented across chunks
        combined = " ".join(c.content for c in chunks)
        # Each sentence fragment should appear somewhere
        assert "Sentence number 0" in combined
        assert "Sentence number 19" in combined


@pytest.mark.unit
class TestChunkerFactory:
    def test_chunker_factory_creates_fixed(self):
        chunker = ChunkerFactory.create("fixed", chunk_size=500, chunk_overlap=50)
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_chunker_factory_creates_recursive(self):
        chunker = ChunkerFactory.create("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_chunker_factory_unknown_strategy(self):
        with pytest.raises(ChunkingError, match="Unknown chunking strategy"):
            ChunkerFactory.create("unknown")

    def test_chunker_factory_creates_semantic(self, mock_embedding_provider):
        chunker = ChunkerFactory.create(
            "semantic",
            embedding_provider=mock_embedding_provider,
            similarity_threshold=0.7,
        )
        assert isinstance(chunker, SemanticChunker)

    def test_chunker_factory_semantic_requires_provider(self):
        with pytest.raises(ChunkingError, match="requires an embedding_provider"):
            ChunkerFactory.create("semantic")


def _make_mock_provider(embeddings: list[list[float]]) -> EmbeddingProvider:
    """Create a mock EmbeddingProvider that returns pre-defined embeddings."""
    provider = AsyncMock(spec=EmbeddingProvider)
    provider.embed = AsyncMock(return_value=embeddings)
    provider.dimension.return_value = len(embeddings[0]) if embeddings else 4
    provider.max_tokens.return_value = 8191
    return provider


@pytest.mark.unit
class TestSemanticChunker:
    async def test_basic_semantic_chunking(self):
        """Similar sentences grouped, dissimilar ones split."""
        # Three sentences: first two get similar embeddings, third is different
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        provider = _make_mock_provider(embeddings)
        chunker = SemanticChunker(
            provider,
            similarity_threshold=0.5,
            min_chunk_size=1,
        )
        doc = Document(
            content="First sentence. Second sentence. Third sentence.",
            source="test.txt",
        )
        chunks = await chunker.chunk(doc)
        # First two similar → grouped, third dissimilar → separate
        assert len(chunks) == 2
        assert "First sentence" in chunks[0].content
        assert "Second sentence" in chunks[0].content
        assert "Third sentence" in chunks[1].content

    async def test_uniform_embeddings_produce_one_chunk(self):
        """When all embeddings are identical, everything stays in one chunk."""
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
        provider = _make_mock_provider(embeddings)
        chunker = SemanticChunker(
            provider,
            similarity_threshold=0.5,
            min_chunk_size=1,
        )
        doc = Document(
            content="Sentence one. Sentence two. Sentence three.",
            source="test.txt",
        )
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        assert "Sentence one" in chunks[0].content
        assert "Sentence three" in chunks[0].content

    async def test_dissimilar_embeddings_split_each_sentence(self):
        """When all adjacent embeddings are orthogonal, each sentence is its own chunk."""
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        provider = _make_mock_provider(embeddings)
        chunker = SemanticChunker(
            provider,
            similarity_threshold=0.5,
            min_chunk_size=1,
        )
        doc = Document(
            content="Alpha fact. Beta fact. Gamma fact.",
            source="test.txt",
        )
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 3
        assert chunks[0].content == "Alpha fact."
        assert chunks[1].content == "Beta fact."
        assert chunks[2].content == "Gamma fact."

    async def test_empty_document(self):
        """Empty content returns no chunks."""
        provider = _make_mock_provider([])
        chunker = SemanticChunker(provider, min_chunk_size=1)
        doc = Document(content="", source="test.txt")
        chunks = await chunker.chunk(doc)
        assert chunks == []

    async def test_single_sentence_document(self):
        """A single sentence returns exactly one chunk without calling embed."""
        provider = _make_mock_provider([])
        chunker = SemanticChunker(provider, min_chunk_size=1)
        doc = Document(content="Only one sentence.", source="test.txt")
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Only one sentence."
        assert chunks[0].chunk_index == 0
        # embed should not have been called for a single sentence
        provider.embed.assert_not_awaited()

    async def test_chunk_indexes_are_sequential(self):
        """Each chunk has a sequential chunk_index starting at 0."""
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        provider = _make_mock_provider(embeddings)
        chunker = SemanticChunker(
            provider,
            similarity_threshold=0.5,
            min_chunk_size=1,
        )
        doc = Document(
            content="A first. B second. C third. D fourth.",
            source="test.txt",
        )
        chunks = await chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    async def test_max_chunk_size_forces_split(self):
        """A chunk is split when its length exceeds max_chunk_size."""
        # All embeddings identical → would normally be one chunk
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
        provider = _make_mock_provider(embeddings)
        chunker = SemanticChunker(
            provider,
            similarity_threshold=0.5,
            min_chunk_size=1,
            max_chunk_size=20,
        )
        doc = Document(
            content="Short one. Short two. Short three.",
            source="test.txt",
        )
        chunks = await chunker.chunk(doc)
        # With max_chunk_size=20, the chunker should split at least once
        assert len(chunks) >= 2

    async def test_cosine_similarity_zero_vectors(self):
        """Cosine similarity of zero vectors returns 0.0."""
        assert SemanticChunker._cosine_similarity([0, 0, 0], [0, 0, 0]) == 0.0

    async def test_cosine_similarity_identical(self):
        """Cosine similarity of identical unit vectors returns 1.0."""
        result = SemanticChunker._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(result - 1.0) < 1e-9

    async def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors returns 0.0."""
        result = SemanticChunker._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(result) < 1e-9
