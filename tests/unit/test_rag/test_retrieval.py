"""Tests for retrieval strategies and context assembler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from ia_agent_fwk.rag.exceptions import RetrievalError
from ia_agent_fwk.rag.models import Chunk, RetrievalResult
from ia_agent_fwk.rag.retrieval.context import ContextAssembler
from ia_agent_fwk.rag.retrieval.factory import RetrieverFactory
from ia_agent_fwk.rag.retrieval.filtered import FilteredRetriever
from ia_agent_fwk.rag.retrieval.mmr import MMRRetriever
from ia_agent_fwk.rag.retrieval.vector import VectorRetriever

from .conftest import FakeEmbeddingProvider

if TYPE_CHECKING:
    from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend


@pytest.mark.unit
class TestVectorRetriever:
    async def test_vector_retriever_returns_results(self, mock_memory_backend: InMemoryBackend):
        # Store some data first
        await mock_memory_backend.store("doc:0", "relevant content", metadata={"source": "a.txt"})
        await mock_memory_backend.store("doc:1", "other content", metadata={"source": "b.txt"})

        provider = FakeEmbeddingProvider()
        retriever = VectorRetriever(backend=mock_memory_backend, embedding_provider=provider)
        results = await retriever.retrieve("relevant", top_k=5)
        # InMemoryBackend uses substring search, so it should find "relevant content"
        assert len(results) >= 1

    async def test_vector_retriever_empty_results(self, mock_memory_backend: InMemoryBackend):
        provider = FakeEmbeddingProvider()
        retriever = VectorRetriever(backend=mock_memory_backend, embedding_provider=provider)
        results = await retriever.retrieve("xyz_no_match", top_k=5)
        assert len(results) == 0

    async def test_vector_retriever_embeds_query(self, mock_memory_backend: InMemoryBackend):
        """The retriever delegates to backend.search() with the query string."""
        await mock_memory_backend.store("key1", "hello world", metadata={})
        provider = FakeEmbeddingProvider()
        retriever = VectorRetriever(backend=mock_memory_backend, embedding_provider=provider)
        results = await retriever.retrieve("hello", top_k=2)
        # Should find "hello world" via substring search
        assert any("hello" in r.chunk.content for r in results)

    async def test_vector_retriever_filters_none(self, mock_memory_backend: InMemoryBackend):
        """Passing filters=None (the default) works without error."""
        await mock_memory_backend.store("doc:0", "content", metadata={"source": "a.txt"})
        provider = FakeEmbeddingProvider()
        retriever = VectorRetriever(backend=mock_memory_backend, embedding_provider=provider)
        results = await retriever.retrieve("content", top_k=5, filters=None)
        assert len(results) >= 1

    async def test_vector_retriever_filters_fallback(self, mock_memory_backend: InMemoryBackend):
        """When backend doesn't support metadata_filter, retriever falls back gracefully."""
        await mock_memory_backend.store("doc:0", "content here", metadata={"source": "a.txt"})
        provider = FakeEmbeddingProvider()
        retriever = VectorRetriever(backend=mock_memory_backend, embedding_provider=provider)
        # InMemoryBackend.search() doesn't accept metadata_filter,
        # so the retriever should catch TypeError and fall back
        results = await retriever.retrieve("content", top_k=5, filters={"source": "a.txt"})
        assert len(results) >= 1

    async def test_vector_retriever_accepts_filters_signature(self):
        """The retrieve method accepts the filters parameter in its signature."""
        import inspect

        sig = inspect.signature(VectorRetriever.retrieve)
        assert "filters" in sig.parameters
        param = sig.parameters["filters"]
        assert param.default is None


@pytest.mark.unit
class TestContextAssembler:
    def _make_result(self, content: str, source: str, score: float) -> RetrievalResult:
        chunk = Chunk(content=content, chunk_index=0, source=source)
        return RetrievalResult(chunk=chunk, score=score)

    def test_context_assembler_formats_chunks(self):
        results = [
            self._make_result("First chunk text.", "doc1.txt", 0.95),
            self._make_result("Second chunk text.", "doc2.txt", 0.80),
        ]
        assembler = ContextAssembler()
        ctx = assembler.assemble(results)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "First chunk text." in ctx
        assert "Second chunk text." in ctx
        assert "doc1.txt" in ctx

    def test_context_assembler_custom_template(self):
        results = [
            self._make_result("Content here.", "src.txt", 0.90),
        ]
        assembler = ContextAssembler()
        tpl = "Result {index}: {content} (from {source})"
        ctx = assembler.assemble(results, template=tpl)
        assert "Result 1: Content here. (from src.txt)" in ctx

    def test_context_assembler_empty_chunks(self):
        assembler = ContextAssembler()
        ctx = assembler.assemble([])
        assert ctx == ""

    # ------------------------------------------------------------------ #
    # Format presets
    # ------------------------------------------------------------------ #

    def test_format_numbered_is_default(self):
        results = [self._make_result("Hello.", "a.txt", 0.91)]
        assembler = ContextAssembler()
        ctx = assembler.assemble(results)
        assert ctx == "[1] (source: a.txt, score: 0.91):\nHello."

    def test_format_xml(self):
        results = [self._make_result("Hello.", "a.txt", 0.91)]
        assembler = ContextAssembler(format="xml")
        ctx = assembler.assemble(results)
        assert '<chunk index="1" source="a.txt" score="0.91">' in ctx
        assert "Hello." in ctx
        assert "</chunk>" in ctx

    def test_format_plain(self):
        results = [
            self._make_result("First.", "a.txt", 0.9),
            self._make_result("Second.", "b.txt", 0.8),
        ]
        assembler = ContextAssembler(format="plain")
        ctx = assembler.assemble(results)
        assert ctx == "First.\n\nSecond."

    def test_explicit_template_overrides_format(self):
        results = [self._make_result("Body.", "s.txt", 0.75)]
        tpl = ">> {content} <<"
        assembler = ContextAssembler(template=tpl, format="xml")
        ctx = assembler.assemble(results)
        assert ctx == ">> Body. <<"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format preset"):
            ContextAssembler(format="unknown")

    # ------------------------------------------------------------------ #
    # Token budget enforcement
    # ------------------------------------------------------------------ #

    def test_token_budget_excludes_chunks(self):
        results = [
            self._make_result("short", "a.txt", 0.9),
            self._make_result("this is a much longer chunk of text", "b.txt", 0.8),
            self._make_result("tiny", "c.txt", 0.7),
        ]
        # Use plain format so rendered text == content.  Budget allows
        # "short" (5) + separator (2) + "tiny" (4) = 11 but not the
        # long middle chunk.
        assembler = ContextAssembler(format="plain", max_tokens=12)
        ctx = assembler.assemble(results)
        assert "short" in ctx
        assert "tiny" in ctx
        assert "much longer" not in ctx

    def test_token_budget_no_chunks_fit(self):
        results = [self._make_result("hello world", "a.txt", 0.9)]
        assembler = ContextAssembler(format="plain", max_tokens=3)
        ctx = assembler.assemble(results)
        assert ctx == ""

    def test_token_budget_default_counter_uses_len(self):
        """When no token_counter is provided, len (char count) is used."""
        results = [self._make_result("abcde", "a.txt", 0.9)]
        assembler = ContextAssembler(format="plain", max_tokens=5)
        ctx = assembler.assemble(results)
        assert ctx == "abcde"

    def test_custom_token_counter(self):
        """A custom token_counter is respected for budget calculation."""

        def word_counter(text: str) -> int:
            return len(text.split())

        results = [
            self._make_result("one two", "a.txt", 0.9),
            self._make_result("three four five six", "b.txt", 0.8),
        ]
        # word budget of 3: "one two" = 2 words fits, separator "\n\n" = 1 word,
        # "three four five six" = 4 words would bring total to 2+1+4 = 7 > 3
        assembler = ContextAssembler(format="plain", max_tokens=3, token_counter=word_counter)
        ctx = assembler.assemble(results)
        assert "one two" in ctx
        assert "three four" not in ctx


# ====================================================================== #
# MMRRetriever
# ====================================================================== #


def _make_memory_result(key: str, value: str, score: float, metadata: dict[str, Any] | None = None) -> Any:
    """Create a ``MemoryResult``-compatible object."""
    from ia_agent_fwk.memory.models import MemoryResult

    return MemoryResult(key=key, value=value, score=score, metadata=metadata or {})


class _ControlledEmbeddingProvider(FakeEmbeddingProvider):
    """Embedding provider that returns distinct, controllable vectors.

    Each text is embedded as a one-hot-ish vector so we can reason about
    cosine similarities deterministically.
    """

    DIMENSION = 8

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._mapping = mapping or {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for t in texts:
            if t in self._mapping:
                results.append(self._mapping[t])
            else:
                # fallback: hash-based pseudo-embedding
                h = hash(t) % (10**8)
                vec = [float((h >> i) & 1) for i in range(self.DIMENSION)]
                results.append(vec)
        return results

    def dimension(self) -> int:
        return self.DIMENSION


@pytest.mark.unit
class TestMMRRetriever:
    """Tests for MMR retriever."""

    async def test_mmr_returns_results(self, mock_memory_backend: InMemoryBackend) -> None:
        """MMR retriever should return results from the backend."""
        await mock_memory_backend.store("d:0", "alpha content", metadata={})
        await mock_memory_backend.store("d:1", "alpha related", metadata={})
        await mock_memory_backend.store("d:2", "beta unrelated", metadata={})

        provider = _ControlledEmbeddingProvider()
        retriever = MMRRetriever(memory_backend=mock_memory_backend, embedding_provider=provider)
        results = await retriever.retrieve("alpha", top_k=2)

        assert len(results) >= 1
        assert all(isinstance(r, RetrievalResult) for r in results)

    async def test_mmr_empty_backend(self, mock_memory_backend: InMemoryBackend) -> None:
        """MMR on an empty backend returns an empty list."""
        provider = _ControlledEmbeddingProvider()
        retriever = MMRRetriever(memory_backend=mock_memory_backend, embedding_provider=provider)
        results = await retriever.retrieve("nothing", top_k=5)
        assert results == []

    async def test_mmr_diversity(self) -> None:
        """MMR should prefer diverse results over redundant ones.

        We set up three candidates: two near-duplicates and one distinct.
        With lambda=0.5 (balanced), MMR should pick the distinct one over
        the second duplicate.
        """
        # Candidate embeddings: A and B are near-identical, C is different
        emb_a = [1.0, 0.0, 0.0, 0.0]
        emb_b = [0.99, 0.01, 0.0, 0.0]  # very similar to A
        emb_c = [0.0, 0.0, 1.0, 0.0]  # orthogonal to A and B
        # Query differs from A so diversity penalty differentiates B vs C
        query_emb = [0.8, 0.0, 0.6, 0.0]

        # Mock the memory backend
        backend = AsyncMock()
        backend.search = AsyncMock(
            return_value=[
                _make_memory_result("a", "doc A", 1.0, {"embedding": emb_a}),
                _make_memory_result("b", "doc B", 0.99, {"embedding": emb_b}),
                _make_memory_result("c", "doc C", 0.5, {"embedding": emb_c}),
            ]
        )

        provider = _ControlledEmbeddingProvider({"query_text": query_emb})

        retriever = MMRRetriever(
            memory_backend=backend,
            embedding_provider=provider,
            lambda_mult=0.5,
        )
        results = await retriever.retrieve("query_text", top_k=2)

        assert len(results) == 2
        # First pick should be A (highest relevance to query)
        assert results[0].chunk.source == "a"
        # Second pick should be C (diverse) rather than B (near-duplicate of A)
        assert results[1].chunk.source == "c"

    async def test_mmr_lambda_1_is_pure_relevance(self) -> None:
        """With lambda=1.0, MMR degenerates to pure relevance ranking."""
        emb_a = [1.0, 0.0, 0.0, 0.0]
        emb_b = [0.99, 0.01, 0.0, 0.0]
        emb_c = [0.0, 0.0, 1.0, 0.0]
        query_emb = [1.0, 0.0, 0.0, 0.0]

        backend = AsyncMock()
        backend.search = AsyncMock(
            return_value=[
                _make_memory_result("a", "doc A", 1.0, {"embedding": emb_a}),
                _make_memory_result("b", "doc B", 0.99, {"embedding": emb_b}),
                _make_memory_result("c", "doc C", 0.5, {"embedding": emb_c}),
            ]
        )

        provider = _ControlledEmbeddingProvider({"query_text": query_emb})
        retriever = MMRRetriever(
            memory_backend=backend,
            embedding_provider=provider,
            lambda_mult=1.0,
        )
        results = await retriever.retrieve("query_text", top_k=2)

        assert len(results) == 2
        # Pure relevance: A then B (most similar to query)
        assert results[0].chunk.source == "a"
        assert results[1].chunk.source == "b"

    async def test_mmr_embeds_candidates_without_stored_embeddings(self) -> None:
        """Candidates without stored embeddings are embedded on the fly."""
        query_emb = [1.0, 0.0, 0.0, 0.0]

        backend = AsyncMock()
        backend.search = AsyncMock(
            return_value=[
                _make_memory_result("a", "doc A", 1.0, {}),  # no embedding in metadata
            ]
        )

        provider = _ControlledEmbeddingProvider({"query_text": query_emb, "doc A": [0.9, 0.1, 0.0, 0.0]})
        retriever = MMRRetriever(
            memory_backend=backend,
            embedding_provider=provider,
        )
        results = await retriever.retrieve("query_text", top_k=1)

        assert len(results) == 1
        assert results[0].chunk.content == "doc A"

    async def test_mmr_candidate_multiplier(self) -> None:
        """The candidate_multiplier controls how many candidates are fetched."""
        backend = AsyncMock()
        backend.search = AsyncMock(return_value=[])

        provider = _ControlledEmbeddingProvider({"q": [1.0, 0.0, 0.0, 0.0]})
        retriever = MMRRetriever(
            memory_backend=backend,
            embedding_provider=provider,
            candidate_multiplier=5,
        )
        await retriever.retrieve("q", top_k=3)

        # Should request top_k * candidate_multiplier = 15 candidates
        backend.search.assert_called_once()
        call_kwargs = backend.search.call_args
        assert call_kwargs[1]["top_k"] == 15


# ====================================================================== #
# FilteredRetriever
# ====================================================================== #


@pytest.mark.unit
class TestFilteredRetriever:
    """Tests for the FilteredRetriever decorator."""

    async def test_filters_are_passed_through(self) -> None:
        """Constructor filters are forwarded to the inner retriever."""
        inner = AsyncMock()
        inner.retrieve = AsyncMock(return_value=[])

        retriever = FilteredRetriever(inner, metadata_filters={"source": "docs"})
        await retriever.retrieve("hello", top_k=3)

        inner.retrieve.assert_called_once_with("hello", top_k=3, filters={"source": "docs"})

    async def test_call_time_filters_merged(self) -> None:
        """Call-time filters are merged with constructor filters."""
        inner = AsyncMock()
        inner.retrieve = AsyncMock(return_value=[])

        retriever = FilteredRetriever(inner, metadata_filters={"source": "docs"})
        await retriever.retrieve("hello", top_k=3, filters={"author": "alice"})

        inner.retrieve.assert_called_once_with("hello", top_k=3, filters={"source": "docs", "author": "alice"})

    async def test_call_time_filters_override_constructor(self) -> None:
        """Call-time filters take precedence over constructor filters for same key."""
        inner = AsyncMock()
        inner.retrieve = AsyncMock(return_value=[])

        retriever = FilteredRetriever(inner, metadata_filters={"source": "old"})
        await retriever.retrieve("hello", top_k=3, filters={"source": "new"})

        inner.retrieve.assert_called_once_with("hello", top_k=3, filters={"source": "new"})

    async def test_no_call_time_filters(self) -> None:
        """When no call-time filters are provided, only constructor filters apply."""
        inner = AsyncMock()
        inner.retrieve = AsyncMock(return_value=[])

        retriever = FilteredRetriever(inner, metadata_filters={"type": "pdf"})
        await retriever.retrieve("search")

        inner.retrieve.assert_called_once_with("search", top_k=5, filters={"type": "pdf"})

    async def test_delegates_results(self) -> None:
        """FilteredRetriever returns whatever the inner retriever returns."""
        expected = [
            RetrievalResult(
                chunk=Chunk(content="hello", source="test"),
                score=0.9,
            )
        ]
        inner = AsyncMock()
        inner.retrieve = AsyncMock(return_value=expected)

        retriever = FilteredRetriever(inner, metadata_filters={})
        results = await retriever.retrieve("q")
        assert results == expected


# ====================================================================== #
# RetrieverFactory
# ====================================================================== #


@pytest.mark.unit
class TestRetrieverFactory:
    """Tests for the RetrieverFactory."""

    def test_create_vector(self, mock_memory_backend: InMemoryBackend) -> None:
        provider = FakeEmbeddingProvider()
        retriever = RetrieverFactory.create("vector", mock_memory_backend, embedding_provider=provider)
        assert isinstance(retriever, VectorRetriever)

    def test_create_mmr(self, mock_memory_backend: InMemoryBackend) -> None:
        provider = FakeEmbeddingProvider()
        retriever = RetrieverFactory.create("mmr", mock_memory_backend, embedding_provider=provider)
        assert isinstance(retriever, MMRRetriever)

    def test_create_mmr_with_kwargs(self, mock_memory_backend: InMemoryBackend) -> None:
        provider = FakeEmbeddingProvider()
        retriever = RetrieverFactory.create(
            "mmr",
            mock_memory_backend,
            embedding_provider=provider,
            lambda_mult=0.7,
            candidate_multiplier=5,
        )
        assert isinstance(retriever, MMRRetriever)
        assert retriever._lambda_mult == 0.7
        assert retriever._candidate_multiplier == 5

    def test_create_vector_without_provider_raises(self, mock_memory_backend: InMemoryBackend) -> None:
        with pytest.raises(RetrievalError, match="Vector retriever requires"):
            RetrieverFactory.create("vector", mock_memory_backend)

    def test_create_mmr_without_provider_raises(self, mock_memory_backend: InMemoryBackend) -> None:
        with pytest.raises(RetrievalError, match="MMR retriever requires"):
            RetrieverFactory.create("mmr", mock_memory_backend)

    def test_unknown_strategy_raises(self, mock_memory_backend: InMemoryBackend) -> None:
        with pytest.raises(RetrievalError, match="Unknown retrieval strategy"):
            RetrieverFactory.create("unknown", mock_memory_backend)
