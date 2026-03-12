"""Semantic chunker that splits based on embedding similarity."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.rag.chunkers.base import Chunker
from ia_agent_fwk.rag.models import Chunk, Document

if TYPE_CHECKING:
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider


class SemanticChunker(Chunker):
    """Split text into semantically coherent chunks using embeddings.

    Groups consecutive sentences whose embeddings are similar.
    When the cosine similarity between adjacent sentence groups drops
    below ``similarity_threshold``, a new chunk boundary is created.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ) -> None:
        # Deliberately skip Chunker.__init__ - SemanticChunker does not use
        # the fixed chunk_size / chunk_overlap parameters from the base class.
        self._embedding_provider = embedding_provider
        self._similarity_threshold = similarity_threshold
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size

    async def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into semantically coherent chunks."""
        collector = get_metrics_collector()
        t0 = time.monotonic()
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        if len(sentences) == 1:
            collector.increment("rag_chunking_total", labels={"strategy": "semantic"})
            collector.observe("rag_chunks_produced", 1, labels={"strategy": "semantic"})
            return [
                Chunk(
                    content=sentences[0],
                    chunk_index=0,
                    source=document.source,
                )
            ]

        # Get embeddings for all sentences
        embeddings = await self._embedding_provider.embed(sentences)

        # Group sentences by semantic similarity
        chunks: list[Chunk] = []
        current_group: list[str] = [sentences[0]]
        current_start = 0

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(
                embeddings[i - 1],
                embeddings[i],
            )
            current_text = " ".join(current_group)

            # Start new chunk if similarity drops or max size exceeded
            should_split = (
                similarity < self._similarity_threshold and len(current_text) >= self._min_chunk_size
            ) or len(current_text) >= self._max_chunk_size

            if should_split:
                chunk_content = " ".join(current_group)
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        chunk_index=len(chunks),
                        source=document.source,
                        start_char=current_start,
                        end_char=current_start + len(chunk_content),
                    )
                )
                current_start += len(chunk_content) + 1
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])

        # Add remaining sentences
        if current_group:
            chunk_content = " ".join(current_group)
            chunks.append(
                Chunk(
                    content=chunk_content,
                    chunk_index=len(chunks),
                    source=document.source,
                    start_char=current_start,
                    end_char=current_start + len(chunk_content),
                )
            )

        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("rag_chunking_total", labels={"strategy": "semantic"})
        collector.observe("rag_chunking_duration_seconds", duration_ms / 1000, labels={"strategy": "semantic"})
        collector.observe("rag_chunks_produced", len(chunks), labels={"strategy": "semantic"})
        collector.observe("rag_semantic_sentences_total", len(sentences))
        if chunks:
            avg_size = sum(len(c.content) for c in chunks) / len(chunks)
            collector.observe("rag_chunk_size_chars", avg_size, labels={"strategy": "semantic"})
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot / (norm_a * norm_b))
