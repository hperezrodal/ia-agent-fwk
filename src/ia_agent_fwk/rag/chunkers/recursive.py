"""Recursive text chunker using a hierarchy of separators."""

from __future__ import annotations

import time

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.rag.chunkers.base import Chunker
from ia_agent_fwk.rag.models import Chunk, Document

_DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(Chunker):
    """Split text using a hierarchy of separators.

    Recursively splits chunks that exceed ``chunk_size`` using
    progressively finer separators: paragraphs, newlines, sentences,
    words, and finally individual characters.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._separators = separators if separators is not None else list(_DEFAULT_SEPARATORS)

    async def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* recursively."""
        collector = get_metrics_collector()
        t0 = time.monotonic()
        text = document.content
        if not text:
            return []

        raw_chunks = self._split_text(text, self._separators)

        chunks: list[Chunk] = []
        search_from = 0
        for idx, chunk_text in enumerate(raw_chunks):
            stripped = chunk_text.strip()
            if stripped:
                # Find the position of the stripped chunk in the original text
                pos = text.find(stripped, search_from)
                if pos == -1:
                    # Fallback: chunk may have overlap text prepended;
                    # search from the beginning
                    pos = text.find(stripped)
                start_char = pos if pos != -1 else 0
                end_char = start_char + len(stripped)
                # Only advance search_from when there's no overlap,
                # to handle overlapping chunks that reuse earlier text
                if self.chunk_overlap == 0:
                    search_from = end_char
                chunks.append(
                    Chunk(
                        content=stripped,
                        metadata=document.metadata,
                        chunk_index=idx,
                        source=document.source,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )

        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("rag_chunking_total", labels={"strategy": "recursive"})
        collector.observe("rag_chunking_duration_seconds", duration_ms / 1000, labels={"strategy": "recursive"})
        collector.observe("rag_chunks_produced", len(chunks), labels={"strategy": "recursive"})
        if chunks:
            avg_size = sum(len(c.content) for c in chunks) / len(chunks)
            collector.observe("rag_chunk_size_chars", avg_size, labels={"strategy": "recursive"})
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:  # noqa: C901, PLR0912
        """Recursively split *text* using the given separators."""
        if len(text) <= self.chunk_size:
            return [text]

        # Find the first separator that actually splits the text
        separator = ""
        remaining_seps: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_seps = []
                break
            if sep in text:
                separator = sep
                remaining_seps = separators[i + 1 :]
                break
        else:
            # No separator found; fall back to character-level split
            return self._split_by_chars(text)

        if separator == "":
            return self._split_by_chars(text)

        parts = text.split(separator)

        # Merge small parts together up to chunk_size, with overlap
        merged: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + separator + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # If this single part exceeds chunk_size, split it recursively
                if len(part) > self.chunk_size and remaining_seps:
                    sub_chunks = self._split_text(part, remaining_seps)
                    merged.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current:
            merged.append(current)

        # Apply overlap between consecutive chunks
        if self.chunk_overlap > 0 and len(merged) > 1:
            merged = self._apply_overlap(merged)

        return merged

    def _split_by_chars(self, text: str) -> list[str]:
        """Split text into chunks of ``chunk_size`` characters."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            step = max(self.chunk_size - self.chunk_overlap, 1)
            start += step
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap from the end of each chunk to the start of the next."""
        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            combined = overlap_text + chunks[i]
            # Trim to chunk_size if necessary
            if len(combined) > self.chunk_size:
                combined = combined[: self.chunk_size]
            result.append(combined)
        return result
