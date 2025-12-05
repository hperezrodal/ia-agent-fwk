"""Fixed-size text chunker with configurable overlap."""

from __future__ import annotations

import time

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.rag.chunkers.base import Chunker
from ia_agent_fwk.rag.models import Chunk, Document


class FixedSizeChunker(Chunker):
    """Split text into fixed character-length chunks.

    Tries to break at word boundaries when possible to avoid splitting
    words in the middle.
    """

    async def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into fixed-size chunks."""
        collector = get_metrics_collector()
        t0 = time.monotonic()
        text = document.content
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at a word boundary (space)
            if end < len(text):
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                # Find the actual start/end positions of the stripped text
                lstripped = text[start:end].lstrip()
                actual_start = start + (len(text[start:end]) - len(lstripped))
                actual_end = actual_start + len(chunk_text)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata=document.metadata,
                        chunk_index=idx,
                        source=document.source,
                        start_char=actual_start,
                        end_char=actual_end,
                    )
                )
                idx += 1

            # If we reached the end of the text, stop
            if end >= len(text):
                break

            # Advance by (chunk_size - overlap), but ensure we always make progress
            step = max(self.chunk_size - self.chunk_overlap, 1)
            start += step

        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("rag_chunking_total", labels={"strategy": "fixed"})
        collector.observe("rag_chunking_duration_seconds", duration_ms / 1000, labels={"strategy": "fixed"})
        collector.observe("rag_chunks_produced", len(chunks), labels={"strategy": "fixed"})
        if chunks:
            avg_size = sum(len(c.content) for c in chunks) / len(chunks)
            collector.observe("rag_chunk_size_chars", avg_size, labels={"strategy": "fixed"})
        return chunks
