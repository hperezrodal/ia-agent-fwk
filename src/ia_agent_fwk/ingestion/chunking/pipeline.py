"""Composable chunking pipeline.

Chains the 4 stages: SPLIT → TRANSFORM → SIZE → ENRICH.
Each stage is a pure function: list[Block] → list[Block].
"""

from __future__ import annotations

from dataclasses import dataclass

from ia_agent_fwk.ingestion.chunking.enrich import enrich_blocks
from ia_agent_fwk.ingestion.chunking.models import Block
from ia_agent_fwk.ingestion.chunking.size import resize_blocks
from ia_agent_fwk.ingestion.chunking.split import split_blocks
from ia_agent_fwk.ingestion.chunking.transform import transform_blocks
from ia_agent_fwk.ingestion.models import ProcessedChunk


@dataclass
class ChunkingConfig:
    """Configuration for the chunking pipeline."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100


class ChunkingPipeline:
    """Generic chunking pipeline by composition.

    Usage::

        pipeline = ChunkingPipeline(chunk_size=1000, chunk_overlap=200)
        chunks = pipeline.process(markdown_string)

    Each stage can also be called independently for debugging::

        blocks = pipeline.split(text)
        blocks = pipeline.transform(blocks)
        blocks = pipeline.resize(blocks)
        blocks = pipeline.enrich(blocks)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        self._config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    def process(self, text: str) -> list[ProcessedChunk]:
        """Run the full pipeline: SPLIT → TRANSFORM → SIZE → ENRICH.

        Returns ProcessedChunk objects compatible with the ingestion pipeline.
        """
        blocks = self.split(text)
        blocks = self.transform(blocks)
        blocks = self.resize(blocks)
        blocks = self.enrich(blocks)
        return self._to_processed_chunks(blocks)

    def split(self, text: str) -> list[Block]:
        """Stage 1: split document into typed semantic blocks."""
        return split_blocks(text)

    def transform(self, blocks: list[Block]) -> list[Block]:
        """Stage 2: type-aware transformations (tables → hierarchical, etc.)."""
        return transform_blocks(blocks)

    def resize(self, blocks: list[Block]) -> list[Block]:
        """Stage 3: split large blocks, merge small ones."""
        return resize_blocks(
            blocks,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            min_chunk_size=self._config.min_chunk_size,
        )

    def enrich(self, blocks: list[Block]) -> list[Block]:
        """Stage 4: attach final metadata."""
        return enrich_blocks(blocks)

    @staticmethod
    def _to_processed_chunks(blocks: list[Block]) -> list[ProcessedChunk]:
        """Convert Block objects to ProcessedChunk for the ingestion pipeline."""
        return [ProcessedChunk(content=b.content, metadata=b.metadata) for b in blocks]
