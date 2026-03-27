"""Shared data models for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProcessedChunk:
    """A chunk produced by the ingestion pipeline, ready for embedding.

    This is the universal output model — produced by ChunkingPipeline,
    consumed by enrichment, embedding_store, and the orchestrator.
    """

    content: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)
