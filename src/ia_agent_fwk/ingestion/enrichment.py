"""Metadata enrichment — composable enricher functions.

Each enricher is a pure function: list[ProcessedChunk] → list[ProcessedChunk].
Enrichers add metadata fields to chunks without modifying content.

Two types of metadata:
  A) Chunk-level: generated during chunking (section, chunk_type, table_role).
     Already present when enrichers run.
  B) Document-level: same for all chunks (doc_type, source, language, insurer).
     Added by enrichers in this module.

Usage:
    from ia_agent_fwk.ingestion.enrichment import enrich_pipeline

    chunks = enrich_pipeline(
        chunks,
        enrichers=[
            document_info(path, parser_used, classification),
            language_detection(default="es"),
            custom_metadata({"insurer": "Sancor"}),
        ],
    )

Each enricher can also be called independently:
    chunks = document_info(path, "docling", classification)(chunks)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ia_agent_fwk.ingestion.classifier import ClassificationResult
from ia_agent_fwk.ingestion.models import ProcessedChunk

# Type alias: an enricher takes chunks and returns enriched chunks
Enricher = Callable[[list[ProcessedChunk]], list[ProcessedChunk]]


# ═══════════════════════════════════════════════════════════════════════════
# Enricher: document info (doc_type, source, document_id, pipeline)
# ═══════════════════════════════════════════════════════════════════════════


def document_info(
    file_path: str | Path,
    parser_used: str,
    classification: ClassificationResult,
) -> Enricher:
    """Add document-level info to all chunks."""
    path = Path(file_path)
    meta = {
        "doc_type": classification.doc_type.value,
        "source": str(path),
        "document_id": path.name,
        "pipeline": parser_used,
    }

    def _enrich(chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
        for chunk in chunks:
            chunk.metadata.update(meta)
        return chunks

    return _enrich


# ═══════════════════════════════════════════════════════════════════════════
# Enricher: language detection
# ═══════════════════════════════════════════════════════════════════════════


def language_detection(default: str = "es") -> Enricher:
    """Detect document language and add to all chunks."""
    from ia_agent_fwk.ingestion.cleaner import detect_language  # noqa: PLC0415

    def _enrich(chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
        if not chunks:
            return chunks
        sample = " ".join(c.content[:200] for c in chunks[:5])
        lang = detect_language(sample, default=default)
        for chunk in chunks:
            chunk.metadata["language"] = lang
        return chunks

    return _enrich


# ═══════════════════════════════════════════════════════════════════════════
# Enricher: custom metadata (insurer, project-specific fields)
# ═══════════════════════════════════════════════════════════════════════════


def custom_metadata(meta: dict[str, str | int | float] | None) -> Enricher:
    """Add arbitrary metadata to all chunks."""

    def _enrich(chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
        if meta:
            for chunk in chunks:
                chunk.metadata.update(meta)
        return chunks

    return _enrich


# ═══════════════════════════════════════════════════════════════════════════
# Enricher: filter empty chunks
# ═══════════════════════════════════════════════════════════════════════════


def filter_empty() -> Enricher:
    """Remove chunks with empty or whitespace-only content."""

    def _enrich(chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
        return [c for c in chunks if c.content.strip()]

    return _enrich


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline: compose enrichers
# ═══════════════════════════════════════════════════════════════════════════


def enrich_pipeline(
    chunks: list[ProcessedChunk],
    enrichers: list[Enricher],
) -> list[ProcessedChunk]:
    """Run chunks through a sequence of enrichers.

    Each enricher is a function list[ProcessedChunk] → list[ProcessedChunk].
    They execute in order. Each can add metadata, filter, or transform chunks.
    """
    for enricher in enrichers:
        chunks = enricher(chunks)
    return chunks
