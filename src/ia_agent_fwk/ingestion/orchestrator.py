"""Ingestion orchestrator — composes: classify → parse → clean → chunk → enrich.

Each step is a generic, independent component:
  - Classifier: EXTRACT → ANALYZE → DECIDE  (classifier.py)
  - Parser: Path → str                      (parsers.py)
  - Cleaner: str → str                      (cleaner.py)
  - Chunking: str → list[ProcessedChunk]     (chunking/)
  - Enrichment: list[Enricher] applied       (enrichment.py)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ia_agent_fwk.ingestion.chunking import ChunkingPipeline
from ia_agent_fwk.ingestion.classifier import ClassificationResult, DocumentClassifier
from ia_agent_fwk.ingestion.cleaner import DocumentCleaner
from ia_agent_fwk.ingestion.enrichment import (
    custom_metadata,
    document_info,
    enrich_pipeline,
    filter_empty,
    language_detection,
)
from ia_agent_fwk.ingestion.models import ProcessedChunk
from ia_agent_fwk.ingestion.parsers import load_parsed, parse

if TYPE_CHECKING:
    from ia_agent_fwk.ingestion.contextual_enrichment import ContextualEnricher

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of processing a single document through the orchestrator."""

    file_path: str
    classification: ClassificationResult
    chunks: list[ProcessedChunk]
    duration_ms: float
    parser_used: str = ""
    language: str = "es"
    errors: list[str] = field(default_factory=list)

    @property
    def pipeline_used(self) -> str:
        return self.parser_used

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class IngestionOrchestrator:
    """Classify → Parse → Clean → Chunk → Enrich.

    Parameters
    ----------
    chunk_size:
        Max characters per chunk.
    chunk_overlap:
        Overlap between chunks.

    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        contextual_model: str | None = None,
        config_provider: object | None = None,
    ) -> None:
        self._classifier = DocumentClassifier()
        self._cleaner = DocumentCleaner()
        self._chunking = ChunkingPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._contextual_model = contextual_model
        self._contextual_enricher: ContextualEnricher | None = None  # lazy init
        self._config_provider = config_provider

    async def process(
        self,
        file_path: str | Path,
        extra_metadata: dict[str, str | int | float] | None = None,
        *,
        save_parsed: str | Path | None = None,
        from_parsed: str | Path | None = None,
    ) -> IngestionResult:
        """Process a document end-to-end.

        1. Classify
        2. Parse (or load pre-parsed)
        3. Clean
        4. Chunk
        5. Enrich (composable enricher pipeline)

        Parameters
        ----------
        save_parsed:
            If provided, save the raw parser output to this path/directory.
            Useful for debugging or to resume later without re-parsing.
        from_parsed:
            If provided, skip parse+clean and load from this pre-parsed file.
            Use for re-chunking/re-embedding without re-parsing.

        """
        path = Path(file_path)
        t0 = time.monotonic()

        # 1. Classify
        classification = self._classifier.classify(path)

        # 2. Parse (or resume from pre-parsed)
        if from_parsed:
            text, parser_used = load_parsed(from_parsed)
            is_markdown = parser_used == "docling"
        else:
            text, parser_used = parse(path, save_to=save_parsed)
            is_markdown = parser_used == "docling"

        # 3. Clean
        text = self._cleaner.clean(text, preserve_tables=is_markdown)

        # 4. Chunk
        chunks = self._chunking.process(text)

        # 4b. Contextual enrichment (optional, LLM-generated context per chunk)
        if self._contextual_model:
            if self._contextual_enricher is None:
                from ia_agent_fwk.ingestion.contextual_enrichment import ContextualEnricher  # noqa: PLC0415

                self._contextual_enricher = ContextualEnricher(model=self._contextual_model)
            chunks = await self._contextual_enricher.enrich(chunks, text)

        # 5. Enrich (composable pipeline of enrichers)
        # Resolve scopes from config provider if available
        scope_metadata: dict[str, str | int | float] | None = None
        if self._config_provider and hasattr(self._config_provider, "resolve_scopes"):
            scopes = self._config_provider.resolve_scopes(path)
            # Store as comma-separated for Qdrant MatchAny compatibility
            scope_metadata = {"scope": scopes[0] if len(scopes) == 1 else ",".join(scopes)}

        enrichers_list = [
            filter_empty(),
            document_info(path, parser_used, classification),
            language_detection(default="es"),
            custom_metadata(extra_metadata),
        ]
        if scope_metadata:
            enrichers_list.append(custom_metadata(scope_metadata))

        chunks = enrich_pipeline(chunks, enrichers=enrichers_list)

        # Extract language from enriched chunks for result
        language = "es"
        if chunks:
            language = str(chunks[0].metadata.get("language", "es"))

        duration_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "Orchestrator: %s → %s → %s → %d chunks in %.0fms (lang=%s)",
            path.name,
            classification.doc_type.value,
            parser_used,
            len(chunks),
            duration_ms,
            language,
        )

        return IngestionResult(
            file_path=str(path),
            classification=classification,
            chunks=chunks,
            duration_ms=duration_ms,
            parser_used=parser_used,
            language=language,
        )
