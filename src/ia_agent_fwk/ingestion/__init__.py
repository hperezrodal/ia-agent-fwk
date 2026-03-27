"""Document ingestion pipeline — generic, composable.

Provides a production-ready pipeline for document ingestion and retrieval:
  - Classify documents by type
  - Parse with multiple backends (Docling, pymupdf, DOCX, OCR)
  - Clean text (Unicode, encoding, whitespace, boilerplate)
  - Chunk with composable stages (SPLIT → TRANSFORM → SIZE → ENRICH)
  - Enrich with composable enrichers (document info, language, custom)
  - Embed + store with hybrid search (dense + BM25 sparse)
  - Query with reranking and post-processing
"""

from ia_agent_fwk.ingestion.classifier import DocumentClassifier, DocumentType
from ia_agent_fwk.ingestion.cleaner import DocumentCleaner
from ia_agent_fwk.ingestion.models import ProcessedChunk
from ia_agent_fwk.ingestion.orchestrator import IngestionOrchestrator

__all__ = [
    "DocumentClassifier",
    "DocumentCleaner",
    "DocumentType",
    "IngestionOrchestrator",
    "ProcessedChunk",
]
