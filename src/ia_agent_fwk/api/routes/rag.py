"""RAG pipeline endpoints: ingest documents and search."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Security
from pydantic import BaseModel

from ia_agent_fwk.api.dependencies import check_rate_limit, get_settings, require_api_key
from ia_agent_fwk.config.settings import AppSettings  # noqa: TC001
from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["rag"],
    dependencies=[Security(require_api_key), Depends(check_rate_limit)],
)


class IngestRequest(BaseModel):
    """Request to ingest a document by filename."""

    filename: str | None = None


class IngestResponse(BaseModel):
    """Response from ingestion."""

    files_ingested: list[str]
    total_chunks: int


class SearchRequest(BaseModel):
    """RAG search request."""

    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """Single search result."""

    content: str
    source: str
    score: float
    chunk_index: int | None = None


class SearchResponse(BaseModel):
    """RAG search response."""

    results: list[SearchResult]
    query: str


def _get_rag_pipeline(settings: AppSettings) -> Any:
    """Create a RAG pipeline wired to Qdrant + Ollama embeddings."""
    from ia_agent_fwk.memory.embeddings.factory import EmbeddingFactory  # noqa: PLC0415
    from ia_agent_fwk.memory.factory import MemoryFactory  # noqa: PLC0415
    from ia_agent_fwk.rag.factory import RAGPipelineFactory  # noqa: PLC0415

    embedding_provider = EmbeddingFactory.create(settings.memory.embedding)
    memory_backend = MemoryFactory.create(settings.memory)
    return RAGPipelineFactory.create(
        settings=settings.rag,
        memory_backend=memory_backend,
        embedding_provider=embedding_provider,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request_body: IngestRequest,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> IngestResponse:
    """Ingest documents from the documents directory into the vector store.

    If ``filename`` is provided, ingest only that file.
    Otherwise, ingest all supported files in the documents directory.
    """
    docs_dir = Path(os.environ.get("DOCUMENTS_DIR", "documents"))
    pipeline = _get_rag_pipeline(settings)

    supported_extensions = {".txt", ".md", ".html", ".htm", ".pdf"}
    files_to_ingest: list[Path] = []

    if request_body.filename:
        target = docs_dir / request_body.filename
        # Path traversal check
        resolved = target.resolve()
        if not str(resolved).startswith(str(docs_dir.resolve())):  # noqa: ASYNC240
            from fastapi import HTTPException  # noqa: PLC0415

            raise HTTPException(status_code=400, detail="Invalid filename")
        if resolved.exists():
            files_to_ingest.append(resolved)
    elif docs_dir.exists():  # noqa: ASYNC240
        # Ingest all supported files
        files_to_ingest.extend(
            f
            for f in sorted(docs_dir.iterdir())  # noqa: ASYNC240
            if f.is_file() and f.suffix.lower() in supported_extensions
        )

    total_chunks = 0
    ingested_files: list[str] = []
    collector = get_metrics_collector()

    for file_path in files_to_ingest:
        try:
            start = time.monotonic()
            result = await pipeline.ingest(file_path)
            duration = time.monotonic() - start
            total_chunks += result.chunk_count
            ingested_files.append(file_path.name)
            collector.increment("rag_documents_ingested_total")
            collector.increment("rag_chunks_stored_total", value=result.chunk_count)
            collector.observe("rag_ingest_duration_seconds", duration)
            logger.info("Ingested %s: %d chunks", file_path.name, result.chunk_count)
        except Exception:
            collector.increment("rag_ingest_errors_total")
            logger.exception("Failed to ingest %s", file_path.name)

    return IngestResponse(files_ingested=ingested_files, total_chunks=total_chunks)


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request_body: SearchRequest,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> SearchResponse:
    """Search ingested documents using vector similarity."""
    collector = get_metrics_collector()
    pipeline = _get_rag_pipeline(settings)

    start = time.monotonic()
    results = await pipeline.query(request_body.query, top_k=request_body.top_k)
    duration = time.monotonic() - start

    collector.increment("rag_searches_total")
    collector.increment("rag_results_returned_total", value=len(results.results))
    collector.observe("rag_search_duration_seconds", duration)
    if results.results:
        collector.observe("rag_top_score", results.results[0].score)

    return SearchResponse(
        query=request_body.query,
        results=[
            SearchResult(
                content=r.chunk.content,
                source=r.chunk.source,
                score=r.score,
                chunk_index=r.chunk.chunk_index,
            )
            for r in results.results
        ],
    )
