"""RAG pipeline orchestrator: ingest and query flows."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer
from ia_agent_fwk.rag.exceptions import DocumentLoadError, EmbeddingError, RetrievalError
from ia_agent_fwk.rag.models import IngestionResult, QueryResult
from ia_agent_fwk.rag.retrieval.context import ContextAssembler

if TYPE_CHECKING:
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
    from ia_agent_fwk.rag.chunkers.base import Chunker
    from ia_agent_fwk.rag.loaders.registry import LoaderRegistry
    from ia_agent_fwk.rag.retrieval.base import Retriever

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline orchestrator.

    Combines document loading, chunking, embedding, storage, retrieval,
    and context assembly into two primary flows:

    - ``ingest``: load a file, chunk it, embed chunks, store them.
    - ``query``: embed a query, retrieve relevant chunks, assemble context.

    Parameters
    ----------
    loader_registry:
        Registry of document loaders.
    chunker:
        Chunking strategy instance.
    embedding_provider:
        Provider for generating embeddings.
    memory_backend:
        Vector memory backend for storing / searching chunks.
    retriever:
        Retrieval strategy instance.

    """

    def __init__(
        self,
        loader_registry: LoaderRegistry,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        memory_backend: MemoryBackend,
        retriever: Retriever,
    ) -> None:
        self._loader_registry = loader_registry
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._backend = memory_backend
        self._retriever = retriever
        self._assembler = ContextAssembler()

    async def ingest(self, file_path: str | Path) -> IngestionResult:
        """Ingest a document: load, chunk, embed, and store.

        Returns an :class:`IngestionResult` with chunk count and timing.
        """
        collector = get_metrics_collector()
        t0 = time.monotonic()
        path = Path(file_path)

        # 1. Load
        load_start = time.monotonic()
        try:
            document = await self._loader_registry.load(path)
        except DocumentLoadError:
            collector.increment("rag_pipeline_ingest_total", labels={"status": "failure", "stage": "load"})
            raise
        except Exception as exc:
            collector.increment("rag_pipeline_ingest_total", labels={"status": "failure", "stage": "load"})
            msg = f"Failed to load document {path}: {exc}"
            raise DocumentLoadError(msg) from exc
        load_ms = (time.monotonic() - load_start) * 1000
        collector.observe("rag_pipeline_load_duration_seconds", load_ms / 1000)

        # 2. Chunk
        chunk_start = time.monotonic()
        raw_chunks = await self._chunker.chunk(document)
        chunk_ms = (time.monotonic() - chunk_start) * 1000
        collector.observe("rag_pipeline_chunk_duration_seconds", chunk_ms / 1000)

        # Stamp each chunk with the document_id (file name)
        doc_id = path.name
        chunks = [c.model_copy(update={"document_id": doc_id}) for c in raw_chunks]
        if not chunks:
            logger.info("No chunks produced for %s", path)
            duration_ms = (time.monotonic() - t0) * 1000
            collector.increment("rag_pipeline_ingest_total", labels={"status": "success", "stage": "complete"})
            return IngestionResult(
                document_id=path.name,
                chunk_count=0,
                duration_ms=duration_ms,
            )

        # 3. Embed
        embed_start = time.monotonic()
        texts = [c.content for c in chunks]
        try:
            embeddings = await self._embedding_provider.embed(texts)
        except Exception as exc:
            collector.increment("rag_pipeline_ingest_total", labels={"status": "failure", "stage": "embed"})
            msg = f"Embedding generation failed: {exc}"
            raise EmbeddingError(msg) from exc
        embed_ms = (time.monotonic() - embed_start) * 1000
        collector.observe("rag_pipeline_embed_duration_seconds", embed_ms / 1000)

        # 4. Store
        store_start = time.monotonic()
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            key = f"{path.name}:chunk:{chunk.chunk_index}"
            metadata = {
                **chunk.metadata,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "document_id": chunk.document_id,
            }
            await self._backend.store(key, chunk.content, metadata={"embedding": embedding, **metadata})
        store_ms = (time.monotonic() - store_start) * 1000
        collector.observe("rag_pipeline_store_duration_seconds", store_ms / 1000)

        duration_ms = (time.monotonic() - t0) * 1000
        collector.increment("rag_pipeline_ingest_total", labels={"status": "success", "stage": "complete"})
        collector.observe("rag_pipeline_ingest_duration_seconds", duration_ms / 1000)
        collector.observe("rag_pipeline_chunks_per_document", len(chunks))

        logger.info(
            "Ingested %d chunks from %s (load=%.0fms, chunk=%.0fms, embed=%.0fms, store=%.0fms, total=%.0fms)",
            len(chunks),
            path,
            load_ms,
            chunk_ms,
            embed_ms,
            store_ms,
            duration_ms,
            extra={
                "rag_data": {
                    "event": "ingest_completed",
                    "document_id": path.name,
                    "chunk_count": len(chunks),
                    "load_ms": round(load_ms, 1),
                    "chunk_ms": round(chunk_ms, 1),
                    "embed_ms": round(embed_ms, 1),
                    "store_ms": round(store_ms, 1),
                    "duration_ms": round(duration_ms, 1),
                    "doc_chars": len(document.content),
                }
            },
        )
        return IngestionResult(
            document_id=path.name,
            chunk_count=len(chunks),
            duration_ms=duration_ms,
        )

    async def ingest_batch(self, file_paths: list[str | Path]) -> list[IngestionResult]:
        """Ingest multiple files. Return a list of :class:`IngestionResult`."""
        return [await self.ingest(fp) for fp in file_paths]

    async def query(self, query: str, top_k: int = 5) -> QueryResult:
        """Embed *query*, search the vector backend, and return a :class:`QueryResult`."""
        collector = get_metrics_collector()
        t0 = time.monotonic()
        try:
            results = await self._retriever.retrieve(query, top_k=top_k)
        except RetrievalError:
            collector.increment("rag_pipeline_query_total", labels={"status": "failure"})
            raise
        except Exception as exc:
            collector.increment("rag_pipeline_query_total", labels={"status": "failure"})
            msg = f"Query failed: {exc}"
            raise RetrievalError(msg) from exc

        retrieval_ms = (time.monotonic() - t0) * 1000
        context = self._assembler.assemble(results)
        total_ms = (time.monotonic() - t0) * 1000

        collector.increment("rag_pipeline_query_total", labels={"status": "success"})
        collector.observe("rag_pipeline_query_duration_seconds", total_ms / 1000)
        collector.observe("rag_pipeline_query_results_count", len(results))
        if results:
            collector.observe("rag_pipeline_query_top_score", results[0].score)

        logger.info(
            "RAG query completed: results=%d, retrieval=%.0fms, total=%.0fms",
            len(results),
            retrieval_ms,
            total_ms,
            extra={
                "rag_data": {
                    "event": "query_completed",
                    "results_count": len(results),
                    "top_score": round(results[0].score, 4) if results else 0,
                    "retrieval_ms": round(retrieval_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "context_chars": len(context),
                }
            },
        )

        return QueryResult(
            results=results,
            context=context,
            retrieval_ms=retrieval_ms,
            total_ms=total_ms,
        )

    async def query_with_context(
        self,
        query: str,
        top_k: int = 5,
        template: str | None = None,
    ) -> str:
        """Query and assemble a formatted context string.

        Parameters
        ----------
        query:
            The search query.
        top_k:
            Number of results to retrieve.
        template:
            Optional format template override for context assembly.

        Returns
        -------
        str
            Assembled context string (empty if no results).

        """
        query_result = await self.query(query, top_k=top_k)
        if template is not None:
            return self._assembler.assemble(query_result.results, template=template)
        return query_result.context
