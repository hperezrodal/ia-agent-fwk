"""Query pipeline — composes: search → post-process → (optional) rerank → context assembly.

Generic query orchestrator. Used by both the CLI script and the agent tool.

Usage:
    pipeline = QueryPipeline(embedding_store)
    results = await pipeline.search("query text", top_k=5)

    # With reranking:
    pipeline = QueryPipeline(embedding_store, rerank_model="BAAI/bge-reranker-v2-m3")
    results = await pipeline.search("query text", top_k=5)
"""

from __future__ import annotations

from collections.abc import Callable

from ia_agent_fwk.ingestion.context_assembly import assemble_context
from ia_agent_fwk.ingestion.embedding_store import EmbeddingStore
from ia_agent_fwk.ingestion.hybrid_store import HybridSearchResult, SearchMode
from ia_agent_fwk.ingestion.post_processing import (
    Processor,
    deduplicate_parent_child,
    limit,
    post_process,
    rerank,
)


class QueryPipeline:
    """Search → Post-process → (optional) Rerank → Context assembly.

    Parameters
    ----------
    store:
        EmbeddingStore instance (handles dual encoding + Qdrant search).
    rerank_model:
        If provided, re-rank results with this cross-encoder model.
        Ej: "BAAI/bge-reranker-v2-m3"
    rerank_device:
        Device for the reranker ("cpu" or "cuda").
    context_format:
        Format for context assembly ("numbered", "xml", "plain").

    """

    def __init__(
        self,
        store: EmbeddingStore,
        rerank_model: str | None = None,
        rerank_device: str = "cpu",
        context_format: str = "numbered",
        query_expander: Callable[..., list[str]] | None = None,
    ) -> None:
        self._store = store
        self._rerank_model = rerank_model
        self._rerank_device = rerank_device
        self._context_format = context_format
        self._query_expander = query_expander

    def _build_processors(self, query: str, top_k: int) -> list[Processor]:
        """Build the processor chain for a specific query."""
        processors: list[Processor] = [
            deduplicate_parent_child(),
        ]
        if self._rerank_model:
            processors.append(
                rerank(
                    query=query,
                    model_name=self._rerank_model,
                    device=self._rerank_device,
                )
            )
            # After reranking, limit to the original top_k
            processors.append(limit(top_k))
        return processors

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        mode: SearchMode | None = None,
    ) -> list[HybridSearchResult]:
        """Search + post-process + (optional) rerank."""
        # Expand query with domain synonyms for better retrieval
        search_query = self._query_expander(query) if self._query_expander else query

        # If reranking, fetch more candidates for the reranker to re-order
        fetch_k = top_k * 10 if self._rerank_model else top_k

        results = await self._store.search(
            query=search_query,
            top_k=fetch_k,
            filters=filters,
            mode=mode,
        )

        # Reranker uses ORIGINAL query (not expanded) for accurate scoring
        processors = self._build_processors(query, top_k)
        return post_process(results, processors)

    async def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        mode: SearchMode | None = None,
        max_context_chars: int = 0,
    ) -> tuple[list[HybridSearchResult], str]:
        """Search + post-process + rerank + assemble context string for LLM."""
        results = await self.search(
            query=query,
            top_k=top_k,
            filters=filters,
            mode=mode,
        )
        context = assemble_context(
            results,
            format=self._context_format,
            max_chars=max_context_chars,
        )
        return results, context
