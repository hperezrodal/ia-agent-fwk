"""Search result post-processing — composable processors.

Each processor is a pure function: list[SearchResult] → list[SearchResult].
Processors run after hybrid search, before results are consumed by the LLM
or the CLI.

Usage:
    from ia_agent_fwk.ingestion.post_processing import post_process, deduplicate_parent_child

    results = await store.search(query, top_k=10)
    results = post_process(results, processors=[
        deduplicate_parent_child(),
        expand_parent_context(store),
    ])
"""

from __future__ import annotations

from collections.abc import Callable

from ia_agent_fwk.ingestion.hybrid_store import HybridSearchResult

# Type alias
Processor = Callable[[list[HybridSearchResult]], list[HybridSearchResult]]


# ═══════════════════════════════════════════════════════════════════════════
# Processor: deduplicate parent-child
# ═══════════════════════════════════════════════════════════════════════════


def deduplicate_parent_child() -> Processor:
    """If a table child matched, remove its parent from results.

    When hierarchical table chunking produces both parent (full table) and
    child (per-row) chunks, both can appear in search results. The child is
    more specific and useful — the parent is redundant context.
    """

    def _process(results: list[HybridSearchResult]) -> list[HybridSearchResult]:
        # Collect table_ids of matched children
        child_table_ids: set[str] = set()
        for r in results:
            if r.metadata.get("table_role") == "child":
                tid = r.metadata.get("table_id", "")
                if tid:
                    child_table_ids.add(tid)

        # Remove parents whose children already appear
        return [
            r
            for r in results
            if not (r.metadata.get("table_role") == "parent" and r.metadata.get("table_id", "") in child_table_ids)
        ]

    return _process


# ═══════════════════════════════════════════════════════════════════════════
# Processor: attach parent context to children
# ═══════════════════════════════════════════════════════════════════════════


def attach_parent_context(all_results: list[HybridSearchResult]) -> Processor:
    """For table children in results, attach the parent's full table as context.

    Adds metadata["parent_content"] with the full table markdown.
    The LLM can use this for broader context when answering.

    Parameters
    ----------
    all_results:
        A broader set of results (eg. top_k * 3) to find parents.
        Pass the full prefetch results, not just the filtered top_k.

    """
    # Build parent lookup: table_id → content
    parent_map: dict[str, str] = {}
    for r in all_results:
        if r.metadata.get("table_role") == "parent":
            tid = r.metadata.get("table_id", "")
            if tid:
                parent_map[tid] = r.content

    def _process(results: list[HybridSearchResult]) -> list[HybridSearchResult]:
        for r in results:
            if r.metadata.get("table_role") == "child":
                tid = r.metadata.get("parent_chunk_id", "")
                parent_content = parent_map.get(tid)
                if parent_content:
                    r.metadata["parent_content"] = parent_content
        return results

    return _process


# ═══════════════════════════════════════════════════════════════════════════
# Processor: score threshold filter
# ═══════════════════════════════════════════════════════════════════════════


def score_threshold(min_score: float = 0.0) -> Processor:
    """Remove results below a minimum score."""

    def _process(results: list[HybridSearchResult]) -> list[HybridSearchResult]:
        if min_score <= 0.0:
            return results
        return [r for r in results if r.score >= min_score]

    return _process


# ═══════════════════════════════════════════════════════════════════════════
# Processor: limit
# ═══════════════════════════════════════════════════════════════════════════


def limit(max_results: int) -> Processor:
    """Cap the number of results."""

    def _process(results: list[HybridSearchResult]) -> list[HybridSearchResult]:
        return results[:max_results]

    return _process


# ═══════════════════════════════════════════════════════════════════════════
# Processor: rerank with cross-encoder
# ═══════════════════════════════════════════════════════════════════════════

_reranker_model = None  # module-level lazy singleton


def rerank(
    query: str,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cpu",
) -> Processor:
    """Re-rank results using a cross-encoder model.

    The cross-encoder scores (query, chunk) pairs jointly — more accurate
    than bi-encoder similarity but slower. Run on top-K candidates only.

    Parameters
    ----------
    query:
        The original query text.
    model_name:
        HuggingFace cross-encoder model.
    device:
        "cpu" or "cuda".

    """
    global _reranker_model  # noqa: PLW0603
    if _reranker_model is None or _reranker_model.config.name_or_path != model_name:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        _reranker_model = CrossEncoder(model_name, device=device)

    model = _reranker_model

    def _process(results: list[HybridSearchResult]) -> list[HybridSearchResult]:
        if not results:
            return results

        pairs = [(query, r.content) for r in results]
        scores = model.predict(pairs)

        # Replace RRF scores with reranker scores and re-sort
        reranked = []
        for r, score in zip(results, scores, strict=True):
            reranked.append(
                HybridSearchResult(
                    content=r.content,
                    score=float(score),
                    metadata=r.metadata,
                )
            )
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    return _process


# ═══════════════════════════════════════════════════════════════════════════
# Compose processors
# ═══════════════════════════════════════════════════════════════════════════


def post_process(
    results: list[HybridSearchResult],
    processors: list[Processor],
) -> list[HybridSearchResult]:
    """Run results through a sequence of processors."""
    for processor in processors:
        results = processor(results)
    return results
