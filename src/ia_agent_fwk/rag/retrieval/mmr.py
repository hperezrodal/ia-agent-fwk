"""Maximal Marginal Relevance retriever for diverse results.

MMR balances relevance to the query with diversity among selected results.
At each step the candidate maximising
``lambda * sim(candidate, query) - (1 - lambda) * max(sim(candidate, s) for s in S)``
is chosen, where *S* is the set of already-selected candidates.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.rag.exceptions import RetrievalError
from ia_agent_fwk.rag.models import Chunk, RetrievalResult
from ia_agent_fwk.rag.retrieval.base import Retriever

if TYPE_CHECKING:
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
    from ia_agent_fwk.memory.models import MemoryResult

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_select(
    candidates: list[dict[str, Any]],
    query_embedding: list[float],
    top_k: int,
    lambda_mult: float,
) -> list[int]:
    """Run the MMR greedy selection loop.

    Returns the indices of the chosen candidates in selection order.
    """
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            cand_emb = candidates[idx]["embedding"]
            relevance = _cosine_similarity(cand_emb, query_embedding)
            max_sim = (
                max(_cosine_similarity(cand_emb, candidates[s]["embedding"]) for s in selected) if selected else 0.0
            )
            score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def _results_to_candidates(raw_results: list[MemoryResult]) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert backend results to candidate dicts, collecting texts that need embedding."""
    candidates: list[dict[str, Any]] = []
    texts_to_embed: list[str] = []

    for result in raw_results:
        content = str(result.value) if result.value is not None else ""
        metadata = result.metadata or {}
        embedding = metadata.get("embedding")
        candidates.append(
            {
                "content": content,
                "key": result.key,
                "score": result.score,
                "metadata": metadata,
                "embedding": embedding,
            }
        )
        if embedding is None:
            texts_to_embed.append(content)

    return candidates, texts_to_embed


def _candidates_to_retrieval_results(
    candidates: list[dict[str, Any]],
    selected_indices: list[int],
) -> list[RetrievalResult]:
    """Convert selected candidate dicts to ``RetrievalResult`` objects."""
    results: list[RetrievalResult] = []
    for idx in selected_indices:
        cand = candidates[idx]
        chunk = Chunk(
            content=cand["content"],
            metadata=cand["metadata"],
            source=cand["key"],
        )
        results.append(
            RetrievalResult(
                chunk=chunk,
                score=cand["score"],
                metadata=cand["metadata"],
            )
        )
    return results


class MMRRetriever(Retriever):
    """Retrieve chunks using Maximal Marginal Relevance reranking.

    Parameters
    ----------
    memory_backend:
        A ``MemoryBackend`` instance used to fetch initial candidate results.
    embedding_provider:
        An ``EmbeddingProvider`` to embed the query text.
    lambda_mult:
        Trade-off parameter between relevance and diversity.  ``1.0``
        means pure relevance; ``0.0`` means pure diversity.
    candidate_multiplier:
        Multiplier applied to *top_k* to determine how many candidates
        to fetch from the backend before MMR reranking.

    """

    def __init__(
        self,
        memory_backend: MemoryBackend,
        embedding_provider: EmbeddingProvider,
        *,
        lambda_mult: float = 0.5,
        candidate_multiplier: int = 3,
    ) -> None:
        self._backend = memory_backend
        self._embedding_provider = embedding_provider
        self._lambda_mult = lambda_mult
        self._candidate_multiplier = candidate_multiplier

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve *top_k* diverse results via MMR reranking.

        1. Embed the query.
        2. Fetch ``top_k * candidate_multiplier`` candidates from the backend.
        3. Apply MMR to select the final *top_k* results.
        """
        collector = get_metrics_collector()
        t0 = time.monotonic()
        try:
            query_embedding = await self._embed_query(query)
            raw_results = await self._fetch_candidates(query, top_k, filters)
            if not raw_results:
                collector.increment("rag_retrieval_total", labels={"strategy": "mmr", "status": "success"})
                collector.observe("rag_retrieval_results_count", 0, labels={"strategy": "mmr"})
                return []

            candidates, texts_to_embed = _results_to_candidates(raw_results)
            collector.observe("rag_retrieval_candidates_count", len(candidates))
            if texts_to_embed:
                await self._fill_missing_embeddings(candidates, texts_to_embed)

            selected = _mmr_select(candidates, query_embedding, top_k, self._lambda_mult)
            results = _candidates_to_retrieval_results(candidates, selected)

            duration_ms = (time.monotonic() - t0) * 1000
            collector.increment("rag_retrieval_total", labels={"strategy": "mmr", "status": "success"})
            collector.observe("rag_retrieval_duration_seconds", duration_ms / 1000, labels={"strategy": "mmr"})
            collector.observe("rag_retrieval_results_count", len(results), labels={"strategy": "mmr"})
            return results
        except RetrievalError:
            collector.increment("rag_retrieval_total", labels={"strategy": "mmr", "status": "failure"})
            raise

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _embed_query(self, query: str) -> list[float]:
        """Embed the query text and return the embedding vector."""
        try:
            embeddings = await self._embedding_provider.embed([query])
        except Exception as exc:
            msg = f"MMR query embedding failed: {exc}"
            raise RetrievalError(msg) from exc
        return embeddings[0]

    async def _fetch_candidates(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[MemoryResult]:
        """Fetch an over-sampled candidate set from the backend."""
        candidate_count = top_k * self._candidate_multiplier
        try:
            kwargs: dict[str, Any] = {}
            if filters:
                kwargs["metadata_filter"] = filters
            return await self._backend.search(query, top_k=candidate_count, **kwargs)
        except TypeError:
            return await self._backend.search(query, top_k=candidate_count)
        except Exception as exc:
            msg = f"MMR candidate retrieval failed: {exc}"
            raise RetrievalError(msg) from exc

    async def _fill_missing_embeddings(
        self,
        candidates: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        """Embed texts for candidates that lack stored embeddings."""
        try:
            new_embeddings = await self._embedding_provider.embed(texts)
        except Exception as exc:
            msg = f"MMR candidate embedding failed: {exc}"
            raise RetrievalError(msg) from exc

        embed_idx = 0
        for cand in candidates:
            if cand["embedding"] is None:
                cand["embedding"] = new_embeddings[embed_idx]
                embed_idx += 1
