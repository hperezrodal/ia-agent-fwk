"""Filtered retriever decorator that applies metadata pre-filters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ia_agent_fwk.rag.retrieval.base import Retriever

if TYPE_CHECKING:
    from ia_agent_fwk.rag.models import RetrievalResult


class FilteredRetriever(Retriever):
    """Wrap any retriever with metadata pre-filtering.

    Constructor-level ``metadata_filters`` are merged with any call-time
    *filters*.  Call-time values take precedence when keys overlap.

    Parameters
    ----------
    inner:
        The underlying ``Retriever`` to delegate to.
    metadata_filters:
        Default metadata filters applied to every ``retrieve`` call.

    """

    def __init__(self, inner: Retriever, metadata_filters: dict[str, Any]) -> None:
        self._inner = inner
        self._metadata_filters = metadata_filters

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Merge constructor filters with call-time filters and delegate."""
        merged = {**self._metadata_filters, **(filters or {})}
        return await self._inner.retrieve(query, top_k=top_k, filters=merged)
