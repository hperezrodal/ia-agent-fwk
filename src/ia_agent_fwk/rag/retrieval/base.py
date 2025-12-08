"""Abstract base class for retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ia_agent_fwk.rag.models import RetrievalResult


class Retriever(ABC):
    """Abstract base class for retrieval strategies.

    Subclasses implement ``retrieve`` to return the most relevant chunks
    for a given query string.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for *query*.

        Parameters
        ----------
        query:
            The search query text.
        top_k:
            Maximum number of results to return.
        filters:
            Optional metadata filters to narrow the search scope.

        """
        ...
