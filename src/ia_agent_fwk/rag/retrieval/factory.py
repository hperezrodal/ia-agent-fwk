"""Factory for creating retriever instances by strategy name."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ia_agent_fwk.rag.exceptions import RetrievalError
from ia_agent_fwk.rag.retrieval.mmr import MMRRetriever
from ia_agent_fwk.rag.retrieval.vector import VectorRetriever

if TYPE_CHECKING:
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
    from ia_agent_fwk.rag.retrieval.base import Retriever


class RetrieverFactory:
    """Create ``Retriever`` instances by strategy name.

    Supported strategies:

    - ``"vector"`` -- standard vector similarity search.
    - ``"mmr"`` -- Maximal Marginal Relevance reranking (requires an
      *embedding_provider*).
    """

    @classmethod
    def create(
        cls,
        strategy: str,
        memory_backend: MemoryBackend,
        embedding_provider: EmbeddingProvider | None = None,
        **kwargs: Any,
    ) -> Retriever:
        """Instantiate a retriever for the given *strategy*.

        Parameters
        ----------
        strategy:
            The retrieval strategy name (``"vector"`` or ``"mmr"``).
        memory_backend:
            Backend used for candidate retrieval.
        embedding_provider:
            Required for ``"vector"`` and ``"mmr"`` strategies.
        **kwargs:
            Extra keyword arguments forwarded to the retriever constructor
            (e.g. ``lambda_mult`` for MMR).

        Raises
        ------
        RetrievalError
            If *strategy* is unknown or a required argument is missing.

        """
        if strategy == "vector":
            if embedding_provider is None:
                msg = "Vector retriever requires an embedding_provider"
                raise RetrievalError(msg)
            return VectorRetriever(
                backend=memory_backend,
                embedding_provider=embedding_provider,
            )

        if strategy == "mmr":
            if embedding_provider is None:
                msg = "MMR retriever requires an embedding_provider"
                raise RetrievalError(msg)
            return MMRRetriever(
                memory_backend=memory_backend,
                embedding_provider=embedding_provider,
                **kwargs,
            )

        msg = f"Unknown retrieval strategy: {strategy}"
        raise RetrievalError(msg)
