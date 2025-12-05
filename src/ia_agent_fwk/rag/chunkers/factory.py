"""Chunker factory for configuration-driven strategy selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ia_agent_fwk.rag.chunkers.fixed import FixedSizeChunker
from ia_agent_fwk.rag.chunkers.recursive import RecursiveChunker
from ia_agent_fwk.rag.chunkers.semantic import SemanticChunker
from ia_agent_fwk.rag.exceptions import ChunkingError

if TYPE_CHECKING:
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
    from ia_agent_fwk.rag.chunkers.base import Chunker

_STRATEGIES: dict[str, type[Chunker]] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
}


class ChunkerFactory:
    """Create a ``Chunker`` from a strategy name and parameters."""

    @staticmethod
    def create(
        strategy: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        **kwargs: Any,
    ) -> Chunker:
        """Instantiate and return the chunker for *strategy*.

        Parameters
        ----------
        strategy:
            One of ``"fixed"``, ``"recursive"``, or ``"semantic"``.
        chunk_size:
            Maximum characters per chunk (used by fixed/recursive).
        chunk_overlap:
            Overlap characters between consecutive chunks (used by
            fixed/recursive).
        embedding_provider:
            An ``EmbeddingProvider`` instance, **required** when
            *strategy* is ``"semantic"``.
        **kwargs:
            Additional keyword arguments forwarded to the
            ``SemanticChunker`` constructor (e.g.
            ``similarity_threshold``, ``min_chunk_size``,
            ``max_chunk_size``).

        """
        if strategy == "semantic":
            if embedding_provider is None:
                msg = (
                    "SemanticChunker requires an embedding_provider. "
                    "Pass embedding_provider=... to ChunkerFactory.create()."
                )
                raise ChunkingError(msg)
            return SemanticChunker(embedding_provider, **kwargs)

        cls = _STRATEGIES.get(strategy)
        if cls is None:
            valid = ", ".join(sorted([*_STRATEGIES, "semantic"]))
            msg = f"Unknown chunking strategy '{strategy}'. Valid strategies: {valid}"
            raise ChunkingError(msg)
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
