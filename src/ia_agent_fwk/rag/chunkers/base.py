"""Abstract base class for text chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ia_agent_fwk.rag.models import Chunk, Document


class Chunker(ABC):
    """Abstract base class for text chunking strategies.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.

    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into a list of ``Chunk`` objects."""
        ...
