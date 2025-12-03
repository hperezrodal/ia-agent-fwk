"""Abstract base class for document file loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ia_agent_fwk.rag.models import Document


class FileLoader(ABC):
    """Abstract base class for document file loaders.

    Subclasses declare which file extensions they support via the
    ``supported_extensions`` class variable and implement the async
    ``load`` method to parse the file into a ``Document``.
    """

    supported_extensions: ClassVar[list[str]]

    @abstractmethod
    async def load(self, file_path: str | Path) -> Document:
        """Load a document from *file_path* and return a ``Document``."""
        ...

    def can_load(self, file_path: str | Path) -> bool:
        """Return ``True`` if this loader supports the given file extension."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions
