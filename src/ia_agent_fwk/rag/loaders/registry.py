"""Loader registry that maps file extensions to loader classes."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ia_agent_fwk.rag.exceptions import DocumentLoadError

if TYPE_CHECKING:
    from ia_agent_fwk.rag.loaders.base import FileLoader
    from ia_agent_fwk.rag.models import Document

logger = logging.getLogger(__name__)


class LoaderRegistry:
    """Registry mapping file extensions to ``FileLoader`` instances.

    Built-in loaders for ``.txt``, ``.md``, ``.html``, ``.htm``, and
    ``.pdf`` are auto-registered on first access.
    """

    _default_map: ClassVar[dict[str, str]] = {
        ".txt": "ia_agent_fwk.rag.loaders.text:TextLoader",
        ".md": "ia_agent_fwk.rag.loaders.markdown:MarkdownLoader",
        ".html": "ia_agent_fwk.rag.loaders.html:HTMLLoader",
        ".htm": "ia_agent_fwk.rag.loaders.html:HTMLLoader",
        ".pdf": "ia_agent_fwk.rag.loaders.pdf:PDFLoader",
        ".docx": "ia_agent_fwk.rag.loaders.docx:DOCXLoader",
    }

    def __init__(self) -> None:
        self._registry: dict[str, FileLoader] = {}
        self._lazy_registry: dict[str, str] = dict(self._default_map)

    def register(self, extension: str, loader: FileLoader | type[FileLoader]) -> None:
        """Register a loader for *extension* (e.g. ``".csv"``)."""
        ext = extension.lower()
        if isinstance(loader, type):
            loader = loader()
        self._registry[ext] = loader

    def get_loader(self, file_path: str | Path) -> FileLoader:
        """Return the appropriate loader for *file_path* based on its extension."""
        ext = Path(file_path).suffix.lower()

        if ext in self._registry:
            return self._registry[ext]

        if ext in self._lazy_registry:
            loader = self._resolve_lazy(ext)
            self._registry[ext] = loader
            return loader

        msg = f"No loader registered for extension '{ext}'"
        raise DocumentLoadError(msg)

    async def load(self, file_path: str | Path) -> Document:
        """Get the loader for *file_path* and load the document."""
        loader = self.get_loader(file_path)
        return await loader.load(file_path)

    def _resolve_lazy(self, ext: str) -> FileLoader:
        """Resolve a lazy dotted-path entry to a loader instance."""
        dotted = self._lazy_registry[ext]
        module_path, _, attr = dotted.rpartition(":")
        mod = importlib.import_module(module_path)
        cls: type[FileLoader] = getattr(mod, attr)
        return cls()
