"""Plain-text file loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from ia_agent_fwk.rag.exceptions import DocumentLoadError
from ia_agent_fwk.rag.loaders.base import FileLoader
from ia_agent_fwk.rag.models import Document

logger = logging.getLogger(__name__)


class TextLoader(FileLoader):
    """Load ``.txt`` files as plain text."""

    supported_extensions: ClassVar[list[str]] = [".txt"]

    async def load(self, file_path: str | Path) -> Document:
        """Read a text file and return a ``Document``."""
        path = Path(file_path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise DocumentLoadError(msg)

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to read text file {path}: {exc}"
            raise DocumentLoadError(msg) from exc

        return Document(
            content=content,
            metadata={"filename": path.name, "file_size_bytes": path.stat().st_size},
            source=str(path),
            doc_type="text",
        )
