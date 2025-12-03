"""HTML file loader using BeautifulSoup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from ia_agent_fwk.rag.exceptions import DocumentLoadError
from ia_agent_fwk.rag.loaders.base import FileLoader
from ia_agent_fwk.rag.models import Document

logger = logging.getLogger(__name__)


class HTMLLoader(FileLoader):
    """Load ``.html`` / ``.htm`` files, extracting text via BeautifulSoup.

    Requires the optional ``beautifulsoup4`` and ``lxml`` packages.
    """

    supported_extensions: ClassVar[list[str]] = [".html", ".htm"]

    async def load(self, file_path: str | Path) -> Document:
        """Read an HTML file and return a ``Document`` with extracted text."""
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            msg = "beautifulsoup4 is required for HTML loading. Install with: pip install beautifulsoup4 lxml"
            raise DocumentLoadError(msg) from exc

        path = Path(file_path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise DocumentLoadError(msg)

        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to read HTML file {path}: {exc}"
            raise DocumentLoadError(msg) from exc

        soup = BeautifulSoup(raw, "lxml")
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
        content: str = soup.get_text(separator="\n", strip=True)

        return Document(
            content=content,
            metadata={"filename": path.name, "file_size_bytes": path.stat().st_size},
            source=str(path),
            doc_type="html",
        )
