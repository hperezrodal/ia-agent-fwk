"""PDF file loader using pypdf."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from ia_agent_fwk.rag.exceptions import DocumentLoadError
from ia_agent_fwk.rag.loaders.base import FileLoader
from ia_agent_fwk.rag.models import Document

logger = logging.getLogger(__name__)


class PDFLoader(FileLoader):
    """Load ``.pdf`` files using ``pypdf``.

    Requires the optional ``pypdf`` package.
    """

    supported_extensions: ClassVar[list[str]] = [".pdf"]

    async def load(self, file_path: str | Path) -> Document:
        """Read a PDF file and return a ``Document`` with extracted text."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            msg = "pypdf is required for PDF loading. Install with: pip install pypdf"
            raise DocumentLoadError(msg) from exc

        path = Path(file_path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise DocumentLoadError(msg)

        try:
            reader = PdfReader(str(path))
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            content = "\n\n".join(pages)
        except DocumentLoadError:
            raise
        except Exception as exc:
            msg = f"Failed to read PDF file {path}: {exc}"
            raise DocumentLoadError(msg) from exc

        return Document(
            content=content,
            metadata={
                "filename": path.name,
                "file_size_bytes": path.stat().st_size,
                "page_count": len(pages),
            },
            source=str(path),
            doc_type="pdf",
        )
