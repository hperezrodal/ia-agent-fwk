"""Markdown file loader."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import ClassVar

from ia_agent_fwk.rag.exceptions import DocumentLoadError
from ia_agent_fwk.rag.loaders.base import FileLoader
from ia_agent_fwk.rag.models import Document

logger = logging.getLogger(__name__)

# Patterns for stripping common Markdown formatting to plain text.
_MD_PATTERNS: list[tuple[str, str]] = [
    (r"^#{1,6}\s+", ""),  # headings
    (r"\*\*(.+?)\*\*", r"\1"),  # bold
    (r"\*(.+?)\*", r"\1"),  # italic
    (r"`(.+?)`", r"\1"),  # inline code
    (r"!\[.*?\]\(.*?\)", ""),  # images
    (r"\[(.+?)\]\(.*?\)", r"\1"),  # links -> keep text
]


def _strip_markdown(text: str) -> str:
    """Remove common Markdown formatting, returning plain text."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        cleaned = raw_line
        for pattern, replacement in _MD_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)
        lines.append(cleaned)
    return "\n".join(lines)


class MarkdownLoader(FileLoader):
    """Load ``.md`` files, stripping Markdown formatting to plain text."""

    supported_extensions: ClassVar[list[str]] = [".md"]

    async def load(self, file_path: str | Path) -> Document:
        """Read a Markdown file and return a ``Document``."""
        path = Path(file_path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise DocumentLoadError(msg)

        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to read Markdown file {path}: {exc}"
            raise DocumentLoadError(msg) from exc

        content = _strip_markdown(raw)

        return Document(
            content=content,
            metadata={"filename": path.name, "file_size_bytes": path.stat().st_size},
            source=str(path),
            doc_type="markdown",
        )
