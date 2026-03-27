"""Document parsers — pure functions: Path → str.

Each parser takes a file path and returns a string (markdown or plain text).
The parser is the ONLY component that varies by format/tool.
Cleaning and chunking are handled downstream by generic stages.

Registry pattern: _PARSERS maps (extension, parser_name) → function.
The orchestrator picks the right parser based on file extension and config.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Docling parser (ML-based, high quality)
# ═══════════════════════════════════════════════════════════════════════════

_docling_converter = None  # module-level lazy singleton


def parse_docling(file_path: Path) -> str:
    """Parse a document with Docling → structured markdown.

    Handles: tables (with/without borders), multi-column, OCR, complex layouts.
    Output: markdown with ## headings, | tables |, - lists.
    Speed: ~3 sec/page on CPU.
    Requires: docling (PyTorch).
    """
    global _docling_converter  # noqa: PLW0603
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter  # noqa: PLC0415

        _docling_converter = DocumentConverter()

    result = _docling_converter.convert(str(file_path))
    return result.document.export_to_markdown()


# ═══════════════════════════════════════════════════════════════════════════
# PyMuPDF parser (fast, lightweight)
# ═══════════════════════════════════════════════════════════════════════════


def parse_pymupdf(file_path: Path) -> str:
    """Parse a PDF with pymupdf → plain text.

    Handles: text extraction, basic layout. No table detection.
    Output: plain text (no markdown structure).
    Speed: ~7 pages/sec.
    Requires: pymupdf.
    """
    import pymupdf  # noqa: PLC0415

    doc = pymupdf.open(str(file_path))
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text") or ""
        pages.append(text)
    doc.close()
    return "\n\n".join(pages)


# ═══════════════════════════════════════════════════════════════════════════
# DOCX parser
# ═══════════════════════════════════════════════════════════════════════════


def parse_docx(file_path: Path) -> str:
    """Parse a DOCX file → plain text.

    Requires: python-docx.
    """
    try:
        import docx  # noqa: PLC0415

        doc = docx.Document(str(file_path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.warning("python-docx not installed, reading as raw text")
        return file_path.read_text(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════════
# Plain text / markdown parser (passthrough)
# ═══════════════════════════════════════════════════════════════════════════


def parse_text(file_path: Path) -> str:
    """Read a plain text or markdown file as-is."""
    return file_path.read_text(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════════
# OCR parser (scanned documents)
# ═══════════════════════════════════════════════════════════════════════════


def parse_ocr(file_path: Path, *, language: str = "eng") -> str:
    """Parse a scanned PDF with OCR → plain text.

    Parameters
    ----------
    language:
        Tesseract language code (e.g. "eng", "spa", "fra").

    Requires: pytesseract, pdf2image.

    """
    import pytesseract  # noqa: PLC0415
    from pdf2image import convert_from_path  # noqa: PLC0415

    images = convert_from_path(str(file_path), dpi=300)
    pages: list[str] = []
    for img in images:
        text = pytesseract.image_to_string(img, lang=language)
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


# ═══════════════════════════════════════════════════════════════════════════
# Parser registry
# ═══════════════════════════════════════════════════════════════════════════

# Maps parser name → function
PARSER_REGISTRY: dict[str, Callable[..., str]] = {
    "docling": parse_docling,
    "pymupdf": parse_pymupdf,
    "docx": parse_docx,
    "text": parse_text,
    "ocr": parse_ocr,
}

# Default parser per file extension
DEFAULT_PARSER: dict[str, str] = {
    ".pdf": "docling",
    ".docx": "docx",
    ".txt": "text",
    ".md": "text",
    ".html": "text",
    ".htm": "text",
}

# Fallback chain: if preferred parser fails, try these
FALLBACK_CHAIN: dict[str, list[str]] = {
    "docling": ["pymupdf"],
    "pymupdf": [],
    "ocr": ["pymupdf"],
    "docx": ["text"],
    "text": [],
}


def parse(
    file_path: str | Path,
    parser_name: str | None = None,
    *,
    save_to: str | Path | None = None,
) -> tuple[str, str]:
    """Parse a document with the appropriate parser.

    Parameters
    ----------
    file_path:
        Path to the document.
    parser_name:
        Force a specific parser. If None, uses DEFAULT_PARSER by extension.
    save_to:
        If provided, save the parsed output to this path (markdown/text).
        Useful for debugging or to resume ingestion without re-parsing.

    Returns
    -------
    tuple[str, str]
        (parsed_text, parser_name_used)

    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Determine parser
    name = parser_name or DEFAULT_PARSER.get(ext, "text")
    chain = [name, *FALLBACK_CHAIN.get(name, [])]

    for parser in chain:
        fn = PARSER_REGISTRY.get(parser)
        if fn is None:
            continue
        try:
            logger.info("Parsing %s with %s", path.name, parser)
            text = fn(path)
            if text and text.strip():
                if save_to:
                    _save_parsed(text, parser, path, save_to)
                return text, parser
            logger.warning("%s returned empty for %s, trying fallback", parser, path.name)
        except ImportError as exc:
            logger.warning("%s not available: %s, trying fallback", parser, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed for %s: %s, trying fallback", parser, path.name, exc)

    # Last resort: raw text
    logger.warning("All parsers failed for %s, reading as raw text", path.name)
    return path.read_text(encoding="utf-8", errors="replace"), "raw"


def load_parsed(parsed_path: str | Path) -> tuple[str, str]:
    """Load a previously saved parsed file. Skips parse + clean.

    The parser name is read from the first line (comment header).

    Returns
    -------
    tuple[str, str]
        (parsed_text, parser_name)

    """
    path = Path(parsed_path)
    content = path.read_text(encoding="utf-8")

    # Extract parser name from header if present
    parser_name = "unknown"
    if content.startswith("<!-- parser:"):
        first_line, _, rest = content.partition("\n")
        # Header format: <!-- parser:docling source:AUTOS.pdf -->
        parts = first_line.lstrip("<!- ").rstrip(" ->").split()
        for part in parts:
            if part.startswith("parser:"):
                parser_name = part.split(":", 1)[1]
                break
        content = rest

    logger.info("Loaded pre-parsed file %s (parser=%s)", path.name, parser_name)
    return content, parser_name


def _save_parsed(text: str, parser_name: str, source: Path, save_to: str | Path) -> None:
    """Save parsed output to disk with metadata header."""
    out = Path(save_to)
    if out.is_dir():
        ext = ".md" if parser_name == "docling" else ".txt"
        out = out / f"{source.stem}.parsed{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)

    header = f"<!-- parser:{parser_name} source:{source.name} -->\n"
    out.write_text(header + text, encoding="utf-8")
    logger.info("Saved parsed output → %s", out)
