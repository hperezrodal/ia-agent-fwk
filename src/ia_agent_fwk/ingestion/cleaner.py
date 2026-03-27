"""Generic document cleaning pipeline by composition.

Each cleaning stage is a pure function: str → str.
Stages are composable and can be enabled/disabled independently.

Stages:
  1. normalize_unicode   — NFC, PUA chars → standard, ligatures → expanded
  2. fix_encoding        — NBSP, soft hyphens, zero-width, control chars, BOM
  3. strip_artifacts     — <!-- image -->, \x00, parser-specific noise
  4. remove_boilerplate  — page numbers, repeated headers/footers, standalone URLs
  5. normalize_whitespace — collapse spaces, trailing whitespace, blank lines

Usage:
    cleaner = DocumentCleaner()
    clean_text = cleaner.clean("raw text from any parser")
    clean_md   = cleaner.clean("markdown", preserve_tables=True)

Each stage can also be called independently:
    from ia_agent_fwk.ingestion.cleaner import normalize_unicode, fix_encoding
    text = fix_encoding(normalize_unicode(raw))
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Unicode normalization
# ═══════════════════════════════════════════════════════════════════════════

# Private Use Area (PUA) chars commonly found in PDFs → standard equivalents
_PUA_MAP: dict[str, str] = {
    "\uf0fc": "\u2022",  # bullet
    "\uf0d8": "\u2022",  # diamond bullet → •
    "\uf0b7": "\u2022",  # bullet variant → •
    "\uf0a7": "\u00a7",  # section sign §
    "\uf0e0": "\u2192",  # arrow →
    "\uf076": "v",
    "\uf06e": "n",
}

# Ligatures → expanded form
_LIGATURE_MAP: dict[str, str] = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def normalize_unicode(text: str) -> str:
    """NFC normalization + PUA replacement + ligature expansion."""
    text = unicodedata.normalize("NFC", text)
    for pua, repl in _PUA_MAP.items():
        text = text.replace(pua, repl)
    for lig, expanded in _LIGATURE_MAP.items():
        text = text.replace(lig, expanded)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Encoding fixes
# ═══════════════════════════════════════════════════════════════════════════

# Control characters to strip (keep \n, \t, \r)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def fix_encoding(text: str) -> str:
    """Fix common encoding issues from PDF extraction."""
    text = text.replace("\u00a0", " ")  # NBSP → regular space
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u200c", "")  # zero-width non-joiner
    text = text.replace("\u200d", "")  # zero-width joiner
    text = text.replace("\ufeff", "")  # BOM
    return _CONTROL_CHARS.sub("", text)  # control characters


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: Parser artifact removal
# ═══════════════════════════════════════════════════════════════════════════

# Patterns to strip from any parser output
_ARTIFACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<!--\s*image\s*-->", re.IGNORECASE),  # Docling image markers
    re.compile(r"<!\-\-.*?\-\->"),  # any HTML comments
]


def strip_artifacts(text: str) -> str:
    """Remove parser-specific artifacts (HTML comments, image markers, etc.)."""
    for pat in _ARTIFACT_PATTERNS:
        text = pat.sub("", text)
    return text


def deduplicate_consecutive_lines(text: str) -> str:
    """Remove consecutive duplicate lines (common in Docling output).

    Keeps the first occurrence. Only deduplicates lines > 20 chars
    to avoid removing intentional short repeats.
    """
    lines = text.split("\n")
    result: list[str] = []
    prev = ""
    for line in lines:
        stripped = line.strip()
        if stripped == prev and len(stripped) > 20:
            continue
        result.append(line)
        prev = stripped
    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: Boilerplate removal
# ═══════════════════════════════════════════════════════════════════════════

_BOILERPLATE_LINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*\d+\s*$"),  # standalone page numbers
    re.compile(r"^\s*pág\.?\s*\d+\s*$", re.IGNORECASE),  # "pág. 3"
    re.compile(r"^\s*página\s+\d+\s+de\s+\d+\s*$", re.IGNORECASE),  # "Página 3 de 21"
    re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),  # "Page 3 of 21"
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),  # "- 3 -"
    re.compile(r"^\s*www\.\S+\.\S+\s*$", re.IGNORECASE),  # standalone URLs
]


def is_boilerplate_line(line: str) -> bool:
    """Check if a single line is a known boilerplate pattern."""
    return any(pat.match(line) for pat in _BOILERPLATE_LINE_PATTERNS)


def detect_repeated_lines(
    pages: list[str],
    threshold: float = 0.5,
    n_lines: int = 3,
) -> set[str]:
    """Detect lines that repeat across many pages (headers/footers).

    Checks the first and last *n_lines* of each page. Lines appearing in
    more than *threshold* fraction of pages are considered boilerplate.

    Requires a list of page strings. If the parser doesn't provide pages
    (e.g. Docling outputs a single markdown string), this step is skipped.
    """
    if len(pages) < 3:
        return set()

    counter: Counter[str] = Counter()
    for page_text in pages:
        lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
        if not lines:
            continue
        candidates = set()
        for ln in lines[:n_lines]:
            candidates.add(ln)
        for ln in lines[-n_lines:]:
            candidates.add(ln)
        for ln in candidates:
            counter[ln] += 1

    min_count = len(pages) * threshold
    return {ln for ln, count in counter.items() if count >= min_count}


def remove_boilerplate(text: str, repeated_lines: set[str] | None = None) -> str:
    """Remove boilerplate lines from text.

    Removes:
    - Lines matching known patterns (page numbers, URLs)
    - Lines in the repeated_lines set (detected cross-page headers/footers)
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if repeated_lines and stripped in repeated_lines:
            continue
        if is_boilerplate_line(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: Whitespace normalization
# ═══════════════════════════════════════════════════════════════════════════

_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def normalize_whitespace(text: str, preserve_tables: bool = False) -> str:
    """Collapse excessive whitespace and blank lines.

    Parameters
    ----------
    preserve_tables:
        If True, don't collapse spaces inside markdown table rows (|...|).
        Useful for markdown output where table alignment matters.

    """
    if preserve_tables:
        # Process line by line, skip table rows
        lines: list[str] = []
        for line in text.split("\n"):
            if _TABLE_ROW.match(line):
                lines.append(line.rstrip())
            else:
                lines.append(_MULTI_SPACE.sub(" ", line).rstrip())
        text = "\n".join(lines)
    else:
        text = _MULTI_SPACE.sub(" ", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))

    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Language detection (utility, not a cleaning stage)
# ═══════════════════════════════════════════════════════════════════════════


def detect_language(text: str, default: str = "es") -> str:
    """Detect the language of a text block.

    Returns ISO 639-1 code (e.g. 'es', 'en').
    Falls back to *default* if detection fails or text is too short.
    """
    try:
        from langdetect import detect  # noqa: PLC0415

        if len(text.strip()) < 20:
            return default
        return detect(text)
    except ImportError:
        return default
    except Exception:  # noqa: BLE001
        return default


# ═══════════════════════════════════════════════════════════════════════════
# Composed pipeline
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CleaningConfig:
    """Configuration for the cleaning pipeline.

    Each flag enables/disables a stage. All enabled by default.
    """

    normalize_unicode_enabled: bool = True
    fix_encoding_enabled: bool = True
    strip_artifacts_enabled: bool = True
    remove_boilerplate_enabled: bool = True
    normalize_whitespace_enabled: bool = True


class DocumentCleaner:
    """Generic document cleaning pipeline.

    Composes all stages in sequence. Parser-agnostic.

    Usage::

        cleaner = DocumentCleaner()

        # Full cleaning (plain text)
        clean = cleaner.clean(raw_text)

        # Markdown-aware (preserves table spacing)
        clean = cleaner.clean(markdown, preserve_tables=True)

        # With page-level boilerplate detection
        clean = cleaner.clean_pages(pages)
    """

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self._config = config or CleaningConfig()

    def clean(self, text: str, *, preserve_tables: bool = False) -> str:
        """Clean a text string through all enabled stages.

        Parameters
        ----------
        text:
            Raw text from any parser.
        preserve_tables:
            If True, don't collapse spaces inside markdown table rows.

        """
        if self._config.normalize_unicode_enabled:
            text = normalize_unicode(text)
        if self._config.fix_encoding_enabled:
            text = fix_encoding(text)
        if self._config.strip_artifacts_enabled:
            text = strip_artifacts(text)
            text = deduplicate_consecutive_lines(text)
        if self._config.remove_boilerplate_enabled:
            text = remove_boilerplate(text)
        if self._config.normalize_whitespace_enabled:
            text = normalize_whitespace(text, preserve_tables=preserve_tables)
        return text

    def clean_pages(self, pages: list[str]) -> list[str]:
        """Clean a list of pages with cross-page boilerplate detection.

        1. Clean each page individually
        2. Detect repeated lines across pages (headers/footers)
        3. Remove repeated lines
        4. Re-normalize whitespace
        """
        # Individual cleaning
        cleaned = [self.clean(p) for p in pages]

        # Cross-page boilerplate detection
        if self._config.remove_boilerplate_enabled and len(cleaned) >= 3:
            repeated = detect_repeated_lines(cleaned)
            if repeated:
                cleaned = [remove_boilerplate(p, repeated) for p in cleaned]
                # Re-normalize after removal
                if self._config.normalize_whitespace_enabled:
                    cleaned = [normalize_whitespace(p) for p in cleaned]

        return cleaned

    # Backwards-compatible aliases
    def clean_text(self, text: str) -> str:
        """Alias for clean(text, preserve_tables=False)."""
        return self.clean(text, preserve_tables=False)

    def clean_markdown(self, markdown: str) -> str:
        """Alias for clean(text, preserve_tables=True)."""
        return self.clean(markdown, preserve_tables=True)
