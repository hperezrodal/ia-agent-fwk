"""SPLIT stage — divide document into semantic blocks.

Takes a raw document string (markdown or plain text) and splits it into
typed blocks: headings, tables, lists, and text paragraphs.

Parser-agnostic: operates on the string output of any parser.
"""

from __future__ import annotations

import re

from ia_agent_fwk.ingestion.chunking.models import Block, BlockType

# Regex for markdown headings
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Regex for markdown list items
_LIST_ITEM_RE = re.compile(r"^(\s*[-*+]|\s*\d+[.\)])\s+")


def split_blocks(text: str) -> list[Block]:
    """Split a document string into typed semantic blocks.

    Detects:
    - **Headings**: lines starting with # (markdown) or ALL CAPS lines
    - **Tables**: consecutive lines starting and ending with |
    - **Lists**: consecutive lines starting with - * + or 1. 2. etc.
    - **Text**: everything else (paragraphs separated by blank lines)
    """
    blocks: list[Block] = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        # Heading?
        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            blocks.append(
                Block(
                    content=heading_text,
                    block_type=BlockType.HEADING,
                    metadata={"heading_level": level},
                )
            )
            i += 1
            continue

        # All-caps heading? (common in PDFs without markdown)
        if stripped.isupper() and 3 < len(stripped) < 80 and not stripped.startswith("|"):
            blocks.append(
                Block(
                    content=stripped,
                    block_type=BlockType.HEADING,
                    metadata={"heading_level": 2},
                )
            )
            i += 1
            continue

        # Table? (consecutive | lines)
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines: list[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith("|") and s.endswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                elif not s:
                    i += 1  # skip blank lines within table
                    # Check if next non-blank line is still table
                    if i < len(lines) and lines[i].strip().startswith("|"):
                        continue
                    break
                else:
                    break
            blocks.append(
                Block(
                    content="\n".join(table_lines),
                    block_type=BlockType.TABLE,
                )
            )
            continue

        # List? (consecutive list items)
        if _LIST_ITEM_RE.match(stripped):
            list_lines: list[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if _LIST_ITEM_RE.match(s) or (s and not s.startswith("#") and list_lines):
                    # Continuation of list (including wrapped lines)
                    list_lines.append(lines[i])
                    i += 1
                    if not s:  # blank line ends list
                        break
                else:
                    break
            blocks.append(
                Block(
                    content="\n".join(list_lines),
                    block_type=BlockType.LIST,
                )
            )
            continue

        # Text paragraph (collect until blank line or type change)
        para_lines: list[str] = []
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                i += 1
                break
            # Stop if next line is heading, table, or list
            if _HEADING_RE.match(s):
                break
            if s.startswith("|") and s.endswith("|"):
                break
            if _LIST_ITEM_RE.match(s) and not para_lines:
                break
            if s.isupper() and 3 < len(s) < 80 and not s.startswith("|"):
                break
            para_lines.append(lines[i])
            i += 1

        if para_lines:
            blocks.append(
                Block(
                    content="\n".join(para_lines),
                    block_type=BlockType.TEXT,
                )
            )

    return blocks
