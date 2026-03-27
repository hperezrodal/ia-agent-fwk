"""TRANSFORM stage — type-aware block transformations.

Takes blocks from SPLIT and transforms them based on type:
- Headings → tracked as section context, prepended to subsequent blocks
- Tables → hierarchical parent + children (per-row as natural text)
- Lists → kept as-is (good chunk units naturally)
- Text → kept as-is
"""

from __future__ import annotations

import uuid

from ia_agent_fwk.ingestion.chunking.models import Block, BlockType


def transform_blocks(blocks: list[Block]) -> list[Block]:
    """Apply type-aware transformations to blocks.

    - Headings are consumed as section context and prepended to following blocks.
    - Tables become parent (full table) + children (per-row natural text).
    - Text and lists pass through with section prefix.
    """
    result: list[Block] = []
    current_section: str | None = None
    prev_was_heading = False

    for block in blocks:
        if block.block_type == BlockType.HEADING:
            current_section = block.content
            # Keep heading as a small text block (searchable by BM25)
            result.append(
                Block(
                    content=block.content,
                    block_type=BlockType.TEXT,
                    metadata={**block.metadata, "section": current_section},
                )
            )
            prev_was_heading = True
            continue

        # Add section to metadata
        meta = {**block.metadata}
        if current_section:
            meta["section"] = current_section

        # Don't prepend section prefix if the previous block was a heading
        # (they'll merge naturally in the SIZE stage, avoiding duplication)
        section_prefix = ""
        if current_section and not prev_was_heading:
            section_prefix = f"{current_section}\n\n"
        prev_was_heading = False

        if block.block_type == BlockType.TABLE:
            # Hierarchical: parent + children
            table_blocks = _transform_table(block.content, current_section, meta)
            result.extend(table_blocks)

        elif block.block_type == BlockType.LIST:
            result.append(
                Block(
                    content=f"{section_prefix}{block.content}" if section_prefix else block.content,
                    block_type=BlockType.TEXT,
                    metadata=meta,
                )
            )

        else:
            result.append(
                Block(
                    content=f"{section_prefix}{block.content}" if section_prefix else block.content,
                    block_type=BlockType.TEXT,
                    metadata=meta,
                )
            )

    return result


def _transform_table(
    table_md: str,
    section: str | None,
    base_meta: dict,
) -> list[Block]:
    """Transform a markdown table into parent + child blocks."""
    table_id = str(uuid.uuid4())[:8]
    section_prefix = f"{section}\n\n" if section else ""
    blocks: list[Block] = []

    # Parent: full table
    blocks.append(
        Block(
            content=f"{section_prefix}{table_md}",
            block_type=BlockType.TABLE_PARENT,
            metadata={
                **base_meta,
                "table_role": "parent",
                "table_id": table_id,
            },
        )
    )

    # Parse header and rows
    lines = table_md.split("\n")
    if len(lines) < 3:
        return blocks

    header = lines[0]
    separator_idx = 1 if "---" in lines[1] else -1
    data_start = separator_idx + 1 if separator_idx >= 0 else 1
    data_rows = [r for r in lines[data_start:] if r.strip() and r.strip().startswith("|")]

    if not data_rows:
        return blocks

    # Extract column names from header
    col_names = [c.strip() for c in header.split("|") if c.strip()]

    # Children: each row as natural text
    for row in data_rows:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if not cells:
            continue

        # Build natural text from cells + column names
        parts: list[str] = []
        seen_cells: set[str] = set()
        used_col_prefix: set[str] = set()
        for j, cell in enumerate(cells):
            if not cell or cell.startswith("---"):
                continue
            # Skip duplicate cells (Docling sometimes duplicates columns)
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            # Add column name as prefix, but only once per unique column name
            col = col_names[j] if j < len(col_names) else ""
            if col and col != cell and col not in used_col_prefix:
                parts.append(f"{col}: {cell}")
                used_col_prefix.add(col)
            else:
                parts.append(cell)

        child_text = ". ".join(parts)
        if not child_text.strip():
            continue

        blocks.append(
            Block(
                content=f"{section_prefix}{child_text}" if section_prefix else child_text,
                block_type=BlockType.TABLE_CHILD,
                metadata={
                    **base_meta,
                    "table_role": "child",
                    "table_id": table_id,
                    "parent_chunk_id": table_id,
                },
            )
        )

    return blocks
