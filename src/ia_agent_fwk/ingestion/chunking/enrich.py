"""ENRICH stage — attach final metadata to each block.

Adds consistent metadata fields needed for retrieval:
- chunk_type (text, table:parent, table:child)
- section (from heading tracker in TRANSFORM)
- Sequential indexing
- Filters empty blocks
"""

from __future__ import annotations

from ia_agent_fwk.ingestion.chunking.models import Block, BlockType


def enrich_blocks(blocks: list[Block]) -> list[Block]:
    """Finalize blocks with consistent metadata and indexing.

    - Filters out empty blocks
    - Normalizes chunk_type in metadata
    - Adds sequential chunk_index
    """
    result: list[Block] = []
    idx = 0

    for block in blocks:
        content = block.content.strip()
        if not content:
            continue

        # Normalize chunk_type for downstream consumers
        if block.block_type == BlockType.TABLE_PARENT:
            chunk_type = "table"
            table_role = "parent"
        elif block.block_type == BlockType.TABLE_CHILD:
            chunk_type = "table"
            table_role = "child"
        else:
            chunk_type = "text"
            table_role = ""

        meta = {**block.metadata, "chunk_type": chunk_type, "chunk_index": idx}
        if table_role:
            meta["table_role"] = table_role

        result.append(
            Block(
                content=content,
                block_type=block.block_type,
                metadata=meta,
            )
        )
        idx += 1

    return result
