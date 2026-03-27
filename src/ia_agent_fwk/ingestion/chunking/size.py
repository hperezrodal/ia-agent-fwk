"""SIZE stage — adjust block sizes (split large, merge small).

Operates on blocks from TRANSFORM:
- Blocks larger than chunk_size get split (paragraph → sentence → word)
- Blocks smaller than min_chunk_size get merged with neighbors
- Table parents and children are NEVER merged with other blocks
- Overlap is added between consecutive text splits

This stage is the only one that cares about chunk_size.
"""

from __future__ import annotations

from ia_agent_fwk.ingestion.chunking.models import Block, BlockType

# Table types should never be merged
_TABLE_TYPES = {BlockType.TABLE_PARENT, BlockType.TABLE_CHILD}


def resize_blocks(
    blocks: list[Block],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 200,
) -> list[Block]:
    """Split large blocks and merge small ones.

    Parameters
    ----------
    chunk_size:
        Maximum characters per block.
    chunk_overlap:
        Overlap characters between consecutive splits of the same block.
    min_chunk_size:
        Blocks smaller than this are merged with neighbors.

    """
    # Step 1: split oversized blocks
    split_result: list[Block] = []
    for block in blocks:
        if block.block_type in _TABLE_TYPES:
            # Tables: keep as-is (parent can be large, children are per-row)
            split_result.append(block)
        elif block.char_count > chunk_size:
            split_result.extend(_split_block(block, chunk_size, chunk_overlap))
        else:
            split_result.append(block)

    # Step 2: merge undersized blocks
    merged = _merge_small_blocks(split_result, chunk_size, min_chunk_size)

    # Step 3: filter out chunks that are still too small after merge
    # (e.g. a lone heading "Plus" that had no neighbors to merge with)
    min_viable = 30  # below this, a chunk has no retrieval value
    return [b for b in merged if b.char_count >= min_viable or b.block_type in _TABLE_TYPES]


def _split_block(
    block: Block,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Block]:
    """Split a large text block into smaller ones with overlap."""
    text = block.content
    chunks = _recursive_split(text, chunk_size)

    # Apply overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap, chunk_size)

    return [block.copy(content=c) for c in chunks if c.strip()]


def _recursive_split(text: str, max_size: int) -> list[str]:
    """Split text recursively: paragraphs → sentences → words → chars."""
    if len(text) <= max_size:
        return [text]

    # Try separators in order of preference
    for sep in ["\n\n", "\n", ". ", " "]:
        if sep in text:
            parts = text.split(sep)
            merged: list[str] = []
            current = ""
            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= max_size:
                    current = candidate
                else:
                    if current:
                        merged.append(current)
                    if len(part) > max_size:
                        merged.extend(_recursive_split(part, max_size))
                        current = ""
                    else:
                        current = part
            if current:
                merged.append(current)
            if len(merged) > 1 or (merged and len(merged[0]) <= max_size):
                return merged

    # Fallback: character-level split
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def _apply_overlap(
    chunks: list[str],
    overlap: int,
    max_size: int,
) -> list[str]:
    """Add overlap from the end of each chunk to the start of the next."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
        combined = f"{overlap_text}{chunks[i]}"
        if len(combined) > max_size:
            combined = combined[:max_size]
        result.append(combined)
    return result


def _merge_small_blocks(
    blocks: list[Block],
    chunk_size: int,
    min_chunk_size: int,
) -> list[Block]:
    """Merge consecutive small text blocks into larger ones.

    Rules:
    - Table blocks (parent/child) are NEVER merged
    - Text blocks merge with neighbors if buffer < min_chunk_size
    - Respects section boundaries when buffer is already substantial
    """
    if not blocks:
        return blocks

    merged: list[Block] = []
    buffer: Block | None = None

    for block in blocks:
        # Never merge table blocks
        if block.block_type in _TABLE_TYPES:
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(block)
            continue

        if buffer is None:
            buffer = block.copy()
            continue

        # Section changed and buffer is substantial → flush
        new_section = block.metadata.get("section", "")
        old_section = buffer.metadata.get("section", "")
        section_changed = new_section and old_section and new_section != old_section
        buffer_is_large = buffer.char_count >= min_chunk_size

        combined_len = buffer.char_count + block.char_count + 2
        if combined_len <= chunk_size and not (section_changed and buffer_is_large):
            buffer = Block(
                content=f"{buffer.content}\n\n{block.content}",
                block_type=buffer.block_type,
                metadata={**buffer.metadata, **block.metadata},
            )
        else:
            merged.append(buffer)
            buffer = block.copy()

    if buffer:
        merged.append(buffer)

    return merged
