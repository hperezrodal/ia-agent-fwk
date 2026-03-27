"""Data models for the chunking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BlockType(str, Enum):
    """Type of content block."""

    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"
    TABLE_PARENT = "table:parent"
    TABLE_CHILD = "table:child"


@dataclass
class Block:
    """A content block flowing through the chunking pipeline.

    Each stage transforms the list of blocks and can modify any field.
    The ``metadata`` dict accumulates information across stages.
    """

    content: str
    block_type: BlockType = BlockType.TEXT
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)

    def copy(self, **overrides) -> Block:  # noqa: ANN003
        """Create a shallow copy with optional field overrides."""
        return Block(
            content=overrides.get("content", self.content),
            block_type=overrides.get("block_type", self.block_type),
            metadata={**self.metadata, **overrides.get("metadata", {})},
        )
