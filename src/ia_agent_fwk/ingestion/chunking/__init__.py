"""Generic chunking pipeline by composition.

4 composable stages, each a pure function: list[Block] → list[Block]

    SPLIT     → divide document into semantic blocks
    TRANSFORM → type-aware transformations (tables → hierarchical, headings → prefix)
    SIZE      → adjust sizes (split large, merge small)
    ENRICH    → attach metadata (section, page, chunk_type)

Usage:
    from ia_agent_fwk.ingestion.chunking import ChunkingPipeline

    pipeline = ChunkingPipeline(chunk_size=1000, chunk_overlap=200)
    chunks = pipeline.process(markdown_text)
"""

from ia_agent_fwk.ingestion.chunking.pipeline import ChunkingPipeline

__all__ = ["ChunkingPipeline"]
