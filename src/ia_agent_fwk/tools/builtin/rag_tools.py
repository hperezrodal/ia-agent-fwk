"""RAG search tool for semantic document search."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class RAGSearchInput(BaseModel):
    """Input schema for the RAG search tool."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="The search query to find relevant document chunks.")
    top_k: int = Field(default=5, description="Number of results to return.", ge=1, le=20)


class RAGSearchOutput(BaseModel):
    """Output schema for the RAG search tool."""

    model_config = ConfigDict(frozen=True)

    results: str


class RAGSearchTool(Tool):
    """Search ingested documents using semantic similarity via the RAG pipeline.

    This tool queries the vector database for document chunks that are
    semantically similar to the query. Documents must be ingested first.
    """

    _pipeline: ClassVar[Any] = None

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Search through ingested documents using semantic similarity. "
            "Returns the most relevant document chunks matching the query. "
            "Use this to find specific information across all documents."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return RAGSearchInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return RAGSearchOutput

    @property
    def tags(self) -> list[str]:
        return ["rag", "search", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Execute a RAG search query."""
        assert isinstance(validated_input, RAGSearchInput)  # noqa: S101

        collector = get_metrics_collector()
        start = time.monotonic()

        try:
            pipeline = self._get_pipeline()
            results = await pipeline.query(validated_input.query, top_k=validated_input.top_k)
            duration = time.monotonic() - start

            collector.increment("rag_tool_searches_total")
            collector.observe("rag_tool_search_duration_seconds", duration)
            collector.increment("rag_tool_results_returned_total", value=len(results))

            if not results:
                return RAGSearchOutput(results=json.dumps({"results": [], "message": "No matching documents found."}))

            if results:
                collector.observe("rag_top_score", results[0].score)

            output = [
                {
                    "content": r.chunk.content,
                    "source": r.chunk.source,
                    "score": round(r.score, 4),
                    "chunk_index": r.chunk.chunk_index,
                }
                for r in results
            ]

            return RAGSearchOutput(results=json.dumps({"results": output}, ensure_ascii=False))

        except Exception as exc:
            collector.increment("rag_tool_errors_total")
            logger.exception("RAG search failed")
            return RAGSearchOutput(results=json.dumps({"error": f"RAG search failed: {exc}"}))

    @classmethod
    def _get_pipeline(cls) -> Any:
        """Lazy-create the RAG pipeline singleton."""
        if cls._pipeline is None:
            from ia_agent_fwk.config.loader import load_config  # noqa: PLC0415
            from ia_agent_fwk.memory.embeddings.factory import EmbeddingFactory  # noqa: PLC0415
            from ia_agent_fwk.memory.factory import MemoryFactory  # noqa: PLC0415
            from ia_agent_fwk.rag.factory import RAGPipelineFactory  # noqa: PLC0415

            settings = load_config()
            embedding_provider = EmbeddingFactory.create(settings.memory.embedding)
            memory_backend = MemoryFactory.create(settings.memory)
            cls._pipeline = RAGPipelineFactory.create(
                settings=settings.rag,
                memory_backend=memory_backend,
                embedding_provider=embedding_provider,
            )
        return cls._pipeline
