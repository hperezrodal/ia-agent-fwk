"""Database query stub built-in tool.

Returns placeholder query results. Full implementation deferred to a later epic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.tools.base import Tool, ToolContext


class DatabaseQueryInput(BaseModel):
    """Input schema for the database query tool."""

    model_config = ConfigDict(frozen=True)

    query: str
    database_name: str = "default"


class DatabaseQueryOutput(BaseModel):
    """Output schema for the database query tool."""

    model_config = ConfigDict(frozen=True)

    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


class DatabaseQueryTool(Tool):
    """Database query stub tool.

    Returns placeholder query results. Full implementation
    (with SQLAlchemy) deferred to a later epic.
    """

    @property
    def name(self) -> str:
        return "database_query"

    @property
    def description(self) -> str:
        return "Execute a database query (stub: returns placeholder results)."

    @property
    def input_schema(self) -> type[BaseModel]:
        return DatabaseQueryInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return DatabaseQueryOutput

    @property
    def tags(self) -> list[str]:
        return ["database", "sql", "builtin", "stub"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Return placeholder query results."""
        assert isinstance(validated_input, DatabaseQueryInput)  # noqa: S101
        return DatabaseQueryOutput(
            columns=["id", "name", "value"],
            rows=[
                {"id": 1, "name": "placeholder_row_1", "value": "data_1"},
                {"id": 2, "name": "placeholder_row_2", "value": "data_2"},
            ],
            row_count=2,
        )
