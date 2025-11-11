"""Tests for the database query stub built-in tool."""

from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.builtin.database_query import (
    DatabaseQueryInput,
    DatabaseQueryOutput,
    DatabaseQueryTool,
)


class TestDatabaseQueryStub:
    async def test_returns_placeholder_results(self):
        tool = DatabaseQueryTool()
        ctx = ToolContext(execution_id="test-db")
        result = await tool.execute(
            DatabaseQueryInput(query="SELECT * FROM users", database_name="test_db"),
            ctx,
        )

        assert isinstance(result, DatabaseQueryOutput)
        assert len(result.columns) > 0
        assert len(result.rows) > 0
        assert result.row_count == 2

    def test_name(self):
        assert DatabaseQueryTool().name == "database_query"

    def test_tags(self):
        assert "stub" in DatabaseQueryTool().tags
        assert "builtin" in DatabaseQueryTool().tags
