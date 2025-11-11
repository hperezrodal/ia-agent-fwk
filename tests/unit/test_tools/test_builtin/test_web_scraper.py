"""Tests for the web scraper stub built-in tool."""

from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.builtin.web_scraper import WebScraperInput, WebScraperOutput, WebScraperTool


class TestWebScraperStub:
    async def test_returns_placeholder_content(self):
        tool = WebScraperTool()
        ctx = ToolContext(execution_id="test-scraper")
        result = await tool.execute(WebScraperInput(url="https://example.com"), ctx)

        assert isinstance(result, WebScraperOutput)
        assert "example.com" in result.title
        assert "placeholder" in result.text_content.lower()
        assert len(result.links) > 0

    def test_name(self):
        assert WebScraperTool().name == "web_scraper"

    def test_tags(self):
        assert "stub" in WebScraperTool().tags
        assert "builtin" in WebScraperTool().tags
