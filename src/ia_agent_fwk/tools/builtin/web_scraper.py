"""Web scraper stub built-in tool.

Returns placeholder content. Full implementation deferred to a later epic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.tools.base import Tool, ToolContext


class WebScraperInput(BaseModel):
    """Input schema for the web scraper tool."""

    model_config = ConfigDict(frozen=True)

    url: str


class WebScraperOutput(BaseModel):
    """Output schema for the web scraper tool."""

    model_config = ConfigDict(frozen=True)

    title: str
    text_content: str
    links: list[str] = Field(default_factory=list)


class WebScraperTool(Tool):
    """Web scraper stub tool.

    Returns placeholder content for a given URL. Full implementation
    (with BeautifulSoup or similar) deferred to a later epic.
    """

    @property
    def name(self) -> str:
        return "web_scraper"

    @property
    def description(self) -> str:
        return "Scrape web page content (stub: returns placeholder data)."

    @property
    def input_schema(self) -> type[BaseModel]:
        return WebScraperInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return WebScraperOutput

    @property
    def tags(self) -> list[str]:
        return ["web", "scraping", "builtin", "stub"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Return placeholder scraping results."""
        assert isinstance(validated_input, WebScraperInput)  # noqa: S101
        return WebScraperOutput(
            title=f"Placeholder title for {validated_input.url}",
            text_content=(
                f"This is placeholder content for {validated_input.url}. "
                "The web scraper tool is a stub. Full implementation is deferred to a later epic."
            ),
            links=[validated_input.url],
        )
