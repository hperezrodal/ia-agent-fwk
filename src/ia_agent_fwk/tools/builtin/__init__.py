"""Built-in tool auto-registration.

``register_builtin_tools()`` creates and registers all built-in tool
instances in a given ``ToolRegistry``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ia_agent_fwk.tools.builtin.calculator import CalculatorTool
from ia_agent_fwk.tools.builtin.calendar_tools import EmailParserTool, EventValidatorTool
from ia_agent_fwk.tools.builtin.current_time import CurrentTimeTool
from ia_agent_fwk.tools.builtin.database_query import DatabaseQueryTool
from ia_agent_fwk.tools.builtin.document_loader_tools import (
    ListDocumentsTool,
    LoadDocumentTool,
)
from ia_agent_fwk.tools.builtin.document_tools import (
    EntityExtractorTool,
    SectionIdentifierTool,
    SummarizerTool,
    TextExtractorTool,
)
from ia_agent_fwk.tools.builtin.echo import EchoTool
from ia_agent_fwk.tools.builtin.file_reader import FileReaderTool
from ia_agent_fwk.tools.builtin.finance_tools import (
    AnomalyDetectorTool,
    FinancialDataLookupTool,
    RatioCalculatorTool,
    ReportGeneratorTool,
)
from ia_agent_fwk.tools.builtin.http_request import HttpRequestTool
from ia_agent_fwk.tools.builtin.rag_tools import RAGSearchTool
from ia_agent_fwk.tools.builtin.support_tools import (
    EscalationTool,
    FAQSearchTool,
    ResponseDraftTool,
    TicketLookupTool,
)
from ia_agent_fwk.tools.builtin.web_scraper import WebScraperTool

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import ToolSandboxingSettings
    from ia_agent_fwk.tools.registry import ToolRegistry


def register_builtin_tools(
    registry: ToolRegistry,
    sandboxing_config: ToolSandboxingSettings | None = None,
) -> None:
    """Register all built-in tools in the given registry.

    Parameters
    ----------
    registry:
        The tool registry to register tools in.
    sandboxing_config:
        Optional sandboxing configuration for domain/path allowlists.

    """
    allowed_domains: list[str] = []
    if sandboxing_config is not None:
        allowed_domains = sandboxing_config.allowed_domains

    # Register core built-in tools
    registry.register(CalculatorTool())
    registry.register(FileReaderTool())
    registry.register(HttpRequestTool(allowed_domains=allowed_domains))
    registry.register(WebScraperTool())
    registry.register(DatabaseQueryTool())
    registry.register(CurrentTimeTool())
    registry.register(EchoTool())

    # Register customer support tools
    registry.register(TicketLookupTool())
    registry.register(FAQSearchTool())
    registry.register(EscalationTool())
    registry.register(ResponseDraftTool())

    # Register document processor tools
    registry.register(TextExtractorTool())
    registry.register(SectionIdentifierTool())
    registry.register(EntityExtractorTool())
    registry.register(SummarizerTool())

    # Register document loader tools (file-based)
    registry.register(ListDocumentsTool())
    registry.register(LoadDocumentTool())

    # Register finance tools
    registry.register(FinancialDataLookupTool())
    registry.register(RatioCalculatorTool())
    registry.register(AnomalyDetectorTool())
    registry.register(ReportGeneratorTool())

    # Register RAG tools
    registry.register(RAGSearchTool())

    # Register calendar tools (no-dependency tools only)
    registry.register(EmailParserTool())
    registry.register(EventValidatorTool())
