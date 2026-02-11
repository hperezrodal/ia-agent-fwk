"""Document Processor Agent example.

Demonstrates how to build a document-analysis agent using the ``ia_agent_fwk``
framework.  The agent is configured with four document-processing tools and
produces structured JSON output describing the document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.tools.builtin.document_tools import (
    EntityExtractorTool,
    SectionIdentifierTool,
    SummarizerTool,
    TextExtractorTool,
)
from ia_agent_fwk.tools.executor import DefaultToolExecutor
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
from ia_agent_fwk.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from ia_agent_fwk.llm.base import LLMProvider

_SYSTEM_PROMPT = """\
You are a document analysis agent.  When given a document, you MUST use the
available tools in the following order:

1. **text_extractor** -- extract the raw text from the document.
2. **section_identifier** -- identify section headings and structure.
3. **entity_extractor** -- extract named entities (dates, amounts, names, emails).
4. **summarizer** -- produce a concise summary.

After calling all four tools, compose a structured JSON response with the keys:
  - "summary": the summary text
  - "sections": list of section titles
  - "entities": object with "dates", "amounts", "emails", "names" arrays
  - "statistics": object with "char_count", "line_count", "section_count", "entity_count"

Always respond with valid JSON and nothing else.
"""

_DEFAULT_AGENT_NAME = "document-processor"


class DocumentProcessorAgent(Agent):
    """Reference agent for document analysis tasks.

    Pre-configured with ``TextExtractorTool``, ``SectionIdentifierTool``,
    ``EntityExtractorTool``, and ``SummarizerTool``.
    """

    @property
    def agent_type(self) -> str:
        return "document_processor"


def create_document_processor_agent(
    provider: LLMProvider,
    *,
    agent_name: str = _DEFAULT_AGENT_NAME,
    max_iterations: int = 10,
    execution_timeout: int = 300,
) -> DocumentProcessorAgent:
    """Create a fully-configured document processor agent.

    Parameters
    ----------
    provider:
        LLM provider instance to use for reasoning.
    agent_name:
        Logical name for the agent.
    max_iterations:
        Maximum reasoning-loop iterations.
    execution_timeout:
        Maximum execution time in seconds.

    Returns
    -------
    DocumentProcessorAgent
        Ready-to-run agent instance.

    """
    config = AgentConfig(
        name=agent_name,
        agent_type="document_processor",
        system_prompt=_SYSTEM_PROMPT,
        max_iterations=max_iterations,
        execution_timeout=execution_timeout,
        max_tokens_per_response=4096,
        tools=[
            "text_extractor",
            "section_identifier",
            "entity_extractor",
            "summarizer",
        ],
    )

    # Build tool registry with document tools
    registry = ToolRegistry()
    registry.register(TextExtractorTool())
    registry.register(SectionIdentifierTool())
    registry.register(EntityExtractorTool())
    registry.register(SummarizerTool())

    permission_manager = ToolPermissionManager(default_mode=PermissionMode.allow_all)
    tool_executor = DefaultToolExecutor(
        registry=registry,
        permission_manager=permission_manager,
        agent_id=agent_name,
    )

    return DocumentProcessorAgent(
        config=config,
        provider=provider,
        tool_executor=tool_executor,
    )
