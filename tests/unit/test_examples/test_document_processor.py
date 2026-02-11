"""Tests for the Document Processor agent example and its tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.models import (
    ChatResponse,
    FinishReason,
    Message,
    TokenUsage,
    ToolCall,
)
from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.builtin.document_tools import (
    EntityExtractorInput,
    EntityExtractorOutput,
    EntityExtractorTool,
    SectionIdentifierInput,
    SectionIdentifierOutput,
    SectionIdentifierTool,
    SummarizerInput,
    SummarizerOutput,
    SummarizerTool,
    TextExtractorInput,
    TextExtractorOutput,
    TextExtractorTool,
)
from ia_agent_fwk.tools.exceptions import ToolExecutionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "document_processor" / "data"


def _ctx() -> ToolContext:
    return ToolContext(execution_id="test-exec-001", agent_id="test-agent")


def _read_sample(filename: str) -> str:
    return (_DATA_DIR / filename).read_text()


# ---------------------------------------------------------------------------
# TextExtractorTool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTextExtractorTool:
    def test_properties(self):
        tool = TextExtractorTool()
        assert tool.name == "text_extractor"
        assert "extract" in tool.description.lower()
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)
        assert "document" in tool.tags

    async def test_extract_plain_text(self):
        tool = TextExtractorTool()
        inp = TextExtractorInput(content="Hello world.\nSecond line.")
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, TextExtractorOutput)
        assert result.char_count > 0
        assert result.line_count == 2

    async def test_extract_strips_trailing_whitespace(self):
        tool = TextExtractorTool()
        inp = TextExtractorInput(content="Line one   \nLine two   \n")
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, TextExtractorOutput)
        assert "   " not in result.text.split("\n")[0]

    async def test_empty_content_raises(self):
        tool = TextExtractorTool()
        inp = TextExtractorInput(content="   ")
        with pytest.raises(ToolExecutionError, match="empty"):
            await tool.execute(inp, _ctx())

    async def test_oversized_content_raises(self):
        tool = TextExtractorTool()
        inp = TextExtractorInput(content="x" * 100_001)
        with pytest.raises(ToolExecutionError, match="maximum length"):
            await tool.execute(inp, _ctx())

    async def test_extract_sample_contract(self):
        tool = TextExtractorTool()
        content = _read_sample("sample_contract.txt")
        inp = TextExtractorInput(content=content)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, TextExtractorOutput)
        assert result.char_count > 100
        assert result.line_count > 5


# ---------------------------------------------------------------------------
# SectionIdentifierTool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSectionIdentifierTool:
    def test_properties(self):
        tool = SectionIdentifierTool()
        assert tool.name == "section_identifier"
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)
        assert "document" in tool.tags

    async def test_identify_allcaps_headings(self):
        tool = SectionIdentifierTool()
        text = "INTRODUCTION\n\nSome text here.\n\nCONCLUSION\n\nFinal text."
        inp = SectionIdentifierInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, SectionIdentifierOutput)
        assert result.total_sections == 2
        titles = [s.title for s in result.sections]
        assert "INTRODUCTION" in titles
        assert "CONCLUSION" in titles

    async def test_identify_numbered_headings(self):
        tool = SectionIdentifierTool()
        text = "1. Introduction\n\nText.\n\n2. Methods\n\nMore text."
        inp = SectionIdentifierInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, SectionIdentifierOutput)
        assert result.total_sections == 2

    async def test_identify_markdown_headings(self):
        tool = SectionIdentifierTool()
        text = "# Title\n\nParagraph.\n\n## Subtitle\n\nMore text."
        inp = SectionIdentifierInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, SectionIdentifierOutput)
        assert result.total_sections == 2
        assert result.sections[0].title == "Title"
        assert result.sections[1].title == "Subtitle"

    async def test_empty_text_raises(self):
        tool = SectionIdentifierTool()
        inp = SectionIdentifierInput(text="   ")
        with pytest.raises(ToolExecutionError, match="empty"):
            await tool.execute(inp, _ctx())

    async def test_no_headings_returns_empty(self):
        tool = SectionIdentifierTool()
        inp = SectionIdentifierInput(text="Just plain text without any headings at all.")
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, SectionIdentifierOutput)
        assert result.total_sections == 0

    async def test_line_numbers_are_correct(self):
        tool = SectionIdentifierTool()
        text = "Some intro text\n\nSECTION ONE\n\nBody text\n\nSECTION TWO\n\nMore body"
        inp = SectionIdentifierInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert result.sections[0].start_line == 3
        assert result.sections[1].start_line == 7


# ---------------------------------------------------------------------------
# EntityExtractorTool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityExtractorTool:
    def test_properties(self):
        tool = EntityExtractorTool()
        assert tool.name == "entity_extractor"
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)
        assert "document" in tool.tags

    async def test_extract_dates(self):
        tool = EntityExtractorTool()
        text = "The contract was signed on January 15, 2025 and expires on 12/31/2025."
        inp = EntityExtractorInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, EntityExtractorOutput)
        assert len(result.entities.dates) >= 2

    async def test_extract_amounts(self):
        tool = EntityExtractorTool()
        text = "The total cost is $15,000.00 with a deposit of $5,000.00."
        inp = EntityExtractorInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, EntityExtractorOutput)
        assert "$15,000.00" in result.entities.amounts
        assert "$5,000.00" in result.entities.amounts

    async def test_extract_emails(self):
        tool = EntityExtractorTool()
        text = "Contact us at info@example.com or support@test.org."
        inp = EntityExtractorInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, EntityExtractorOutput)
        assert "info@example.com" in result.entities.emails
        assert "support@test.org" in result.entities.emails

    async def test_extract_names(self):
        tool = EntityExtractorTool()
        text = "John Smith and Jane Doe signed the agreement."
        inp = EntityExtractorInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, EntityExtractorOutput)
        assert "John Smith" in result.entities.names

    async def test_deduplication(self):
        tool = EntityExtractorTool()
        text = "Contact John Smith. Also, reach out to John Smith for details."
        inp = EntityExtractorInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, EntityExtractorOutput)
        # Should not have duplicates
        assert len(result.entities.names) == len(set(result.entities.names))

    async def test_empty_text_raises(self):
        tool = EntityExtractorTool()
        inp = EntityExtractorInput(text="")
        with pytest.raises(ToolExecutionError, match="empty"):
            await tool.execute(inp, _ctx())

    async def test_total_entities_count(self):
        tool = EntityExtractorTool()
        text = "John Smith signed on January 15, 2025 for $10,000.00. Email: js@co.com"
        inp = EntityExtractorInput(text=text)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, EntityExtractorOutput)
        expected_total = (
            len(result.entities.dates)
            + len(result.entities.amounts)
            + len(result.entities.emails)
            + len(result.entities.names)
        )
        assert result.total_entities == expected_total

    async def test_extract_from_sample_invoice(self):
        tool = EntityExtractorTool()
        content = _read_sample("sample_invoice.txt")
        inp = EntityExtractorInput(text=content)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, EntityExtractorOutput)
        assert len(result.entities.amounts) > 0
        assert len(result.entities.emails) > 0


# ---------------------------------------------------------------------------
# SummarizerTool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSummarizerTool:
    def test_properties(self):
        tool = SummarizerTool()
        assert tool.name == "summarizer"
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)
        assert "document" in tool.tags

    async def test_summarize_short_text(self):
        tool = SummarizerTool()
        text = "First sentence. Second sentence. Third sentence."
        inp = SummarizerInput(text=text, max_sentences=2)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, SummarizerOutput)
        assert "First sentence." in result.summary
        assert "Second sentence." in result.summary
        assert result.original_length == len(text)
        assert result.summary_length <= result.original_length

    async def test_summarize_respects_max_sentences(self):
        tool = SummarizerTool()
        text = "One. Two. Three. Four. Five. Six. Seven."
        inp = SummarizerInput(text=text, max_sentences=3)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, SummarizerOutput)
        # Should contain at most 3 sentence-like segments
        assert result.summary_length < result.original_length

    async def test_summarize_single_sentence(self):
        tool = SummarizerTool()
        text = "This is a single sentence without any period-delimited breaks"
        inp = SummarizerInput(text=text, max_sentences=5)
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, SummarizerOutput)
        assert result.summary == text

    async def test_empty_text_raises(self):
        tool = SummarizerTool()
        inp = SummarizerInput(text="   ")
        with pytest.raises(ToolExecutionError, match="empty"):
            await tool.execute(inp, _ctx())

    async def test_default_max_sentences(self):
        tool = SummarizerTool()
        inp = SummarizerInput(text="A. B. C. D. E. F. G. H.")
        result = await tool.execute(inp, _ctx())
        assert isinstance(result, SummarizerOutput)
        # Default is 5 sentences
        assert result.summary_length <= result.original_length


# ---------------------------------------------------------------------------
# Agent creation and configuration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDocumentProcessorAgentCreation:
    def test_agent_type(self):
        from examples.document_processor.agent import DocumentProcessorAgent

        # We need a mock provider to instantiate the agent
        from tests.unit.test_agents.conftest import MockLLMProvider, make_chat_response

        provider = MockLLMProvider(responses=[make_chat_response()])
        config = AgentConfig(
            name="test-doc-agent",
            agent_type="document_processor",
            system_prompt="Test prompt.",
            max_iterations=5,
            execution_timeout=60,
            max_tokens_per_response=2048,
        )
        agent = DocumentProcessorAgent(config=config, provider=provider)
        assert agent.agent_type == "document_processor"
        assert agent.state == AgentState.IDLE
        assert agent.config.name == "test-doc-agent"

    def test_factory_creates_configured_agent(self):
        from examples.document_processor.agent import create_document_processor_agent

        from tests.unit.test_agents.conftest import MockLLMProvider, make_chat_response

        provider = MockLLMProvider(responses=[make_chat_response()])
        agent = create_document_processor_agent(provider, agent_name="my-doc-agent")
        assert agent.agent_type == "document_processor"
        assert agent.config.name == "my-doc-agent"
        assert "text_extractor" in agent.config.tools
        assert "section_identifier" in agent.config.tools
        assert "entity_extractor" in agent.config.tools
        assert "summarizer" in agent.config.tools

    def test_factory_default_name(self):
        from examples.document_processor.agent import create_document_processor_agent

        from tests.unit.test_agents.conftest import MockLLMProvider, make_chat_response

        provider = MockLLMProvider(responses=[make_chat_response()])
        agent = create_document_processor_agent(provider)
        assert agent.config.name == "document-processor"

    def test_factory_custom_params(self):
        from examples.document_processor.agent import create_document_processor_agent

        from tests.unit.test_agents.conftest import MockLLMProvider, make_chat_response

        provider = MockLLMProvider(responses=[make_chat_response()])
        agent = create_document_processor_agent(
            provider,
            max_iterations=20,
            execution_timeout=600,
        )
        assert agent.config.max_iterations == 20
        assert agent.config.execution_timeout == 600


# ---------------------------------------------------------------------------
# End-to-end test with mocked LLM
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDocumentProcessorE2E:
    async def test_agent_processes_document(self):
        """End-to-end: agent receives a document and produces structured output."""
        from examples.document_processor.agent import create_document_processor_agent

        from tests.unit.test_agents.conftest import MockLLMProvider

        expected_output = json.dumps(
            {
                "summary": "This is a service agreement between Acme Corp and Global Industries.",
                "sections": ["SCOPE OF SERVICES", "TERM AND TERMINATION", "COMPENSATION"],
                "entities": {
                    "dates": ["January 15, 2025"],
                    "amounts": ["$15,000.00"],
                    "emails": ["john.smith@acmecorp.com"],
                    "names": ["John Smith"],
                },
                "statistics": {
                    "char_count": 500,
                    "line_count": 20,
                    "section_count": 3,
                    "entity_count": 4,
                },
            }
        )

        # The mock provider returns a single stop response with the expected JSON
        response = ChatResponse(
            message=Message(role="assistant", content=expected_output),
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            model="mock-model",
            finish_reason=FinishReason.stop,
        )
        provider = MockLLMProvider(responses=[response])

        agent = create_document_processor_agent(provider)
        contract_text = _read_sample("sample_contract.txt")
        result = await agent.run(f"Analyze this document:\n\n{contract_text}")

        assert result.state == AgentState.COMPLETED
        assert result.error is None
        # The agent should produce the mocked LLM output
        parsed = json.loads(result.output)
        assert "summary" in parsed
        assert "sections" in parsed
        assert "entities" in parsed

    async def test_agent_handles_tool_calls(self):
        """Agent processes tool calls returned by the LLM."""
        from examples.document_processor.agent import create_document_processor_agent

        from tests.unit.test_agents.conftest import MockLLMProvider

        # First response: LLM requests text_extractor tool
        tool_call = ToolCall(
            id="tc-1",
            name="text_extractor",
            arguments=json.dumps({"content": "Hello world.", "format": "plain"}),
        )
        tool_call_response = ChatResponse(
            message=Message(
                role="assistant",
                content="Let me extract the text.",
                tool_calls=[tool_call],
            ),
            usage=TokenUsage(prompt_tokens=50, completion_tokens=30),
            model="mock-model",
            finish_reason=FinishReason.tool_calls,
        )

        # Second response: LLM produces final output
        final_response = ChatResponse(
            message=Message(
                role="assistant",
                content='{"summary": "A simple document.", "sections": [], "entities": {}, "statistics": {}}',
            ),
            usage=TokenUsage(prompt_tokens=80, completion_tokens=40),
            model="mock-model",
            finish_reason=FinishReason.stop,
        )

        provider = MockLLMProvider(responses=[tool_call_response, final_response])
        agent = create_document_processor_agent(provider)

        result = await agent.run("Analyze: Hello world.")
        assert result.state == AgentState.COMPLETED
        assert result.iterations >= 1


# ---------------------------------------------------------------------------
# Sample data existence tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSampleData:
    def test_sample_contract_exists(self):
        path = _DATA_DIR / "sample_contract.txt"
        assert path.exists()
        content = path.read_text()
        assert len(content) > 100
        assert "Agreement" in content

    def test_sample_invoice_exists(self):
        path = _DATA_DIR / "sample_invoice.txt"
        assert path.exists()
        content = path.read_text()
        assert len(content) > 100
        assert "Invoice" in content or "INVOICE" in content

    def test_sample_report_exists(self):
        path = _DATA_DIR / "sample_report.txt"
        assert path.exists()
        content = path.read_text()
        assert len(content) > 100
        assert "REPORT" in content or "Report" in content
