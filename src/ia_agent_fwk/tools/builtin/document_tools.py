"""Document processing built-in tools.

Provides tools for text extraction, section identification, entity extraction,
and summarization.  All implementations are mock/rule-based and do not require
external services.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.tools.base import Tool
from ia_agent_fwk.tools.exceptions import ToolExecutionError

if TYPE_CHECKING:
    from ia_agent_fwk.tools.base import ToolContext

# ---------------------------------------------------------------------------
# TextExtractorTool
# ---------------------------------------------------------------------------

_TEXT_EXTRACTOR_MAX_LENGTH = 100_000


class TextExtractorInput(BaseModel):
    """Input schema for the text extractor tool."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="Raw document content to extract text from.")
    format: str = Field(
        default="plain",
        description="Source format hint (plain, markdown, html). Default: plain.",
    )


class TextExtractorOutput(BaseModel):
    """Output schema for the text extractor tool."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Extracted plain text.")
    char_count: int = Field(description="Number of characters in the extracted text.")
    line_count: int = Field(description="Number of lines in the extracted text.")


class TextExtractorTool(Tool):
    """Extract plain text from a document.

    This is a mock implementation that normalises whitespace and returns
    basic statistics.  A production version would delegate to libraries
    such as ``pypdf``, ``python-docx``, or ``beautifulsoup4``.
    """

    @property
    def name(self) -> str:
        return "text_extractor"

    @property
    def description(self) -> str:
        return "Extract plain text from a document and return basic statistics."

    @property
    def input_schema(self) -> type[BaseModel]:
        return TextExtractorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return TextExtractorOutput

    @property
    def tags(self) -> list[str]:
        return ["document", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Extract text from the provided content."""
        assert isinstance(validated_input, TextExtractorInput)  # noqa: S101

        raw = validated_input.content
        if not raw.strip():
            msg = "Document content is empty."
            raise ToolExecutionError(msg, tool_name="text_extractor")

        if len(raw) > _TEXT_EXTRACTOR_MAX_LENGTH:
            msg = f"Document exceeds maximum length of {_TEXT_EXTRACTOR_MAX_LENGTH} characters."
            raise ToolExecutionError(msg, tool_name="text_extractor")

        # Normalise whitespace: collapse multiple blank lines, strip trailing spaces
        lines = [line.rstrip() for line in raw.splitlines()]
        text = "\n".join(lines).strip()

        return TextExtractorOutput(
            text=text,
            char_count=len(text),
            line_count=text.count("\n") + 1,
        )


# ---------------------------------------------------------------------------
# SectionIdentifierTool
# ---------------------------------------------------------------------------

# Heuristic patterns for section headings
_HEADING_PATTERNS: list[re.Pattern[str]] = [
    # Markdown-style headings
    re.compile(r"^#{1,6}\s+(.+)$"),
    # ALL-CAPS lines (at least 3 chars, allow trailing colon)
    re.compile(r"^([A-Z][A-Z .&/]{2,}):?\s*$"),
    # Numbered sections like "1. Introduction" or "1.2 Overview"
    re.compile(r"^(\d+(?:\.\d+)*\.?\s+.+)$"),
]


class SectionIdentifierInput(BaseModel):
    """Input schema for the section identifier tool."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Plain text to scan for section headings.")


class SectionInfo(BaseModel):
    """A single identified section."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Section heading text.")
    start_line: int = Field(description="1-based line number where the section starts.")


class SectionIdentifierOutput(BaseModel):
    """Output schema for the section identifier tool."""

    model_config = ConfigDict(frozen=True)

    sections: list[SectionInfo] = Field(description="List of identified sections.")
    total_sections: int = Field(description="Total number of sections found.")


class SectionIdentifierTool(Tool):
    """Identify document sections/structure using heuristic heading detection.

    Scans plain text for common heading patterns (Markdown, ALL-CAPS,
    numbered headings) and returns a list of identified sections with
    their line numbers.
    """

    @property
    def name(self) -> str:
        return "section_identifier"

    @property
    def description(self) -> str:
        return "Identify document sections and their structure from plain text."

    @property
    def input_schema(self) -> type[BaseModel]:
        return SectionIdentifierInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return SectionIdentifierOutput

    @property
    def tags(self) -> list[str]:
        return ["document", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Identify sections in the provided text."""
        assert isinstance(validated_input, SectionIdentifierInput)  # noqa: S101

        if not validated_input.text.strip():
            msg = "Text is empty."
            raise ToolExecutionError(msg, tool_name="section_identifier")

        sections: list[SectionInfo] = []
        for line_no, line in enumerate(validated_input.text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            for pattern in _HEADING_PATTERNS:
                match = pattern.match(stripped)
                if match:
                    title = match.group(1).strip() if match.lastindex else stripped.strip("# ").strip()
                    sections.append(SectionInfo(title=title, start_line=line_no))
                    break  # one match per line is enough

        return SectionIdentifierOutput(sections=sections, total_sections=len(sections))


# ---------------------------------------------------------------------------
# EntityExtractorTool
# ---------------------------------------------------------------------------

# Named-entity patterns (simple regex-based)
_DATE_PATTERN = re.compile(
    r"\b("
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"  # MM/DD/YYYY or DD-MM-YYYY
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}"  # January 1, 2024
    r")\b",
    re.IGNORECASE,
)
_AMOUNT_PATTERN = re.compile(
    r"\$[\d,]+(?:\.\d{2})?",
)
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
)
# Simple name pattern: two or more capitalised words in sequence
_NAME_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
)


class EntityExtractorInput(BaseModel):
    """Input schema for the entity extractor tool."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Text to extract entities from.")


class ExtractedEntities(BaseModel):
    """Container for extracted entities by category."""

    model_config = ConfigDict(frozen=True)

    dates: list[str] = Field(default_factory=list, description="Extracted date strings.")
    amounts: list[str] = Field(default_factory=list, description="Extracted monetary amounts.")
    emails: list[str] = Field(default_factory=list, description="Extracted email addresses.")
    names: list[str] = Field(default_factory=list, description="Extracted person/org names.")


class EntityExtractorOutput(BaseModel):
    """Output schema for the entity extractor tool."""

    model_config = ConfigDict(frozen=True)

    entities: ExtractedEntities = Field(description="Extracted entities grouped by category.")
    total_entities: int = Field(description="Total number of entities extracted.")


class EntityExtractorTool(Tool):
    """Extract named entities from text using regex heuristics.

    Detects dates, monetary amounts, email addresses, and proper names.
    This is a rule-based mock; a production version would use an NER
    model or LLM-based extraction.
    """

    @property
    def name(self) -> str:
        return "entity_extractor"

    @property
    def description(self) -> str:
        return "Extract named entities (dates, amounts, emails, names) from text."

    @property
    def input_schema(self) -> type[BaseModel]:
        return EntityExtractorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return EntityExtractorOutput

    @property
    def tags(self) -> list[str]:
        return ["document", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Extract entities from the provided text."""
        assert isinstance(validated_input, EntityExtractorInput)  # noqa: S101

        text = validated_input.text
        if not text.strip():
            msg = "Text is empty."
            raise ToolExecutionError(msg, tool_name="entity_extractor")

        dates = _dedupe(_DATE_PATTERN.findall(text))
        amounts = _dedupe(_AMOUNT_PATTERN.findall(text))
        emails = _dedupe(_EMAIL_PATTERN.findall(text))
        names = _dedupe(_NAME_PATTERN.findall(text))

        entities = ExtractedEntities(dates=dates, amounts=amounts, emails=emails, names=names)
        total = len(dates) + len(amounts) + len(emails) + len(names)

        return EntityExtractorOutput(entities=entities, total_entities=total)


def _dedupe(items: list[str]) -> list[str]:
    """Return unique items preserving insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# SummarizerTool
# ---------------------------------------------------------------------------

_SUMMARIZER_MAX_SENTENCES = 5
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SummarizerInput(BaseModel):
    """Input schema for the summarizer tool."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Text to summarize.")
    max_sentences: int = Field(
        default=_SUMMARIZER_MAX_SENTENCES,
        ge=1,
        description="Maximum number of sentences in the summary.",
    )


class SummarizerOutput(BaseModel):
    """Output schema for the summarizer tool."""

    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="Summarized text.")
    original_length: int = Field(description="Character count of the original text.")
    summary_length: int = Field(description="Character count of the summary.")


class SummarizerTool(Tool):
    """Produce a simple extractive summary of a document.

    Uses a naive first-N-sentences approach.  A production implementation
    would use an LLM or a dedicated summarization model.
    """

    @property
    def name(self) -> str:
        return "summarizer"

    @property
    def description(self) -> str:
        return "Produce an extractive summary of a document by selecting the first N sentences."

    @property
    def input_schema(self) -> type[BaseModel]:
        return SummarizerInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return SummarizerOutput

    @property
    def tags(self) -> list[str]:
        return ["document", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Summarize the provided text."""
        assert isinstance(validated_input, SummarizerInput)  # noqa: S101

        text = validated_input.text.strip()
        if not text:
            msg = "Text is empty."
            raise ToolExecutionError(msg, tool_name="summarizer")

        sentences = _SENTENCE_END.split(text)
        selected = sentences[: validated_input.max_sentences]
        summary = " ".join(s.strip() for s in selected if s.strip())

        return SummarizerOutput(
            summary=summary,
            original_length=len(text),
            summary_length=len(summary),
        )
