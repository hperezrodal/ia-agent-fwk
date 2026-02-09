"""Customer support built-in tools.

Provides tools for ticket lookup, FAQ search, escalation, and response
drafting. All tools return mock/in-memory data and are intended as
reference implementations for the customer support agent example.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.tools.base import Tool
from ia_agent_fwk.tools.exceptions import ToolExecutionError

if TYPE_CHECKING:
    from ia_agent_fwk.tools.base import ToolContext

# ---------------------------------------------------------------------------
# Default data directory (relative to this file for standalone use)
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "examples" / "customer_support" / "data"
)

# ---------------------------------------------------------------------------
# TicketLookupTool
# ---------------------------------------------------------------------------


class TicketLookupInput(BaseModel):
    """Input schema for the ticket lookup tool."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str = Field(description="The support ticket ID to look up (e.g. 'TKT-001').")


class TicketLookupOutput(BaseModel):
    """Output schema for the ticket lookup tool."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    customer_name: str
    subject: str
    status: str
    priority: str
    description: str
    created_at: str
    updated_at: str


class TicketLookupTool(Tool):
    """Look up a support ticket by ID.

    Loads ticket data from a JSON file or an in-memory dict.
    Returns mock data suitable for demonstration purposes.
    """

    def __init__(
        self,
        tickets: dict[str, dict[str, Any]] | None = None,
        data_path: Path | None = None,
    ) -> None:
        self._tickets: dict[str, dict[str, Any]] = tickets or {}
        if not self._tickets and data_path is not None:
            self._tickets = self._load_tickets(data_path)

    @staticmethod
    def _load_tickets(data_path: Path) -> dict[str, dict[str, Any]]:
        tickets_file = data_path / "tickets.json"
        if tickets_file.exists():
            with tickets_file.open() as f:
                raw: list[dict[str, Any]] = json.load(f)
            return {t["ticket_id"]: t for t in raw}
        return {}

    @property
    def name(self) -> str:
        return "ticket_lookup"

    @property
    def description(self) -> str:
        return "Look up a customer support ticket by its ID and return its details."

    @property
    def input_schema(self) -> type[BaseModel]:
        return TicketLookupInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return TicketLookupOutput

    @property
    def tags(self) -> list[str]:
        return ["support", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Look up ticket by ID."""
        assert isinstance(validated_input, TicketLookupInput)  # noqa: S101
        ticket_id = validated_input.ticket_id.upper()

        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            msg = f"Ticket '{ticket_id}' not found."
            raise ToolExecutionError(msg, tool_name="ticket_lookup")

        return TicketLookupOutput(
            ticket_id=ticket["ticket_id"],
            customer_name=ticket["customer_name"],
            subject=ticket["subject"],
            status=ticket["status"],
            priority=ticket["priority"],
            description=ticket["description"],
            created_at=ticket["created_at"],
            updated_at=ticket["updated_at"],
        )


# ---------------------------------------------------------------------------
# FAQSearchTool
# ---------------------------------------------------------------------------


class FAQSearchInput(BaseModel):
    """Input schema for the FAQ search tool."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Search query to find relevant FAQ entries.")
    max_results: int = Field(default=3, ge=1, le=10, description="Maximum number of results to return.")


class FAQSearchResult(BaseModel):
    """A single FAQ search result."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    category: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class FAQSearchOutput(BaseModel):
    """Output schema for the FAQ search tool."""

    model_config = ConfigDict(frozen=True)

    results: list[FAQSearchResult]
    total_found: int


class FAQSearchTool(Tool):
    """Search the FAQ knowledge base using simple keyword matching.

    Uses an in-memory list of FAQ entries with basic keyword overlap
    scoring. Suitable for demonstration; production use should integrate
    a proper search/RAG backend.
    """

    def __init__(
        self,
        faq_entries: list[dict[str, str]] | None = None,
        data_path: Path | None = None,
    ) -> None:
        self._faq_entries: list[dict[str, str]] = faq_entries or []
        if not self._faq_entries and data_path is not None:
            self._faq_entries = self._load_faq(data_path)

    @staticmethod
    def _load_faq(data_path: Path) -> list[dict[str, str]]:
        faq_file = data_path / "faq.json"
        if faq_file.exists():
            with faq_file.open() as f:
                entries: list[dict[str, str]] = json.load(f)
            return entries
        return []

    @property
    def name(self) -> str:
        return "faq_search"

    @property
    def description(self) -> str:
        return "Search the FAQ knowledge base for answers to common customer questions."

    @property
    def input_schema(self) -> type[BaseModel]:
        return FAQSearchInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return FAQSearchOutput

    @property
    def tags(self) -> list[str]:
        return ["support", "builtin"]

    def _score_entry(self, query: str, entry: dict[str, str]) -> float:
        """Compute a simple keyword-overlap relevance score."""
        query_words = set(query.lower().split())
        entry_words = set(entry.get("question", "").lower().split()) | set(entry.get("answer", "").lower().split())
        if not query_words:
            return 0.0
        overlap = query_words & entry_words
        return len(overlap) / len(query_words)

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Search FAQ entries by keyword overlap."""
        assert isinstance(validated_input, FAQSearchInput)  # noqa: S101

        scored: list[tuple[float, dict[str, str]]] = []
        for entry in self._faq_entries:
            score = self._score_entry(validated_input.query, entry)
            if score > 0.0:
                scored.append((score, entry))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[FAQSearchResult] = []
        for score, entry in scored[: validated_input.max_results]:
            results.append(
                FAQSearchResult(
                    question=entry.get("question", ""),
                    answer=entry.get("answer", ""),
                    category=entry.get("category", "general"),
                    relevance_score=round(score, 2),
                )
            )

        return FAQSearchOutput(results=results, total_found=len(scored))


# ---------------------------------------------------------------------------
# EscalationTool
# ---------------------------------------------------------------------------


class EscalationInput(BaseModel):
    """Input schema for the escalation tool."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str = Field(description="The ticket ID to escalate.")
    reason: str = Field(description="Reason for escalation.")
    priority: str = Field(default="high", description="Escalation priority level (low, medium, high, urgent).")


class EscalationOutput(BaseModel):
    """Output schema for the escalation tool."""

    model_config = ConfigDict(frozen=True)

    escalation_id: str
    ticket_id: str
    assigned_to: str
    status: str
    message: str


class EscalationTool(Tool):
    """Escalate a support ticket to a human agent.

    Returns a mock escalation confirmation. In production, this would
    integrate with a ticketing system API.
    """

    def __init__(self) -> None:
        self._escalation_counter = 0

    @property
    def name(self) -> str:
        return "escalation"

    @property
    def description(self) -> str:
        return "Escalate a support ticket to a human agent for manual review."

    @property
    def input_schema(self) -> type[BaseModel]:
        return EscalationInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return EscalationOutput

    @property
    def tags(self) -> list[str]:
        return ["support", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Create a mock escalation."""
        assert isinstance(validated_input, EscalationInput)  # noqa: S101
        self._escalation_counter += 1
        escalation_id = f"ESC-{self._escalation_counter:04d}"

        return EscalationOutput(
            escalation_id=escalation_id,
            ticket_id=validated_input.ticket_id,
            assigned_to="Senior Support Agent",
            status="escalated",
            message=(
                f"Ticket {validated_input.ticket_id} has been escalated to a human agent. "
                f"Reason: {validated_input.reason}. Priority: {validated_input.priority}."
            ),
        )


# ---------------------------------------------------------------------------
# ResponseDraftTool
# ---------------------------------------------------------------------------


class ResponseDraftInput(BaseModel):
    """Input schema for the response draft tool."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str = Field(description="The ticket ID this response is for.")
    customer_name: str = Field(description="The customer's name for personalization.")
    issue_summary: str = Field(description="Summary of the customer's issue.")
    resolution: str = Field(description="The proposed resolution or answer.")
    tone: str = Field(default="professional", description="Desired tone (professional, friendly, formal).")


class ResponseDraftOutput(BaseModel):
    """Output schema for the response draft tool."""

    model_config = ConfigDict(frozen=True)

    draft: str
    ticket_id: str
    word_count: int


class ResponseDraftTool(Tool):
    """Draft a customer support response based on context.

    Generates a templated response. In production, this could use
    an LLM for more natural language generation.
    """

    @property
    def name(self) -> str:
        return "response_draft"

    @property
    def description(self) -> str:
        return "Draft a customer support response email based on ticket context and resolution."

    @property
    def input_schema(self) -> type[BaseModel]:
        return ResponseDraftInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return ResponseDraftOutput

    @property
    def tags(self) -> list[str]:
        return ["support", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Draft a response from a template."""
        assert isinstance(validated_input, ResponseDraftInput)  # noqa: S101

        greeting = self._greeting(validated_input.tone, validated_input.customer_name)
        closing = self._closing(validated_input.tone)

        draft = (
            f"{greeting}\n\n"
            f"Thank you for reaching out regarding your issue: {validated_input.issue_summary}\n\n"
            f"{validated_input.resolution}\n\n"
            f"If you have any further questions, please don't hesitate to reply to this "
            f"message or reference ticket {validated_input.ticket_id}.\n\n"
            f"{closing}"
        )

        word_count = len(draft.split())

        return ResponseDraftOutput(
            draft=draft,
            ticket_id=validated_input.ticket_id,
            word_count=word_count,
        )

    @staticmethod
    def _greeting(tone: str, customer_name: str) -> str:
        if tone == "friendly":
            return f"Hi {customer_name}!"
        if tone == "formal":
            return f"Dear {customer_name},"
        return f"Hello {customer_name},"

    @staticmethod
    def _closing(tone: str) -> str:
        if tone == "friendly":
            return "Cheers,\nThe Support Team"
        if tone == "formal":
            return "Respectfully,\nCustomer Support Department"
        return "Best regards,\nCustomer Support Team"
