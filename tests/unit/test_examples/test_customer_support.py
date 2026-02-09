"""Unit tests for the customer support agent example.

Tests cover all four support tools, agent creation, and an end-to-end
scenario with a mocked LLM provider.
"""

from __future__ import annotations

from typing import Any

import pytest

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
from ia_agent_fwk.tools.builtin.support_tools import (
    EscalationInput,
    EscalationTool,
    FAQSearchInput,
    FAQSearchTool,
    ResponseDraftInput,
    ResponseDraftTool,
    TicketLookupInput,
    TicketLookupTool,
)
from ia_agent_fwk.tools.exceptions import ToolExecutionError

# ---------------------------------------------------------------------------
# Sample data for tests
# ---------------------------------------------------------------------------

SAMPLE_TICKETS: dict[str, dict[str, Any]] = {
    "TKT-001": {
        "ticket_id": "TKT-001",
        "customer_name": "Alice Johnson",
        "subject": "Unable to log in to my account",
        "status": "open",
        "priority": "high",
        "description": "Cannot log in with correct password.",
        "created_at": "2026-03-08T10:30:00Z",
        "updated_at": "2026-03-08T10:30:00Z",
    },
    "TKT-002": {
        "ticket_id": "TKT-002",
        "customer_name": "Bob Martinez",
        "subject": "Billing discrepancy",
        "status": "open",
        "priority": "medium",
        "description": "Invoice shows $149.99 instead of $99.99.",
        "created_at": "2026-03-07T14:15:00Z",
        "updated_at": "2026-03-08T09:00:00Z",
    },
}

SAMPLE_FAQ: list[dict[str, str]] = [
    {
        "question": "How do I reset my password?",
        "answer": "Go to the login page and click Forgot Password.",
        "category": "account",
    },
    {
        "question": "How do I upgrade my plan?",
        "answer": "Navigate to Settings > Billing > Change Plan.",
        "category": "billing",
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, MasterCard, American Express, and PayPal.",
        "category": "billing",
    },
    {
        "question": "How do I export my data?",
        "answer": "Go to Settings > Data Management > Export.",
        "category": "data",
    },
    {
        "question": "What are the API rate limits?",
        "answer": "Basic plan: 100 requests/minute.",
        "category": "api",
    },
]


@pytest.fixture
def tool_context():
    return ToolContext(execution_id="test-exec-001", agent_id="test-agent")


@pytest.fixture
def ticket_tool():
    return TicketLookupTool(tickets=SAMPLE_TICKETS)


@pytest.fixture
def faq_tool():
    return FAQSearchTool(faq_entries=SAMPLE_FAQ)


@pytest.fixture
def escalation_tool():
    return EscalationTool()


@pytest.fixture
def response_tool():
    return ResponseDraftTool()


# ===========================================================================
# TicketLookupTool tests
# ===========================================================================


@pytest.mark.unit
class TestTicketLookupTool:
    async def test_lookup_existing_ticket(self, ticket_tool, tool_context):
        inp = TicketLookupInput(ticket_id="TKT-001")
        result = await ticket_tool.execute(inp, tool_context)
        assert result.ticket_id == "TKT-001"
        assert result.customer_name == "Alice Johnson"
        assert result.status == "open"
        assert result.priority == "high"

    async def test_lookup_case_insensitive(self, ticket_tool, tool_context):
        inp = TicketLookupInput(ticket_id="tkt-001")
        result = await ticket_tool.execute(inp, tool_context)
        assert result.ticket_id == "TKT-001"

    async def test_lookup_nonexistent_ticket(self, ticket_tool, tool_context):
        inp = TicketLookupInput(ticket_id="TKT-999")
        with pytest.raises(ToolExecutionError, match="not found"):
            await ticket_tool.execute(inp, tool_context)

    def test_tool_properties(self, ticket_tool):
        assert ticket_tool.name == "ticket_lookup"
        assert "ticket" in ticket_tool.description.lower()
        assert "support" in ticket_tool.tags
        assert "builtin" in ticket_tool.tags

    def test_input_output_schemas(self, ticket_tool):
        assert ticket_tool.input_schema is TicketLookupInput
        from ia_agent_fwk.tools.builtin.support_tools import TicketLookupOutput

        assert ticket_tool.output_schema is TicketLookupOutput

    async def test_nonexistent_data_path_fallback(self, tool_context):
        """TicketLookupTool with nonexistent data_path falls back to empty dict."""
        from pathlib import Path

        tool = TicketLookupTool(data_path=Path("/nonexistent/path/that/does/not/exist"))
        assert tool._tickets == {}
        inp = TicketLookupInput(ticket_id="TKT-001")
        with pytest.raises(ToolExecutionError, match="not found"):
            await tool.execute(inp, tool_context)


# ===========================================================================
# FAQSearchTool tests
# ===========================================================================


@pytest.mark.unit
class TestFAQSearchTool:
    async def test_search_finds_results(self, faq_tool, tool_context):
        inp = FAQSearchInput(query="reset password")
        result = await faq_tool.execute(inp, tool_context)
        assert result.total_found >= 1
        assert len(result.results) >= 1
        assert any("password" in r.question.lower() for r in result.results)

    async def test_search_respects_max_results(self, faq_tool, tool_context):
        inp = FAQSearchInput(query="how do I", max_results=2)
        result = await faq_tool.execute(inp, tool_context)
        assert len(result.results) <= 2

    async def test_search_no_results(self, faq_tool, tool_context):
        inp = FAQSearchInput(query="xyznonexistent")
        result = await faq_tool.execute(inp, tool_context)
        assert result.total_found == 0
        assert len(result.results) == 0

    async def test_search_relevance_ordering(self, faq_tool, tool_context):
        inp = FAQSearchInput(query="payment methods accept", max_results=5)
        result = await faq_tool.execute(inp, tool_context)
        if len(result.results) >= 2:
            scores = [r.relevance_score for r in result.results]
            assert scores == sorted(scores, reverse=True)

    def test_tool_properties(self, faq_tool):
        assert faq_tool.name == "faq_search"
        assert "faq" in faq_tool.description.lower()
        assert "support" in faq_tool.tags

    def test_input_output_schemas(self, faq_tool):
        assert faq_tool.input_schema is FAQSearchInput
        from ia_agent_fwk.tools.builtin.support_tools import FAQSearchOutput

        assert faq_tool.output_schema is FAQSearchOutput

    async def test_empty_query(self, faq_tool, tool_context):
        inp = FAQSearchInput(query="")
        result = await faq_tool.execute(inp, tool_context)
        assert result.total_found == 0

    async def test_nonexistent_data_path_fallback(self, tool_context):
        """FAQSearchTool with nonexistent data_path falls back to empty list."""
        from pathlib import Path

        tool = FAQSearchTool(data_path=Path("/nonexistent/path/that/does/not/exist"))
        assert tool._faq_entries == []
        inp = FAQSearchInput(query="reset password")
        result = await tool.execute(inp, tool_context)
        assert result.total_found == 0
        assert result.results == []


# ===========================================================================
# EscalationTool tests
# ===========================================================================


@pytest.mark.unit
class TestEscalationTool:
    async def test_escalate_ticket(self, escalation_tool, tool_context):
        inp = EscalationInput(
            ticket_id="TKT-001",
            reason="Customer is very frustrated",
            priority="urgent",
        )
        result = await escalation_tool.execute(inp, tool_context)
        assert result.escalation_id == "ESC-0001"
        assert result.ticket_id == "TKT-001"
        assert result.status == "escalated"
        assert "frustrated" in result.message
        assert "urgent" in result.message

    async def test_escalation_counter_increments(self, escalation_tool, tool_context):
        inp1 = EscalationInput(ticket_id="TKT-001", reason="Issue 1")
        inp2 = EscalationInput(ticket_id="TKT-002", reason="Issue 2")
        r1 = await escalation_tool.execute(inp1, tool_context)
        r2 = await escalation_tool.execute(inp2, tool_context)
        assert r1.escalation_id == "ESC-0001"
        assert r2.escalation_id == "ESC-0002"

    async def test_default_priority(self, escalation_tool, tool_context):
        inp = EscalationInput(ticket_id="TKT-001", reason="Needs attention")
        result = await escalation_tool.execute(inp, tool_context)
        assert "high" in result.message

    def test_tool_properties(self, escalation_tool):
        assert escalation_tool.name == "escalation"
        assert "escalat" in escalation_tool.description.lower()
        assert "support" in escalation_tool.tags


# ===========================================================================
# ResponseDraftTool tests
# ===========================================================================


@pytest.mark.unit
class TestResponseDraftTool:
    async def test_draft_professional_response(self, response_tool, tool_context):
        inp = ResponseDraftInput(
            ticket_id="TKT-001",
            customer_name="Alice",
            issue_summary="login issue",
            resolution="Password has been reset.",
            tone="professional",
        )
        result = await response_tool.execute(inp, tool_context)
        assert "Hello Alice," in result.draft
        assert "login issue" in result.draft
        assert "Password has been reset." in result.draft
        assert "TKT-001" in result.draft
        assert result.ticket_id == "TKT-001"
        assert result.word_count > 0

    async def test_draft_friendly_response(self, response_tool, tool_context):
        inp = ResponseDraftInput(
            ticket_id="TKT-002",
            customer_name="Bob",
            issue_summary="billing issue",
            resolution="Credit applied.",
            tone="friendly",
        )
        result = await response_tool.execute(inp, tool_context)
        assert "Hi Bob!" in result.draft
        assert "Cheers" in result.draft

    async def test_draft_formal_response(self, response_tool, tool_context):
        inp = ResponseDraftInput(
            ticket_id="TKT-003",
            customer_name="Carol",
            issue_summary="data export",
            resolution="Issue resolved.",
            tone="formal",
        )
        result = await response_tool.execute(inp, tool_context)
        assert "Dear Carol," in result.draft
        assert "Respectfully" in result.draft

    def test_tool_properties(self, response_tool):
        assert response_tool.name == "response_draft"
        assert "draft" in response_tool.description.lower()
        assert "support" in response_tool.tags

    def test_input_output_schemas(self, response_tool):
        assert response_tool.input_schema is ResponseDraftInput
        from ia_agent_fwk.tools.builtin.support_tools import ResponseDraftOutput

        assert response_tool.output_schema is ResponseDraftOutput


# ===========================================================================
# Agent creation tests
# ===========================================================================


@pytest.mark.unit
class TestCustomerSupportAgentCreation:
    def _make_mock_provider(self):
        """Create a minimal mock LLM provider."""
        from ia_agent_fwk.llm.base import LLMProvider
        from ia_agent_fwk.llm.models import HealthStatus

        class _MockProvider(LLMProvider):
            def __init__(self):
                self.provider_name = "mock"

            async def complete(self, prompt, **kwargs):
                raise NotImplementedError

            async def chat(self, messages, **kwargs):
                return ChatResponse(
                    message=Message(role="assistant", content="Hello!"),
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                    model="mock-model",
                    finish_reason=FinishReason.stop,
                )

            async def stream(self, messages, **kwargs):
                raise NotImplementedError
                yield  # type: ignore[misc]  # pragma: no cover

            def count_tokens(self, text, model=None):
                return len(text) // 4

            async def health_check(self):
                return HealthStatus(status="healthy")

        return _MockProvider()

    def test_agent_creation_with_factory(self):
        from examples.customer_support.agent import create_support_agent

        provider = self._make_mock_provider()
        agent = create_support_agent(provider)

        assert agent.agent_type == "customer_support"
        assert agent.config.name == "customer-support-agent"
        assert "ticket_lookup" in agent.config.tools
        assert "faq_search" in agent.config.tools
        assert "escalation" in agent.config.tools
        assert "response_draft" in agent.config.tools

    def test_agent_creation_with_overrides(self):
        from examples.customer_support.agent import create_support_agent

        provider = self._make_mock_provider()
        agent = create_support_agent(
            provider,
            config_overrides={"max_iterations": 5, "execution_timeout": 60},
        )
        assert agent.config.max_iterations == 5
        assert agent.config.execution_timeout == 60

    def test_agent_initial_state_is_idle(self):
        from examples.customer_support.agent import create_support_agent

        provider = self._make_mock_provider()
        agent = create_support_agent(provider)
        assert agent.state == AgentState.IDLE

    def test_agent_has_system_prompt(self):
        from examples.customer_support.agent import create_support_agent

        provider = self._make_mock_provider()
        agent = create_support_agent(provider)
        assert "customer support" in agent.config.system_prompt.lower()
        assert "ticket_lookup" in agent.config.system_prompt


# ===========================================================================
# End-to-end test with mocked LLM
# ===========================================================================


@pytest.mark.unit
class TestCustomerSupportE2E:
    async def test_agent_handles_support_query(self):
        """End-to-end test: agent processes a support query with mocked LLM."""
        from examples.customer_support.agent import CustomerSupportAgent

        # Build a mock provider that returns a simple response (no tool calls)
        from ia_agent_fwk.llm.base import LLMProvider
        from ia_agent_fwk.llm.models import HealthStatus

        class _E2EProvider(LLMProvider):
            def __init__(self):
                self.provider_name = "mock"

            async def complete(self, prompt, **kwargs):
                raise NotImplementedError

            async def chat(self, messages, **kwargs):
                return ChatResponse(
                    message=Message(
                        role="assistant",
                        content=(
                            "I'd be happy to help you with your login issue. "
                            "Please try resetting your password using the "
                            "'Forgot Password' link on the login page."
                        ),
                    ),
                    usage=TokenUsage(prompt_tokens=50, completion_tokens=30),
                    model="mock-model",
                    finish_reason=FinishReason.stop,
                )

            async def stream(self, messages, **kwargs):
                raise NotImplementedError
                yield  # type: ignore[misc]  # pragma: no cover

            def count_tokens(self, text, model=None):
                return len(text) // 4

            async def health_check(self):
                return HealthStatus(status="healthy")

        provider = _E2EProvider()

        config = AgentConfig(
            name="test-support-agent",
            agent_type="customer_support",
            system_prompt="You are a support agent.",
            provider_name="mock",
            max_iterations=5,
            execution_timeout=30,
        )

        agent = CustomerSupportAgent(config=config, provider=provider)
        result = await agent.run("I can't log into my account. Help!")

        assert result.state == AgentState.COMPLETED
        assert result.output
        assert "password" in result.output.lower() or "login" in result.output.lower()
        assert result.iterations >= 1
        assert result.usage.total_tokens > 0

    async def test_agent_handles_tool_call_flow(self):
        """Test agent with a tool call response followed by a final answer."""
        from examples.customer_support.agent import CustomerSupportAgent

        from ia_agent_fwk.llm.base import LLMProvider
        from ia_agent_fwk.llm.models import HealthStatus
        from ia_agent_fwk.tools.builtin.support_tools import TicketLookupTool
        from ia_agent_fwk.tools.executor import DefaultToolExecutor
        from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
        from ia_agent_fwk.tools.registry import ToolRegistry

        call_count = 0

        class _ToolCallProvider(LLMProvider):
            def __init__(self):
                self.provider_name = "mock"

            async def complete(self, prompt, **kwargs):
                raise NotImplementedError

            async def chat(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call: request a tool call
                    return ChatResponse(
                        message=Message(
                            role="assistant",
                            content="Let me look up that ticket for you.",
                            tool_calls=[
                                ToolCall(
                                    id="tc-1",
                                    name="ticket_lookup",
                                    arguments='{"ticket_id": "TKT-001"}',
                                ),
                            ],
                        ),
                        usage=TokenUsage(prompt_tokens=30, completion_tokens=20),
                        model="mock-model",
                        finish_reason=FinishReason.tool_calls,
                    )
                # Second call: final answer after tool result
                return ChatResponse(
                    message=Message(
                        role="assistant",
                        content=(
                            "I found your ticket TKT-001. It's about a login issue "
                            "and has high priority. Our team is working on it."
                        ),
                    ),
                    usage=TokenUsage(prompt_tokens=60, completion_tokens=30),
                    model="mock-model",
                    finish_reason=FinishReason.stop,
                )

            async def stream(self, messages, **kwargs):
                raise NotImplementedError
                yield  # type: ignore[misc]  # pragma: no cover

            def count_tokens(self, text, model=None):
                return len(text) // 4

            async def health_check(self):
                return HealthStatus(status="healthy")

        provider = _ToolCallProvider()

        # Set up registry with ticket tool
        registry = ToolRegistry()
        registry.register(TicketLookupTool(tickets=SAMPLE_TICKETS))
        permission_manager = ToolPermissionManager(default_mode=PermissionMode.allow_all)
        executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=permission_manager,
            agent_id="test-agent",
        )

        config = AgentConfig(
            name="test-support-agent",
            agent_type="customer_support",
            system_prompt="You are a support agent.",
            provider_name="mock",
            max_iterations=5,
            execution_timeout=30,
            tools=["ticket_lookup"],
        )

        agent = CustomerSupportAgent(
            config=config,
            provider=provider,
            tool_executor=executor,
        )
        result = await agent.run("Can you look up ticket TKT-001?")

        assert result.state == AgentState.COMPLETED
        assert "TKT-001" in result.output
        assert result.iterations >= 2
        assert call_count == 2
