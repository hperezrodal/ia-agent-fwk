"""Customer support agent example.

Demonstrates how to build a customer support agent using the ia-agent-fwk
framework. The agent is configured with four support-specific tools and
a system prompt that guides its behavior for handling customer inquiries.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.tools.builtin.support_tools import (
    EscalationTool,
    FAQSearchTool,
    ResponseDraftTool,
    TicketLookupTool,
)
from ia_agent_fwk.tools.executor import DefaultToolExecutor
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
from ia_agent_fwk.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from ia_agent_fwk.agents.protocols import ToolExecutor
    from ia_agent_fwk.llm.base import LLMProvider

_DATA_DIR = Path(__file__).resolve().parent / "data"

_SYSTEM_PROMPT = """\
You are a helpful and empathetic customer support agent. Your goal is to \
assist customers with their inquiries efficiently and professionally.

Guidelines:
1. Always greet the customer and acknowledge their issue.
2. Use the ticket_lookup tool to retrieve ticket details when a ticket ID \
is mentioned.
3. Search the FAQ knowledge base with faq_search before escalating.
4. Only escalate to a human agent when the issue cannot be resolved with \
available information.
5. Use response_draft to compose well-structured replies.
6. Maintain a professional yet friendly tone.
7. Never share internal system details or sensitive information.
8. If you cannot resolve an issue, clearly explain next steps.
"""


class CustomerSupportAgent(Agent):
    """Customer support agent with ticket, FAQ, escalation, and drafting tools.

    This agent demonstrates a complete support workflow using the
    ia-agent-fwk tool and agent system.
    """

    @property
    def agent_type(self) -> str:
        return "customer_support"


def create_support_agent(
    provider: LLMProvider,
    *,
    data_dir: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> CustomerSupportAgent:
    """Create a fully configured customer support agent.

    Parameters
    ----------
    provider:
        The LLM provider instance to use for reasoning.
    data_dir:
        Path to the directory containing tickets.json and faq.json.
        Defaults to the ``data/`` folder next to this module.
    config_overrides:
        Optional dict of AgentConfig field overrides.

    Returns
    -------
    CustomerSupportAgent
        A ready-to-use customer support agent instance.

    """
    resolved_data_dir = data_dir or _DATA_DIR

    # Build tools
    ticket_tool = TicketLookupTool(data_path=resolved_data_dir)
    faq_tool = FAQSearchTool(data_path=resolved_data_dir)
    escalation_tool = EscalationTool()
    response_tool = ResponseDraftTool()

    # Build registry and register tools
    registry = ToolRegistry()
    registry.register(ticket_tool)
    registry.register(faq_tool)
    registry.register(escalation_tool)
    registry.register(response_tool)

    # Build executor
    permission_manager = ToolPermissionManager(default_mode=PermissionMode.allow_all)
    executor: ToolExecutor = DefaultToolExecutor(
        registry=registry,
        permission_manager=permission_manager,
        agent_id="customer-support-agent",
    )

    # Build config
    defaults: dict[str, Any] = {
        "name": "customer-support-agent",
        "agent_type": "customer_support",
        "system_prompt": _SYSTEM_PROMPT,
        "provider_name": provider.provider_name,
        "max_iterations": 10,
        "execution_timeout": 120,
        "max_tokens_per_response": 4096,
        "tools": ["ticket_lookup", "faq_search", "escalation", "response_draft"],
    }
    if config_overrides:
        defaults.update(config_overrides)

    agent_config = AgentConfig(**defaults)

    return CustomerSupportAgent(
        config=agent_config,
        provider=provider,
        tool_executor=executor,
    )
