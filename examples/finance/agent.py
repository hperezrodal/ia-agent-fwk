"""Finance agent example using the ia_agent_fwk framework.

Demonstrates a financial analysis agent configured with four finance tools:
``FinancialDataLookupTool``, ``RatioCalculatorTool``, ``AnomalyDetectorTool``,
and ``ReportGeneratorTool``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.tools.builtin.finance_tools import (
    AnomalyDetectorTool,
    FinancialDataLookupTool,
    RatioCalculatorTool,
    ReportGeneratorTool,
)
from ia_agent_fwk.tools.permissions import PermissionMode, ToolPermissionManager
from ia_agent_fwk.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from ia_agent_fwk.agents.protocols import ToolExecutor
    from ia_agent_fwk.llm.base import LLMProvider

_FINANCE_SYSTEM_PROMPT = """\
You are a financial analysis assistant. Your role is to analyze financial data,
calculate key ratios, detect anomalies, and generate structured reports.

You have access to the following tools:
- financial_data_lookup: Retrieve data from financial statements
- ratio_calculator: Calculate financial ratios (profit margin, ROE, etc.)
- anomaly_detector: Detect anomalous transactions using statistical analysis
- report_generator: Generate structured financial reports in JSON format

Guidelines:
1. Always base your analysis on the provided financial data.
2. When calculating ratios, use the exact values from financial statements.
3. Present anomaly findings with clear explanations and z-scores.
4. Structure your final output as valid JSON with financial insights.
5. Include both quantitative metrics and qualitative interpretations.

Your output should be structured JSON containing analysis results.
"""


class FinanceAgent(Agent):
    """Financial analysis agent with built-in finance tools.

    Extends the Agent ABC to provide a specialised financial analysis agent
    pre-configured with a finance-oriented system prompt and four financial
    tools.
    """

    @property
    def agent_type(self) -> str:
        return "finance"


def create_finance_agent(
    provider: LLMProvider,
    tool_executor: ToolExecutor | None = None,
    *,
    max_iterations: int = 10,
    execution_timeout: int = 300,
) -> FinanceAgent:
    """Create a pre-configured FinanceAgent.

    Parameters
    ----------
    provider:
        LLM provider instance.
    tool_executor:
        Optional tool executor. If not supplied one will be built from
        a fresh registry containing the four finance tools.
    max_iterations:
        Maximum reasoning iterations.
    execution_timeout:
        Execution timeout in seconds.

    Returns
    -------
    FinanceAgent
        A ready-to-use finance agent.

    """
    config = AgentConfig(
        name="finance-agent",
        agent_type="finance",
        system_prompt=_FINANCE_SYSTEM_PROMPT,
        max_iterations=max_iterations,
        execution_timeout=execution_timeout,
        tools=[
            "financial_data_lookup",
            "ratio_calculator",
            "anomaly_detector",
            "report_generator",
        ],
    )

    if tool_executor is None:
        from ia_agent_fwk.tools.executor import DefaultToolExecutor  # noqa: PLC0415

        registry = ToolRegistry()
        registry.register(FinancialDataLookupTool())
        registry.register(RatioCalculatorTool())
        registry.register(AnomalyDetectorTool())
        registry.register(ReportGeneratorTool())

        permission_manager = ToolPermissionManager(default_mode=PermissionMode.allow_all)

        tool_executor = DefaultToolExecutor(
            registry=registry,
            permission_manager=permission_manager,
            agent_id="finance-agent",
        )

    return FinanceAgent(config=config, provider=provider, tool_executor=tool_executor)
