"""Tests for the finance agent example and finance tools."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.models import (
    ChatResponse,
    FinishReason,
    Message,
    TokenUsage,
)
from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.builtin.finance_tools import (
    AnomalyDetectorInput,
    AnomalyDetectorTool,
    FinancialDataLookupInput,
    FinancialDataLookupTool,
    RatioCalculatorInput,
    RatioCalculatorTool,
    ReportGeneratorInput,
    ReportGeneratorTool,
    ReportSection,
)
from ia_agent_fwk.tools.exceptions import ToolExecutionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_FINANCIAL_DATA = json.dumps(
    {
        "income_statement": [
            {
                "period": "2024-Q1",
                "revenue": 2500000,
                "cost_of_goods_sold": 1500000,
                "gross_profit": 1000000,
                "operating_expenses": 600000,
                "net_income": 281250,
            },
        ],
        "balance_sheet": [
            {
                "period": "2024-Q1",
                "total_assets": 8500000,
                "current_assets": 3200000,
                "inventory": 700000,
                "current_liabilities": 1800000,
                "total_debt": 4500000,
                "shareholders_equity": 4000000,
            },
        ],
    }
)

SAMPLE_TRANSACTIONS = json.dumps(
    [
        {"amount": 10000, "description": "Normal expense A"},
        {"amount": 12000, "description": "Normal expense B"},
        {"amount": 11000, "description": "Normal expense C"},
        {"amount": 9500, "description": "Normal expense D"},
        {"amount": 10500, "description": "Normal expense E"},
        {"amount": 250000, "description": "Anomalous large payment"},
        {"amount": 11500, "description": "Normal expense F"},
        {"amount": 9000, "description": "Normal expense G"},
    ]
)


def _make_context() -> ToolContext:
    return ToolContext(execution_id="test-exec-001", agent_id="test-agent")


# ---------------------------------------------------------------------------
# FinancialDataLookupTool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFinancialDataLookupTool:
    def test_properties(self):
        tool = FinancialDataLookupTool()
        assert tool.name == "financial_data_lookup"
        assert "finance" in tool.tags
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)

    async def test_lookup_all_metrics(self):
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2024-Q1",
            data_source=SAMPLE_FINANCIAL_DATA,
        )
        result = await tool.execute(inp, _make_context())
        assert result.period == "2024-Q1"
        assert result.data["revenue"] == 2500000
        assert result.data["net_income"] == 281250

    async def test_lookup_specific_metric(self):
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2024-Q1",
            metric="revenue",
            data_source=SAMPLE_FINANCIAL_DATA,
        )
        result = await tool.execute(inp, _make_context())
        assert result.data == {"revenue": 2500000}

    async def test_lookup_balance_sheet(self):
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="balance_sheet",
            period="2024-Q1",
            data_source=SAMPLE_FINANCIAL_DATA,
        )
        result = await tool.execute(inp, _make_context())
        assert result.data["total_assets"] == 8500000

    async def test_invalid_statement_type(self):
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="cash_flow",
            period="2024-Q1",
            data_source=SAMPLE_FINANCIAL_DATA,
        )
        with pytest.raises(ToolExecutionError, match="Invalid statement_type"):
            await tool.execute(inp, _make_context())

    async def test_no_data_source(self):
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2024-Q1",
        )
        with pytest.raises(ToolExecutionError, match="No data_source provided"):
            await tool.execute(inp, _make_context())

    async def test_invalid_json_data_source(self):
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2024-Q1",
            data_source="not valid json{",
        )
        with pytest.raises(ToolExecutionError, match="Invalid data_source JSON"):
            await tool.execute(inp, _make_context())

    async def test_period_not_found(self):
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2099-Q1",
            data_source=SAMPLE_FINANCIAL_DATA,
        )
        with pytest.raises(ToolExecutionError, match="No data found for period"):
            await tool.execute(inp, _make_context())

    async def test_metric_not_found(self):
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2024-Q1",
            metric="nonexistent",
            data_source=SAMPLE_FINANCIAL_DATA,
        )
        with pytest.raises(ToolExecutionError, match="Metric 'nonexistent' not found"):
            await tool.execute(inp, _make_context())

    async def test_lookup_dict_based_period_data(self):
        """Exercise the dict lookup path in _find_period_data (lines 87-88)."""
        dict_based_data = json.dumps(
            {
                "income_statement": {
                    "2024-Q1": {
                        "revenue": 3000000,
                        "net_income": 400000,
                    },
                },
                "balance_sheet": {
                    "2024-Q1": {
                        "total_assets": 9000000,
                    },
                },
            }
        )
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2024-Q1",
            data_source=dict_based_data,
        )
        result = await tool.execute(inp, _make_context())
        assert result.period == "2024-Q1"
        assert result.data["revenue"] == 3000000
        assert result.data["net_income"] == 400000

    async def test_lookup_dict_based_specific_metric(self):
        """Dict-based period data with a specific metric request."""
        dict_based_data = json.dumps(
            {
                "income_statement": {
                    "2024-Q1": {
                        "revenue": 3000000,
                        "net_income": 400000,
                    },
                },
            }
        )
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2024-Q1",
            metric="revenue",
            data_source=dict_based_data,
        )
        result = await tool.execute(inp, _make_context())
        assert result.data == {"revenue": 3000000}

    async def test_lookup_dict_based_period_not_found(self):
        """Dict-based period data where period key is missing."""
        dict_based_data = json.dumps(
            {
                "income_statement": {
                    "2024-Q1": {"revenue": 3000000},
                },
            }
        )
        tool = FinancialDataLookupTool()
        inp = FinancialDataLookupInput(
            statement_type="income_statement",
            period="2099-Q1",
            data_source=dict_based_data,
        )
        with pytest.raises(ToolExecutionError, match="No data found for period"):
            await tool.execute(inp, _make_context())


# ---------------------------------------------------------------------------
# RatioCalculatorTool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRatioCalculatorTool:
    def test_properties(self):
        tool = RatioCalculatorTool()
        assert tool.name == "ratio_calculator"
        assert "finance" in tool.tags
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)

    async def test_profit_margin(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="profit_margin",
            values={"net_income": 500000, "revenue": 2000000},
        )
        result = await tool.execute(inp, _make_context())
        assert result.ratio_name == "profit_margin"
        assert abs(result.value - 0.25) < 1e-6
        assert "25.0%" in result.interpretation

    async def test_gross_margin(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="gross_margin",
            values={"gross_profit": 1000000, "revenue": 2500000},
        )
        result = await tool.execute(inp, _make_context())
        assert abs(result.value - 0.4) < 1e-6
        assert "40.0%" in result.interpretation

    async def test_roe(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="roe",
            values={"net_income": 281250, "shareholders_equity": 4000000},
        )
        result = await tool.execute(inp, _make_context())
        assert result.ratio_name == "roe"
        assert result.value > 0

    async def test_roa(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="roa",
            values={"net_income": 281250, "total_assets": 8500000},
        )
        result = await tool.execute(inp, _make_context())
        assert result.ratio_name == "roa"
        assert result.value > 0

    async def test_current_ratio(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="current_ratio",
            values={"current_assets": 5000000, "current_liabilities": 1800000},
        )
        result = await tool.execute(inp, _make_context())
        assert abs(result.value - 5000000 / 1800000) < 1e-6
        assert "strong liquidity" in result.interpretation.lower()

    async def test_debt_to_equity(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="debt_to_equity",
            values={"total_debt": 4500000, "shareholders_equity": 4000000},
        )
        result = await tool.execute(inp, _make_context())
        assert abs(result.value - 1.125) < 1e-6

    async def test_quick_ratio(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="quick_ratio",
            values={"current_assets": 3200000, "inventory": 700000, "current_liabilities": 1800000},
        )
        result = await tool.execute(inp, _make_context())
        expected = (3200000 - 700000) / 1800000
        assert abs(result.value - expected) < 1e-6

    async def test_unsupported_ratio(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="sharpe_ratio",
            values={"returns": 0.1},
        )
        with pytest.raises(ToolExecutionError, match="Unsupported ratio"):
            await tool.execute(inp, _make_context())

    async def test_missing_values(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="profit_margin",
            values={"net_income": 500000},
        )
        with pytest.raises(ToolExecutionError, match="Missing required values"):
            await tool.execute(inp, _make_context())

    async def test_division_by_zero(self):
        tool = RatioCalculatorTool()
        inp = RatioCalculatorInput(
            ratio_name="profit_margin",
            values={"net_income": 500000, "revenue": 0},
        )
        with pytest.raises(ToolExecutionError, match="division by zero"):
            await tool.execute(inp, _make_context())


# ---------------------------------------------------------------------------
# AnomalyDetectorTool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnomalyDetectorTool:
    def test_properties(self):
        tool = AnomalyDetectorTool()
        assert tool.name == "anomaly_detector"
        assert "finance" in tool.tags
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)

    async def test_detects_anomaly(self):
        tool = AnomalyDetectorTool()
        inp = AnomalyDetectorInput(
            transactions_json=SAMPLE_TRANSACTIONS,
            std_dev_threshold=2.0,
        )
        result = await tool.execute(inp, _make_context())
        assert result.total_transactions == 8
        assert result.anomalies_found >= 1
        # The 250000 transaction should be flagged
        anomaly_amounts = [a.amount for a in result.anomalies]
        assert 250000 in anomaly_amounts

    async def test_no_anomalies_with_high_threshold(self):
        tool = AnomalyDetectorTool()
        uniform = json.dumps(
            [
                {"amount": 100, "description": "A"},
                {"amount": 100, "description": "B"},
                {"amount": 100, "description": "C"},
            ]
        )
        inp = AnomalyDetectorInput(
            transactions_json=uniform,
            std_dev_threshold=2.0,
        )
        result = await tool.execute(inp, _make_context())
        assert result.anomalies_found == 0

    async def test_anomaly_has_z_score(self):
        tool = AnomalyDetectorTool()
        inp = AnomalyDetectorInput(
            transactions_json=SAMPLE_TRANSACTIONS,
            std_dev_threshold=2.0,
        )
        result = await tool.execute(inp, _make_context())
        for anomaly in result.anomalies:
            assert anomaly.z_score != 0
            assert anomaly.reason

    async def test_invalid_json(self):
        tool = AnomalyDetectorTool()
        inp = AnomalyDetectorInput(transactions_json="not json")
        with pytest.raises(ToolExecutionError, match="Invalid transactions JSON"):
            await tool.execute(inp, _make_context())

    async def test_too_few_transactions(self):
        tool = AnomalyDetectorTool()
        inp = AnomalyDetectorInput(
            transactions_json=json.dumps([{"amount": 100, "description": "A"}]),
        )
        with pytest.raises(ToolExecutionError, match="At least 2 transactions"):
            await tool.execute(inp, _make_context())

    async def test_missing_amount_field(self):
        tool = AnomalyDetectorTool()
        inp = AnomalyDetectorInput(
            transactions_json=json.dumps(
                [
                    {"description": "A"},
                    {"amount": 100, "description": "B"},
                ]
            ),
        )
        with pytest.raises(ToolExecutionError, match="missing 'amount' field"):
            await tool.execute(inp, _make_context())

    async def test_mean_and_std_dev_in_output(self):
        tool = AnomalyDetectorTool()
        inp = AnomalyDetectorInput(
            transactions_json=SAMPLE_TRANSACTIONS,
            std_dev_threshold=2.0,
        )
        result = await tool.execute(inp, _make_context())
        assert result.mean_amount > 0
        assert result.std_dev_amount > 0


# ---------------------------------------------------------------------------
# ReportGeneratorTool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReportGeneratorTool:
    def test_properties(self):
        tool = ReportGeneratorTool()
        assert tool.name == "report_generator"
        assert "finance" in tool.tags
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)

    async def test_generates_report(self):
        tool = ReportGeneratorTool()
        inp = ReportGeneratorInput(
            title="Q1 Financial Report",
            period="2024-Q1",
            sections=[
                ReportSection(
                    title="Revenue",
                    content="Revenue grew by 12%.",
                    metrics={"revenue": 2500000, "growth": 0.12},
                ),
                ReportSection(
                    title="Expenses",
                    content="Operating expenses remained stable.",
                    metrics={"operating_expenses": 600000},
                ),
            ],
        )
        result = await tool.execute(inp, _make_context())
        assert result.section_count == 2
        assert result.period == "2024-Q1"
        report = json.loads(result.report_json)
        assert report["title"] == "Q1 Financial Report"
        assert report["currency"] == "USD"
        assert len(report["sections"]) == 2

    async def test_custom_currency(self):
        tool = ReportGeneratorTool()
        inp = ReportGeneratorInput(
            title="Report",
            period="2024-Q1",
            currency="EUR",
            sections=[
                ReportSection(title="Summary", content="Overview."),
            ],
        )
        result = await tool.execute(inp, _make_context())
        report = json.loads(result.report_json)
        assert report["currency"] == "EUR"

    async def test_empty_sections_error(self):
        tool = ReportGeneratorTool()
        inp = ReportGeneratorInput(
            title="Report",
            period="2024-Q1",
            sections=[],
        )
        with pytest.raises(ToolExecutionError, match="At least one report section"):
            await tool.execute(inp, _make_context())


# ---------------------------------------------------------------------------
# FinanceAgent tests
# ---------------------------------------------------------------------------


def _make_mock_chat_response(
    content="Analysis complete.",
    finish_reason=FinishReason.stop,
):
    return ChatResponse(
        message=Message(role="assistant", content=content),
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        model="mock-model",
        finish_reason=finish_reason,
    )


@pytest.mark.unit
class TestFinanceAgent:
    def test_agent_creation(self):
        from examples.finance.agent import FinanceAgent

        from tests.unit.test_agents.conftest import MockLLMProvider

        config = AgentConfig(
            name="finance-agent",
            agent_type="finance",
            system_prompt="You are a financial analyst.",
            max_iterations=10,
            execution_timeout=300,
        )
        provider = MockLLMProvider(responses=[_make_mock_chat_response()])
        agent = FinanceAgent(config=config, provider=provider)
        assert agent.agent_type == "finance"
        assert agent.state == AgentState.IDLE

    def test_create_finance_agent_factory(self):
        from examples.finance.agent import create_finance_agent

        from tests.unit.test_agents.conftest import MockLLMProvider

        provider = MockLLMProvider(responses=[_make_mock_chat_response()])
        agent = create_finance_agent(provider=provider)
        assert agent.agent_type == "finance"
        assert agent.config.name == "finance-agent"
        assert "financial_data_lookup" in agent.config.tools
        assert "ratio_calculator" in agent.config.tools
        assert "anomaly_detector" in agent.config.tools
        assert "report_generator" in agent.config.tools

    async def test_agent_run_with_mocked_llm(self):
        from examples.finance.agent import create_finance_agent

        from tests.unit.test_agents.conftest import MockLLMProvider

        response_content = json.dumps(
            {
                "analysis": "Q1 2024 Financial Summary",
                "revenue": 2500000,
                "profit_margin": "11.25%",
                "insights": ["Revenue is strong", "Profit margin is moderate"],
            }
        )
        provider = MockLLMProvider(
            responses=[_make_mock_chat_response(content=response_content)],
        )
        agent = create_finance_agent(provider=provider)
        result = await agent.run("Analyze the Q1 2024 financial performance.")

        assert result.state == AgentState.COMPLETED
        assert result.output == response_content
        assert result.error is None
        assert result.iterations == 1

        # Verify the output is valid JSON with financial data
        parsed = json.loads(result.output)
        assert "analysis" in parsed
        assert "revenue" in parsed

    async def test_agent_run_produces_structured_output(self):
        from examples.finance.agent import create_finance_agent

        from tests.unit.test_agents.conftest import MockLLMProvider

        structured = {
            "report_title": "Financial Analysis",
            "period": "2024-Q1",
            "ratios": {
                "profit_margin": 0.1125,
                "current_ratio": 1.78,
            },
            "anomalies_detected": 2,
            "recommendation": "Continue monitoring profit margins.",
        }
        provider = MockLLMProvider(
            responses=[_make_mock_chat_response(content=json.dumps(structured))],
        )
        agent = create_finance_agent(provider=provider)
        result = await agent.run("Generate a full financial report for Q1 2024.")

        assert result.state == AgentState.COMPLETED
        parsed = json.loads(result.output)
        assert parsed["period"] == "2024-Q1"
        assert "ratios" in parsed
        assert parsed["anomalies_detected"] == 2
