"""Finance built-in tools for financial analysis.

Provides four tools: ``FinancialDataLookupTool``, ``RatioCalculatorTool``,
``AnomalyDetectorTool``, and ``ReportGeneratorTool``.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.tools.base import Tool
from ia_agent_fwk.tools.exceptions import ToolExecutionError

if TYPE_CHECKING:
    from ia_agent_fwk.tools.base import ToolContext

# ---------------------------------------------------------------------------
# FinancialDataLookupTool
# ---------------------------------------------------------------------------

_VALID_STATEMENT_TYPES = ("income_statement", "balance_sheet")
_TOOL_TAG = "finance"


class FinancialDataLookupInput(BaseModel):
    """Input schema for the financial data lookup tool."""

    model_config = ConfigDict(frozen=True)

    statement_type: str = Field(
        description="Type of financial statement: 'income_statement' or 'balance_sheet'.",
    )
    period: str = Field(
        description="Period to look up, e.g. '2024-Q1'.",
    )
    metric: str | None = Field(
        default=None,
        description="Specific metric to retrieve. If None, returns all metrics for the period.",
    )
    data_source: str | None = Field(
        default=None,
        description="JSON string containing financial statements data. If None, returns an error asking for data.",
    )


class FinancialDataLookupOutput(BaseModel):
    """Output schema for the financial data lookup tool."""

    model_config = ConfigDict(frozen=True)

    statement_type: str
    period: str
    data: dict[str, Any]


def _parse_financial_data(data_source: str | None, tool_name: str) -> dict[str, Any]:
    """Parse and validate the data_source JSON string."""
    if data_source is None:
        msg = "No data_source provided. Supply financial statements JSON data."
        raise ToolExecutionError(msg, tool_name=tool_name)
    try:
        result: dict[str, Any] = json.loads(data_source)
    except (json.JSONDecodeError, TypeError) as exc:
        msg = f"Invalid data_source JSON: {exc}"
        raise ToolExecutionError(msg, tool_name=tool_name) from exc
    return result


def _find_period_data(
    statements: Any,
    period: str,
    statement_type: str,
    tool_name: str,
) -> dict[str, Any]:
    """Find period data from a list or dict of statements."""
    period_data: dict[str, Any] | None = None
    if isinstance(statements, list):
        for entry in statements:
            if isinstance(entry, dict) and entry.get("period") == period:
                period_data = entry
                break
    elif isinstance(statements, dict):
        period_data = statements.get(period)

    if period_data is None:
        msg = f"No data found for period '{period}' in '{statement_type}'."
        raise ToolExecutionError(msg, tool_name=tool_name)
    return period_data


class FinancialDataLookupTool(Tool):
    """Look up financial data from structured financial statements.

    Retrieves revenue, expenses, and other financial metrics from
    income statements and balance sheets.
    """

    @property
    def name(self) -> str:
        return "financial_data_lookup"

    @property
    def description(self) -> str:
        return (
            "Look up financial data (revenue, expenses, assets, liabilities) "
            "from income statements and balance sheets for a given period."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return FinancialDataLookupInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return FinancialDataLookupOutput

    @property
    def tags(self) -> list[str]:
        return [_TOOL_TAG, "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Look up financial data from the provided data source."""
        assert isinstance(validated_input, FinancialDataLookupInput)  # noqa: S101

        if validated_input.statement_type not in _VALID_STATEMENT_TYPES:
            msg = (
                f"Invalid statement_type '{validated_input.statement_type}'. "
                f"Must be one of: {', '.join(_VALID_STATEMENT_TYPES)}"
            )
            raise ToolExecutionError(msg, tool_name=self.name)

        all_data = _parse_financial_data(validated_input.data_source, self.name)

        statements = all_data.get(validated_input.statement_type)
        if statements is None:
            msg = f"No '{validated_input.statement_type}' found in data source."
            raise ToolExecutionError(msg, tool_name=self.name)

        period_data = _find_period_data(statements, validated_input.period, validated_input.statement_type, self.name)

        if validated_input.metric is not None:
            metric_val = period_data.get(validated_input.metric)
            if metric_val is None:
                msg = f"Metric '{validated_input.metric}' not found for period '{validated_input.period}'."
                raise ToolExecutionError(msg, tool_name=self.name)
            return FinancialDataLookupOutput(
                statement_type=validated_input.statement_type,
                period=validated_input.period,
                data={validated_input.metric: metric_val},
            )

        return FinancialDataLookupOutput(
            statement_type=validated_input.statement_type,
            period=validated_input.period,
            data=period_data,
        )


# ---------------------------------------------------------------------------
# RatioCalculatorTool
# ---------------------------------------------------------------------------

_SUPPORTED_RATIOS = (
    "profit_margin",
    "gross_margin",
    "roe",
    "roa",
    "current_ratio",
    "debt_to_equity",
    "quick_ratio",
)


class RatioCalculatorInput(BaseModel):
    """Input schema for the ratio calculator tool."""

    model_config = ConfigDict(frozen=True)

    ratio_name: str = Field(
        description=(
            "Financial ratio to compute. Supported: "
            "profit_margin, gross_margin, roe, roa, current_ratio, "
            "debt_to_equity, quick_ratio."
        ),
    )
    values: dict[str, float] = Field(
        description=(
            "Named numeric values required for the ratio. "
            "E.g. {'net_income': 500000, 'revenue': 2000000} for profit_margin."
        ),
    )


class RatioCalculatorOutput(BaseModel):
    """Output schema for the ratio calculator tool."""

    model_config = ConfigDict(frozen=True)

    ratio_name: str
    value: float
    interpretation: str


def _require_values(ratio_name: str, values: dict[str, float], *keys: str) -> list[float]:
    """Validate and return required values for a ratio computation."""
    missing = [k for k in keys if k not in values]
    if missing:
        msg = f"Missing required values for '{ratio_name}': {', '.join(missing)}"
        raise ToolExecutionError(msg, tool_name="ratio_calculator")
    return [values[k] for k in keys]


def _safe_divide(numerator: float, denominator: float, label: str) -> float:
    """Divide numerator by denominator, raising on zero."""
    if denominator == 0:
        msg = f"Cannot compute {label}: division by zero."
        raise ToolExecutionError(msg, tool_name="ratio_calculator")
    return numerator / denominator


def _ratio_profit_margin(values: dict[str, float]) -> tuple[float, str]:
    net_income, revenue = _require_values("profit_margin", values, "net_income", "revenue")
    ratio = _safe_divide(net_income, revenue, "profit_margin")
    pct = ratio * 100
    interp = "strong" if pct > 20 else ("moderate" if pct > 10 else "low")  # noqa: PLR2004
    return ratio, f"Profit margin is {pct:.1f}% ({interp} profitability)."


def _ratio_gross_margin(values: dict[str, float]) -> tuple[float, str]:
    gross_profit, revenue = _require_values("gross_margin", values, "gross_profit", "revenue")
    ratio = _safe_divide(gross_profit, revenue, "gross_margin")
    pct = ratio * 100
    interp = "strong" if pct > 40 else ("moderate" if pct > 20 else "low")  # noqa: PLR2004
    return ratio, f"Gross margin is {pct:.1f}% ({interp})."


def _ratio_roe(values: dict[str, float]) -> tuple[float, str]:
    net_income, equity = _require_values("roe", values, "net_income", "shareholders_equity")
    ratio = _safe_divide(net_income, equity, "ROE")
    pct = ratio * 100
    interp = "excellent" if pct > 15 else ("good" if pct > 10 else "below average")  # noqa: PLR2004
    return ratio, f"Return on equity is {pct:.1f}% ({interp})."


def _ratio_roa(values: dict[str, float]) -> tuple[float, str]:
    net_income, total_assets = _require_values("roa", values, "net_income", "total_assets")
    ratio = _safe_divide(net_income, total_assets, "ROA")
    pct = ratio * 100
    interp = "efficient" if pct > 5 else "below average"  # noqa: PLR2004
    return ratio, f"Return on assets is {pct:.1f}% ({interp} asset utilization)."


def _ratio_current(values: dict[str, float]) -> tuple[float, str]:
    current_assets, current_liabilities = _require_values(
        "current_ratio", values, "current_assets", "current_liabilities"
    )
    ratio = _safe_divide(current_assets, current_liabilities, "current_ratio")
    interp = "strong liquidity" if ratio > 2 else ("adequate" if ratio > 1 else "liquidity risk")  # noqa: PLR2004
    return ratio, f"Current ratio is {ratio:.2f} ({interp})."


def _ratio_debt_to_equity(values: dict[str, float]) -> tuple[float, str]:
    total_debt, equity = _require_values("debt_to_equity", values, "total_debt", "shareholders_equity")
    ratio = _safe_divide(total_debt, equity, "debt_to_equity")
    interp = "high leverage" if ratio > 2 else ("moderate" if ratio > 1 else "conservative")  # noqa: PLR2004
    return ratio, f"Debt-to-equity ratio is {ratio:.2f} ({interp} leverage)."


def _ratio_quick(values: dict[str, float]) -> tuple[float, str]:
    current_assets, inventory, current_liabilities = _require_values(
        "quick_ratio", values, "current_assets", "inventory", "current_liabilities"
    )
    ratio = _safe_divide(current_assets - inventory, current_liabilities, "quick_ratio")
    interp = "strong" if ratio > 1 else "potential liquidity concern"
    return ratio, f"Quick ratio is {ratio:.2f} ({interp})."


_RATIO_DISPATCH: dict[str, Any] = {
    "profit_margin": _ratio_profit_margin,
    "gross_margin": _ratio_gross_margin,
    "roe": _ratio_roe,
    "roa": _ratio_roa,
    "current_ratio": _ratio_current,
    "debt_to_equity": _ratio_debt_to_equity,
    "quick_ratio": _ratio_quick,
}


def _compute_ratio(ratio_name: str, values: dict[str, float]) -> tuple[float, str]:
    """Compute a financial ratio and return (value, interpretation)."""
    handler = _RATIO_DISPATCH.get(ratio_name)
    if handler is None:
        msg = f"Unsupported ratio '{ratio_name}'. Supported: {', '.join(_SUPPORTED_RATIOS)}"
        raise ToolExecutionError(msg, tool_name="ratio_calculator")
    result: tuple[float, str] = handler(values)
    return result


class RatioCalculatorTool(Tool):
    """Calculate financial ratios from provided numeric values.

    Supports profit margin, gross margin, ROE, ROA, current ratio,
    debt-to-equity, and quick ratio.
    """

    @property
    def name(self) -> str:
        return "ratio_calculator"

    @property
    def description(self) -> str:
        return (
            "Calculate financial ratios (profit margin, gross margin, ROE, ROA, "
            "current ratio, debt-to-equity, quick ratio) from provided values."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return RatioCalculatorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return RatioCalculatorOutput

    @property
    def tags(self) -> list[str]:
        return [_TOOL_TAG, "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Calculate the requested financial ratio."""
        assert isinstance(validated_input, RatioCalculatorInput)  # noqa: S101

        value, interpretation = _compute_ratio(validated_input.ratio_name, validated_input.values)
        return RatioCalculatorOutput(
            ratio_name=validated_input.ratio_name,
            value=value,
            interpretation=interpretation,
        )


# ---------------------------------------------------------------------------
# AnomalyDetectorTool
# ---------------------------------------------------------------------------


class AnomalyDetectorInput(BaseModel):
    """Input schema for the anomaly detector tool."""

    model_config = ConfigDict(frozen=True)

    transactions_json: str = Field(
        description="JSON string containing a list of transaction objects with 'amount' and 'description' fields.",
    )
    std_dev_threshold: float = Field(
        default=2.0,
        description="Number of standard deviations from the mean to flag as anomaly.",
        ge=0.5,
    )


class AnomalyRecord(BaseModel):
    """A single detected anomaly."""

    model_config = ConfigDict(frozen=True)

    index: int
    amount: float
    description: str
    z_score: float
    reason: str


class AnomalyDetectorOutput(BaseModel):
    """Output schema for the anomaly detector tool."""

    model_config = ConfigDict(frozen=True)

    total_transactions: int
    anomalies_found: int
    anomalies: list[AnomalyRecord]
    mean_amount: float
    std_dev_amount: float


class AnomalyDetectorTool(Tool):
    """Detect anomalies in transaction data using statistical analysis.

    Uses z-score based detection to identify transactions that deviate
    significantly from the mean.
    """

    @property
    def name(self) -> str:
        return "anomaly_detector"

    @property
    def description(self) -> str:
        return (
            "Detect anomalous transactions using statistical z-score analysis. "
            "Flags transactions that deviate beyond a configurable standard deviation threshold."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return AnomalyDetectorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return AnomalyDetectorOutput

    @property
    def tags(self) -> list[str]:
        return [_TOOL_TAG, "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Detect anomalies in the provided transaction data."""
        assert isinstance(validated_input, AnomalyDetectorInput)  # noqa: S101

        try:
            transactions: list[dict[str, Any]] = json.loads(validated_input.transactions_json)
        except (json.JSONDecodeError, TypeError) as exc:
            msg = f"Invalid transactions JSON: {exc}"
            raise ToolExecutionError(msg, tool_name=self.name) from exc

        if not isinstance(transactions, list) or len(transactions) < 2:  # noqa: PLR2004
            msg = "At least 2 transactions are required for anomaly detection."
            raise ToolExecutionError(msg, tool_name=self.name)

        amounts: list[float] = []
        for i, txn in enumerate(transactions):
            if not isinstance(txn, dict) or "amount" not in txn:
                msg = f"Transaction at index {i} missing 'amount' field."
                raise ToolExecutionError(msg, tool_name=self.name)
            amounts.append(float(txn["amount"]))

        mean = statistics.mean(amounts)
        stdev = statistics.stdev(amounts) if len(amounts) > 1 else 0.0

        anomalies: list[AnomalyRecord] = []
        if stdev > 0:
            for i, (txn, amount) in enumerate(zip(transactions, amounts, strict=True)):
                z_score = (amount - mean) / stdev
                if math.fabs(z_score) > validated_input.std_dev_threshold:
                    direction = "above" if z_score > 0 else "below"
                    anomalies.append(
                        AnomalyRecord(
                            index=i,
                            amount=amount,
                            description=str(txn.get("description", "")),
                            z_score=round(z_score, 4),
                            reason=f"Amount is {math.fabs(z_score):.1f} std devs {direction} the mean.",
                        )
                    )

        return AnomalyDetectorOutput(
            total_transactions=len(transactions),
            anomalies_found=len(anomalies),
            anomalies=anomalies,
            mean_amount=round(mean, 2),
            std_dev_amount=round(stdev, 2),
        )


# ---------------------------------------------------------------------------
# ReportGeneratorTool
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    """A single section in a financial report."""

    model_config = ConfigDict(frozen=True)

    title: str
    content: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class ReportGeneratorInput(BaseModel):
    """Input schema for the report generator tool."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Report title.")
    period: str = Field(description="Reporting period, e.g. '2024-Q1'.")
    sections: list[ReportSection] = Field(
        description="List of report sections with title, content, and optional metrics.",
    )
    currency: str = Field(default="USD", description="Currency code for the report.")


class ReportGeneratorOutput(BaseModel):
    """Output schema for the report generator tool."""

    model_config = ConfigDict(frozen=True)

    report_json: str = Field(description="The generated report as a JSON string.")
    section_count: int
    period: str


class ReportGeneratorTool(Tool):
    """Generate structured financial reports in JSON format.

    Takes report sections with metrics and produces a well-formatted
    financial report.
    """

    @property
    def name(self) -> str:
        return "report_generator"

    @property
    def description(self) -> str:
        return (
            "Generate a structured financial report in JSON format from "
            "provided sections, metrics, and period information."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return ReportGeneratorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return ReportGeneratorOutput

    @property
    def tags(self) -> list[str]:
        return [_TOOL_TAG, "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Generate the financial report."""
        assert isinstance(validated_input, ReportGeneratorInput)  # noqa: S101

        if not validated_input.sections:
            msg = "At least one report section is required."
            raise ToolExecutionError(msg, tool_name=self.name)

        report: dict[str, Any] = {
            "title": validated_input.title,
            "period": validated_input.period,
            "currency": validated_input.currency,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "metrics": section.metrics,
                }
                for section in validated_input.sections
            ],
        }

        report_json = json.dumps(report, indent=2)

        return ReportGeneratorOutput(
            report_json=report_json,
            section_count=len(validated_input.sections),
            period=validated_input.period,
        )
