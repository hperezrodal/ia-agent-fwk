# Finance Agent Example

A reference implementation showing how to use `ia_agent_fwk` for financial analysis.

## Overview

The finance agent demonstrates:
- Looking up financial data from income statements and balance sheets
- Calculating financial ratios (profit margin, ROE, current ratio, etc.)
- Detecting anomalies in transaction data using z-score analysis
- Generating structured financial reports in JSON format

## Tools

| Tool | Description |
|---|---|
| `financial_data_lookup` | Retrieve metrics from financial statements |
| `ratio_calculator` | Compute financial ratios with interpretations |
| `anomaly_detector` | Statistical anomaly detection on transactions |
| `report_generator` | Generate structured JSON reports |

## Quick Start

```python
from examples.finance.agent import create_finance_agent
from ia_agent_fwk.llm.providers.openai import OpenAIProvider
from ia_agent_fwk.llm.config import OpenAISettings

provider = OpenAIProvider(settings=OpenAISettings(api_key="sk-..."))
agent = create_finance_agent(provider=provider)
result = await agent.run("Analyze the Q1 2024 financial performance.")
print(result.output)
```

## Sample Data

- `data/financial_statements.json` -- P&L and balance sheet data for 2024 Q1-Q4
- `data/transactions.json` -- 20 sample transactions with embedded anomalies

## Configuration

See `config.yaml` for an example YAML configuration.

## Running Tests

```bash
python3 -m pytest tests/unit/test_examples/test_finance.py -v
```
