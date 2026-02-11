# Document Processor Agent Example

A reference implementation showing how to build a document-analysis agent
with the `ia_agent_fwk` framework.

## What it does

The agent processes documents through four built-in tools:

| Tool                 | Purpose                                      |
|----------------------|----------------------------------------------|
| `text_extractor`     | Extract plain text and compute statistics    |
| `section_identifier` | Detect section headings and structure        |
| `entity_extractor`   | Find dates, amounts, emails, and names       |
| `summarizer`         | Produce an extractive summary                |

## Quick start

```bash
# 1. Install the framework (from the repo root)
pip install -e ".[dev]"

# 2. Set your LLM provider API key
export OPENAI_API_KEY="sk-..."

# 3. Run the agent (Python snippet)
python3 -c "
import asyncio
from ia_agent_fwk.llm.factory import create_provider
from examples.document_processor.agent import create_document_processor_agent

async def main():
    provider = create_provider('openai')
    agent = create_document_processor_agent(provider)

    with open('examples/document_processor/data/sample_contract.txt') as f:
        document = f.read()

    result = await agent.run(f'Analyze this document:\n\n{document}')
    print(result.output)

asyncio.run(main())
"
```

## Sample data

Three sample documents are included in `data/`:

- `sample_contract.txt` -- Service agreement between two companies
- `sample_invoice.txt` -- Invoice with line items and payment terms
- `sample_report.txt` -- Quarterly business report with financial data

## Configuration

See `config.yaml` for the full YAML configuration.  You can override any
setting with environment variables using the `IAFWK_` prefix.

## Running tests

```bash
python3 -m pytest tests/unit/test_examples/test_document_processor.py -v
```
