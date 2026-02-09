# Customer Support Agent Example

A reference implementation showing how to build a customer support agent using the `ia_agent_fwk` framework.

## Overview

This example demonstrates:

- Extending the `Agent` ABC with a domain-specific agent (`CustomerSupportAgent`)
- Implementing four custom tools: `TicketLookupTool`, `FAQSearchTool`, `EscalationTool`, `ResponseDraftTool`
- Configuring tools, registry, and executor for the agent
- Using conversation memory for multi-turn support interactions

## Structure

```
examples/customer_support/
  agent.py         - CustomerSupportAgent class and factory function
  config.yaml      - Example YAML configuration
  data/
    tickets.json   - Sample support tickets (8 entries)
    faq.json       - FAQ knowledge base (15 Q&A pairs)
```

## Quick Start

1. Install the framework in development mode:

```bash
pip install -e ".[dev]"
```

2. Set your LLM provider API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

3. Use the agent programmatically:

```python
from examples.customer_support.agent import create_support_agent
from ia_agent_fwk.llm.providers.openai import OpenAIProvider
from ia_agent_fwk.config.settings import LLMProviderSettings

# Create provider
settings = LLMProviderSettings(model="gpt-4o", api_key="your-key")
provider = OpenAIProvider(settings=settings, provider_name="openai")

# Create agent
agent = create_support_agent(provider)

# Run a query
import asyncio
result = asyncio.run(agent.run("Can you look up ticket TKT-001?"))
print(result.output)
```

## Tools

| Tool | Description |
|------|-------------|
| `ticket_lookup` | Look up support tickets by ID |
| `faq_search` | Search FAQ knowledge base with keyword matching |
| `escalation` | Escalate a ticket to a human agent |
| `response_draft` | Draft a customer response from a template |

## Running Tests

```bash
python3 -m pytest tests/unit/test_examples/test_customer_support.py -v
```
