# ia-agent-fwk

![CI](https://github.com/hperezrodal/ia-agent-fwk/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

A modular, configuration-driven AI agent framework for building production-ready intelligent agents. ia-agent-fwk provides a complete toolkit for creating agents that reason, use tools, remember context, and integrate with external channels -- all orchestrated through a clean Python API and a declarative YAML configuration system.

## Key Features

- **Multi-Provider LLM Layer** -- OpenAI, Anthropic, Ollama, vLLM, and HuggingFace with retry, circuit breaker, and cost controls
- **Agent System** -- Perceive-reason-act-observe loop with pause/resume, lifecycle hooks, and configurable timeouts
- **Tool System** -- 7 built-in tools (calculator, HTTP, web scraper, file reader, database query, current time, echo) plus a permission system with allow/deny lists
- **Memory Backends** -- InMemory, Conversation, PgVector, Qdrant, Structured, and Weaviate backends with a unified ABC interface
- **RAG Pipeline** -- Document loaders (Text, Markdown, HTML, PDF), chunkers (fixed, recursive), and vector retrievers
- **Multi-Agent Orchestration** -- Sequential, Parallel, Supervisor, and Conditional workflows with delegation depth controls
- **Execution Layer** -- Celery-based background jobs, cron schedules, and event triggers
- **Streaming** -- Real-time output via Server-Sent Events (SSE) and WebSocket
- **Channel Integrations** -- Slack, Email, and WhatsApp with a unified channel router
- **Plugin System** -- Extend the framework with custom tools via entry points or directory discovery
- **Observability** -- OpenTelemetry tracing, structured JSON logging, Prometheus metrics, and prompt logging
- **Security** -- Rate limiting, audit logging, input sanitization, tool sandboxing, and cost controls
- **REST API** -- FastAPI-based with API key and JWT authentication, health checks, and conversation management

## Quick Start

### Installation

```bash
pip install ia-agent-fwk
```

For development with all extras:

```bash
pip install ia-agent-fwk[dev,rag,slack,email,weaviate,huggingface]
```

### Create Your First Agent

```python
import asyncio
from ia_agent_fwk.agents import Agent, AgentConfig, AgentRegistry
from ia_agent_fwk.llm import LLMProviderFactory
from ia_agent_fwk.config import load_config

# Define a custom agent
class GreeterAgent(Agent):
    @property
    def agent_type(self) -> str:
        return "greeter"

# Register the agent type
AgentRegistry.register("greeter", GreeterAgent)

# Load config and create an LLM provider
settings = load_config()
provider = LLMProviderFactory.create(settings.llm, provider_name="openai")

# Configure and run the agent
config = AgentConfig(
    name="my-greeter",
    agent_type="greeter",
    system_prompt="You are a friendly greeter. Say hello!",
)

agent = GreeterAgent(config=config, provider=provider)
result = asyncio.run(agent.run("Hi there!"))
print(result.output)
```

### Run via the REST API

```bash
# Start infrastructure services
docker compose up -d

# Set your API key
export IAFWK_LLM__PROVIDERS__OPENAI__API_KEY="sk-..."

# Start the API server
uvicorn ia_agent_fwk.api:create_app --factory --host 0.0.0.0 --port 8000

# Send a request
curl -X POST http://localhost:8000/api/v1/agents/greeter/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Hello!"}'
```

## Architecture Overview

```
ia-agent-fwk
├── config          Configuration system (YAML + env vars + Pydantic Settings v2)
├── api             FastAPI REST API with auth, health checks, and routes
├── agents          Agent ABC, state machine, reasoning loop, factory, registry
├── llm             LLM provider ABC, 5 providers, circuit breaker, cost estimator
├── tools           Tool ABC, registry, permissions, executor, 7 built-in tools
├── memory          Memory backend ABC, 6 backends, embedding providers
├── rag             RAG pipeline: loaders, chunkers, retrievers
├── orchestration   Multi-agent workflows (sequential, parallel, supervisor, conditional)
├── execution       Celery jobs, cron schedules, event triggers
├── streaming       SSE and WebSocket real-time streaming
├── plugins         Plugin ABC, manifest, discovery, loader, manager
├── integrations    Channel integrations (Slack, Email, WhatsApp) and router
├── observability   OTel tracing, JSON logging, Prometheus metrics, prompt logging
├── security        Rate limiting, audit logging, sanitization, cost controls
└── db              Database utilities
```

## Configuration

Configuration follows a four-layer precedence (highest to lowest):

1. Environment variables (`IAFWK_` prefix, `__` nesting separator)
2. Environment-specific YAML (`config/{environment}.yaml`)
3. Default YAML (`config/default.yaml`)
4. Pydantic Settings defaults

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for complete reference.

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) -- Step-by-step first agent tutorial
- [Architecture](docs/ARCHITECTURE.md) -- Detailed module overview and design patterns
- [Configuration](docs/CONFIGURATION.md) -- All configuration options with examples
- [Deployment](docs/DEPLOYMENT.md) -- Development setup and production deployment
- [Creating Agents](docs/CREATING_AGENTS.md) -- How to build custom agents
- [Creating Tools](docs/CREATING_TOOLS.md) -- How to build custom tools
- [Creating Plugins](docs/CREATING_PLUGINS.md) -- How to develop plugins
- [Integrations](docs/INTEGRATIONS.md) -- Slack, Email, and WhatsApp setup

## Examples

The `examples/` directory contains three complete agent implementations:

- **customer_support** -- Ticket lookup, FAQ search, escalation, and response drafting
- **document_processor** -- Document analysis and processing
- **finance** -- Financial data analysis

## Development

```bash
# Clone and install
git clone https://github.com/hperezrodal/ia-agent-fwk.git
cd ia-agent-fwk
pip install -e ".[dev]"

# Start infrastructure
docker compose up -d

# Run tests
pytest

# Lint and type check
ruff check src/ tests/
mypy src/
```

## License

MIT License. See [pyproject.toml](pyproject.toml) for details.
