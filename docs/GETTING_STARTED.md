# Getting Started

This guide walks you through installing ia-agent-fwk, creating your first agent, adding tools, and running it via the REST API.

## Prerequisites

- **Python 3.11+**
- **Docker and Docker Compose** (for infrastructure services)
- An API key for at least one LLM provider (OpenAI, Anthropic, or a local Ollama instance)

## Installation

### From Source (Development)

```bash
git clone https://github.com/hperezrodal/ia-agent-fwk.git
cd ia-agent-fwk
pip install -e ".[dev]"
```

### With Optional Extras

```bash
# RAG support (PDF, HTML parsing)
pip install -e ".[rag]"

# Slack integration
pip install -e ".[slack]"

# Email integration
pip install -e ".[email]"

# Weaviate vector store
pip install -e ".[weaviate]"

# HuggingFace local models
pip install -e ".[huggingface]"

# All extras
pip install -e ".[dev,rag,slack,email,weaviate,huggingface]"
```

## Start Infrastructure Services

The framework requires PostgreSQL (with pgvector), Redis, and Qdrant for full functionality. Start them with Docker Compose:

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 16 + pgvector** on `127.0.0.1:5434`
- **Redis 7** on `127.0.0.1:6380`
- **Qdrant v1.12.6** on `127.0.0.1:6333`

## Set Up Your LLM Provider

Set your API key as an environment variable:

```bash
# For OpenAI
export IAFWK_LLM__PROVIDERS__OPENAI__API_KEY="sk-..."

# For Anthropic
export IAFWK_LLM__PROVIDERS__ANTHROPIC__API_KEY="sk-ant-..."

# For Ollama (no API key needed, just ensure Ollama is running)
# Ollama runs at http://localhost:11434 by default
```

## Create Your First Agent

### Step 1: Define an Agent Class

Every agent extends the `Agent` ABC and provides an `agent_type` property:

```python
from ia_agent_fwk.agents import Agent, AgentConfig, AgentRegistry

class MyAgent(Agent):
    """A simple agent that responds to user input."""

    @property
    def agent_type(self) -> str:
        return "my_agent"
```

### Step 2: Register the Agent Type

Before the framework can instantiate your agent, register it in the `AgentRegistry`:

```python
AgentRegistry.register("my_agent", MyAgent)
```

### Step 3: Create a Provider and Run

```python
import asyncio
from ia_agent_fwk.config import load_config
from ia_agent_fwk.llm import LLMProviderFactory

async def main():
    # Load configuration (reads config/default.yaml + env vars)
    settings = load_config()

    # Create an LLM provider
    provider = LLMProviderFactory.create(settings.llm, provider_name="openai")

    # Create agent config
    config = AgentConfig(
        name="my-first-agent",
        agent_type="my_agent",
        system_prompt="You are a helpful assistant. Be concise.",
        max_iterations=5,
        execution_timeout=60,
    )

    # Create and run the agent
    agent = MyAgent(config=config, provider=provider)
    result = await agent.run("What is the capital of France?")

    print(f"Output: {result.output}")
    print(f"Iterations: {result.iterations}")
    print(f"Tokens used: {result.usage.total_tokens}")
    print(f"Duration: {result.duration_ms:.0f}ms")

asyncio.run(main())
```

## Add Tools to Your Agent

Tools give your agent the ability to perform actions beyond text generation.

### Step 1: Create a Tool

```python
from pydantic import BaseModel
from ia_agent_fwk.tools import Tool, ToolContext

class WeatherInput(BaseModel):
    city: str

class WeatherOutput(BaseModel):
    temperature: float
    condition: str

class WeatherTool(Tool):
    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Get the current weather for a city."

    @property
    def input_schema(self) -> type[BaseModel]:
        return WeatherInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return WeatherOutput

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        # In a real tool, you would call a weather API here
        city = validated_input.city  # type: ignore[attr-defined]
        return WeatherOutput(temperature=22.5, condition="Sunny")
```

### Step 2: Register the Tool and Create an Executor

```python
from ia_agent_fwk.tools import ToolRegistry, DefaultToolExecutor, ToolPermissionManager
from ia_agent_fwk.tools.permissions import PermissionMode

# Create a registry and register your tool
registry = ToolRegistry()
registry.register(WeatherTool())

# Create permission manager and executor
permission_manager = ToolPermissionManager(default_mode=PermissionMode.allow_all)
executor = DefaultToolExecutor(
    registry=registry,
    permission_manager=permission_manager,
    agent_id="my-first-agent",
)
```

### Step 3: Pass the Executor to Your Agent

```python
agent = MyAgent(
    config=AgentConfig(
        name="my-first-agent",
        agent_type="my_agent",
        system_prompt="You are a helpful assistant with access to weather data.",
        tools=["get_weather"],
    ),
    provider=provider,
    tool_executor=executor,
)

result = await agent.run("What is the weather in Paris?")
```

## Run via the REST API

The framework includes a FastAPI-based REST API.

### Start the API Server

```bash
uvicorn ia_agent_fwk.api:create_app --factory --host 0.0.0.0 --port 8000
```

### Make Requests

```bash
# Health check
curl http://localhost:8000/health

# Run an agent (requires agent type to be registered)
curl -X POST http://localhost:8000/api/v1/agents/my_agent/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Hello!"}'
```

The API response includes the agent output, token usage, iteration count, and conversation ID for multi-turn conversations:

```json
{
  "conversation_id": "abc-123",
  "output": "Hello! How can I help you today?",
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 10,
    "total_tokens": 35
  },
  "iterations": 1,
  "duration_ms": 1250.5,
  "agent_type": "my_agent"
}
```

### Multi-Turn Conversations

Pass the `conversation_id` from the first response to continue the conversation:

```bash
curl -X POST http://localhost:8000/api/v1/agents/my_agent/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Tell me more.", "conversation_id": "abc-123"}'
```

## Next Steps

- [Creating Agents](CREATING_AGENTS.md) -- Agent lifecycle, hooks, and advanced patterns
- [Creating Tools](CREATING_TOOLS.md) -- Custom tools with input/output validation
- [Creating Plugins](CREATING_PLUGINS.md) -- Package tools as plugins
- [Configuration](CONFIGURATION.md) -- All configuration options
- [Architecture](ARCHITECTURE.md) -- Module overview and design patterns
- [Deployment](DEPLOYMENT.md) -- Production deployment guide
- [Integrations](INTEGRATIONS.md) -- Slack, Email, and WhatsApp setup
