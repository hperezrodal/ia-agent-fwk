# Architecture

ia-agent-fwk is a modular monolith with 15 modules, each responsible for a distinct concern. The modules communicate through well-defined Python interfaces (ABCs) and use the Factory+Registry pattern for extensibility.

## Module Overview

| Module | Responsibility |
|---|---|
| `config` | YAML + env var configuration with Pydantic Settings v2 validation |
| `api` | FastAPI REST API: routes, middleware, authentication, error handling |
| `agents` | Agent ABC, perceive-reason-act-observe loop, state machine, factory, registry |
| `llm` | LLM provider ABC, 5 provider implementations, circuit breaker, cost estimator |
| `tools` | Tool ABC, registry, permissions, executor, 7 built-in tools |
| `memory` | Memory backend ABC, 6 backends (InMemory, Conversation, PgVector, Qdrant, Structured, Weaviate) |
| `rag` | RAG pipeline: document loaders, chunkers, retrievers, context assembler |
| `orchestration` | Multi-agent workflows: sequential, parallel, supervisor, conditional |
| `execution` | Celery-based background jobs, cron schedules, event triggers |
| `streaming` | Real-time output via SSE and WebSocket |
| `plugins` | Plugin ABC, manifest, discovery (entry points / directory), loader, manager |
| `integrations` | Channel integrations (Slack, Email, WhatsApp), channel router |
| `observability` | OpenTelemetry tracing, structured JSON logging, Prometheus metrics, prompt logging |
| `security` | Rate limiting, audit logging, input sanitization, tool sandboxing, cost controls |
| `db` | Database connection utilities |

## Data Flow

```
                            ┌──────────────────┐
                            │   REST API (api)  │
                            │  FastAPI + Auth   │
                            └────────┬─────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  ▼                  ▼                  ▼
           ┌────────────┐   ┌──────────────┐   ┌──────────────┐
           │   Agents    │   │  Streaming   │   │  Execution   │
           │ (reasoning  │   │ (SSE / WS)   │   │ (Celery jobs │
           │   loop)     │   │              │   │  schedules)  │
           └──────┬──────┘   └──────────────┘   └──────────────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
  ┌──────────┐ ┌───────┐ ┌────────┐
  │   LLM    │ │ Tools │ │ Memory │
  │ Provider │ │       │ │Backend │
  └──────────┘ └───────┘ └────────┘
        │                     │
        ▼                     ▼
  ┌──────────┐         ┌──────────┐
  │ OpenAI   │         │ PgVector │
  │ Anthropic│         │ Qdrant   │
  │ Ollama   │         │ Weaviate │
  │ vLLM     │         │ InMemory │
  │ HuggingF.│         │ ...      │
  └──────────┘         └──────────┘
```

### Request Lifecycle

1. A client sends a POST request to `/api/v1/agents/{agent_type}/run`
2. The API layer authenticates the request (API key or JWT) and applies rate limiting
3. The `AgentFactory` looks up the agent class in `AgentRegistry` and creates an `LLMProvider` via `LLMProviderFactory`
4. The agent's `run()` method starts the reasoning loop:
   - Creates an `AgentContext` with the system prompt and user message
   - Runs the perceive-reason-act-observe loop (delegated to `ReasoningLoop`)
   - On each iteration, calls the LLM provider and optionally executes tools
   - Continues until the LLM signals completion, max iterations, or timeout
5. The result is returned with output text, token usage, iteration count, and duration

## Key Abstractions

### LLMProvider (ABC)

All LLM interactions go through the `LLMProvider` interface:

```python
class LLMProvider(ABC):
    def __init__(self, settings: LLMProviderSettings, provider_name: str) -> None: ...

    async def complete(self, prompt: str, **kwargs) -> CompletionResponse: ...
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse: ...
    async def stream(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]: ...
    def count_tokens(self, text: str, model: str | None = None) -> int: ...
    async def health_check(self) -> HealthStatus: ...
    def format_tools(self, schemas: list[dict]) -> list[dict]: ...
    async def close(self) -> None: ...
```

Five built-in providers: `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`, `VLLMProvider`, `HuggingFaceProvider`.

### Agent (ABC)

Agents implement a `perceive-reason-act-observe` loop:

```python
class Agent(ABC):
    def __init__(self, config: AgentConfig, provider: LLMProvider,
                 tool_executor: ToolExecutor | None = None) -> None: ...

    @property
    @abstractmethod
    def agent_type(self) -> str: ...

    async def run(self, input_text: str,
                  conversation_history: list[Message] | None = None) -> AgentResult: ...

    # Lifecycle hooks (override for custom behavior)
    async def on_start(self) -> None: ...
    async def on_complete(self, result: AgentResult) -> None: ...
    async def on_error(self, error: Exception) -> None: ...
    async def on_timeout(self) -> None: ...

    # Control operations
    async def stop(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self, input_text: str) -> None: ...
```

Agent states: `IDLE -> RUNNING -> COMPLETED | FAILED | WAITING_FOR_INPUT`.

### Tool (ABC)

Tools extend agent capabilities:

```python
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]: ...

    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]: ...

    @abstractmethod
    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel: ...
```

### MemoryBackend (ABC)

Unified interface for all storage backends:

```python
class MemoryBackend(ABC):
    @property
    @abstractmethod
    def backend_type(self) -> str: ...

    async def store(self, key: str, value: Any, metadata: dict | None = None) -> None: ...
    async def retrieve(self, key: str) -> Any | None: ...
    async def search(self, query: str, top_k: int = 5) -> list[MemoryResult]: ...
    async def delete(self, key: str) -> bool: ...
    async def clear(self) -> None: ...
    async def health_check(self) -> bool: ...
    async def close(self) -> None: ...
```

## Factory + Registry Pattern

The framework uses a consistent pattern across all extensible components:

1. **Registry** -- A `ClassVar[dict]` maps logical names to classes (or lazy dotted-path strings)
2. **Factory** -- A `create()` class method resolves the name, imports the class if needed, and instantiates it with configuration
3. **Lazy Imports** -- Built-in implementations are registered as colon-delimited paths (e.g., `"ia_agent_fwk.llm.providers.openai:OpenAIProvider"`) and only imported when first used

This pattern appears in:
- `LLMProviderFactory` -- Maps provider names (`"openai"`, `"anthropic"`, ...) to provider classes
- `AgentFactory` / `AgentRegistry` -- Maps agent type names to agent classes
- `MemoryFactory` -- Maps backend names (`"in_memory"`, `"pgvector"`, ...) to backend classes
- `ChunkerFactory` -- Maps chunking strategy names to chunker classes
- `EmbeddingFactory` -- Maps embedding provider names to provider classes

### Registering a Custom Provider

```python
from ia_agent_fwk.llm import LLMProviderFactory

# Register with a lazy path (imported only when used)
LLMProviderFactory.register("my_provider", "my_package.providers:MyProvider")

# Or register with a class reference
LLMProviderFactory.register("my_provider", MyProvider)
```

## Config System Overview

The configuration system has four layers with clear precedence:

```
Env vars (IAFWK_*)          -- highest priority
  └─ config/{env}.yaml      -- environment-specific overrides
      └─ config/default.yaml -- base defaults
          └─ Pydantic defaults -- lowest priority
```

The root `AppSettings` model (a Pydantic Settings v2 `BaseSettings` subclass) contains 16 nested sections:

```
AppSettings
├── app (name, version, environment, debug, log_level)
├── server (host, port, workers, reload, cors)
├── auth (api_key, jwt)
├── database (url, pool_size, echo)
├── redis (url, max_connections)
├── llm (default_provider, providers: {name: LLMProviderSettings})
├── memory (default_backend, backends, embedding)
├── rag (chunking, retrieval, storage_backend)
├── execution (mode, celery, scheduler)
├── streaming (sse, websocket)
├── plugins (enabled, discovery_method, plugin_dirs)
├── integrations (slack, email, whatsapp)
├── agents (default_agent, agents: {name: AgentConfigSettings})
├── observability (logging, tracing, metrics, prompt_logging)
├── security (rate_limiting, tool_sandboxing, cost_controls)
├── tools (default_timeout, default_permission_mode, builtin_tools_enabled)
└── orchestration (max_delegation_depth, default_workflow_timeout, max_parallel_agents)
```

See [CONFIGURATION.md](CONFIGURATION.md) for the complete reference.

## Orchestration

The orchestration module provides four workflow types for multi-agent coordination:

| Workflow | Description |
|---|---|
| `SequentialWorkflow` | Executes agents one after another, passing output as input to the next |
| `ParallelWorkflow` | Executes agents concurrently and collects all results |
| `SupervisorAgent` | A meta-agent that delegates tasks to sub-agents based on LLM reasoning |
| `ConditionalWorkflow` | Routes to different agents based on conditions evaluated at runtime |

Workflows are defined declaratively using `WorkflowDefinition` and `WorkflowStep` models, and can be built with the `build_workflow()` helper.

## Infrastructure

### Development

```
docker compose up -d
├── PostgreSQL 16 + pgvector  →  127.0.0.1:5434
├── Redis 7                   →  127.0.0.1:6380
└── Qdrant v1.12.6            →  127.0.0.1:6333
```

### Production

```
docker compose -f docker/docker-compose.prod.yml up -d
├── PostgreSQL 16 + pgvector  (internal)
├── Redis 7 with persistence  (internal)
├── Qdrant v1.12.6            (internal)
├── API Server                →  :8000
├── Celery Worker(s)          (internal)
├── Celery Beat Scheduler     (internal)
├── [optional] Ollama         (--profile self-hosted)
└── [optional] Flower         (--profile monitoring)
```
