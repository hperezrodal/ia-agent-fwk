# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-10

### Added

#### LLM Provider Layer
- Abstract `LLMProvider` base class with Factory + Registry pattern for extensibility
- **OpenAI** provider with chat completion, streaming, and function calling support
- **Anthropic** provider with Claude model family support
- **Ollama** provider for local/self-hosted open-source models
- **vLLM** provider for high-throughput inference serving
- **HuggingFace** provider with Transformers integration (optional dependency)
- LLM streaming support with async iterator interface
- Circuit breaker pattern for resilient provider calls
- Automatic retry with configurable backoff strategies
- Token cost tracking and estimation via `tiktoken`

#### Agent Core
- Abstract `Agent` base class with perceive-reason-act-observe execution loop
- Configurable agent lifecycle with pause and resume capabilities
- Agent state management with immutable Pydantic v2 models (`ConfigDict(frozen=True)`)
- Agent error hierarchy (`AgentError`) for structured exception handling

#### Tool System
- Abstract `Tool` base class with permission-based execution model
- Tool Registry with lazy colon-delimited dotted-path imports
- Tool permission system for fine-grained access control
- 10 built-in tools:
  - `calculator` -- mathematical expression evaluation
  - `current_time` -- current date/time retrieval
  - `database_query` -- parameterized database queries
  - `document_tools` -- document processing utilities
  - `echo` -- input echo for testing and debugging
  - `file_reader` -- local file reading
  - `finance_tools` -- financial data operations
  - `http_request` -- HTTP client for external API calls
  - `support_tools` -- customer support utilities
  - `web_scraper` -- web page content extraction
- Tool executor with timeout and error handling

#### Plugin System
- Plugin base class and discovery mechanism
- Dynamic plugin loader with entry-point and directory-based discovery
- Plugin manager for lifecycle management (register, enable, disable)
- Plugin models for metadata and configuration

#### Memory System
- Abstract `MemoryBackend` base class with factory pattern
- **InMemory** backend for development and testing
- **PgVector** backend for PostgreSQL with pgvector extension
- **Qdrant** backend for dedicated vector search
- **Weaviate** backend for hybrid vector/keyword search (optional dependency)
- **Conversation** memory for multi-turn dialogue history
- **Structured** memory for typed key-value storage
- Embedding abstraction layer for pluggable embedding models
- Memory factory with configuration-driven backend selection

#### RAG Pipeline
- Modular RAG pipeline with loader-chunker-retriever architecture
- **Document Loaders**: text, PDF, HTML, Markdown with loader registry
- **Chunkers**: fixed-size and recursive character text splitting
- **Retrievers**: vector similarity search, Weaviate hybrid retriever, contextual retrieval
- RAG factory for configuration-driven pipeline assembly

#### Multi-Agent Orchestration
- Abstract orchestration base class with builder pattern
- **Sequential** orchestrator for pipeline-style agent chains
- **Parallel** orchestrator for concurrent agent execution
- **Supervisor** orchestrator for manager-worker delegation patterns
- **Conditional** orchestrator for branching logic based on runtime conditions
- Agent-as-tool adapter for composing agents within orchestration flows

#### Execution Layer
- **Celery** integration for distributed task execution
- Task manager for submitting, tracking, and cancelling async jobs
- **Scheduler** for cron-based and interval-based recurring tasks
- **Triggers** for event-driven task activation
- Execution models for task state and result tracking

#### Streaming
- **Server-Sent Events (SSE)** endpoint for real-time token streaming
- **WebSocket** endpoint for bidirectional streaming communication
- Streaming models for structured event payloads

#### Integrations
- Integration base class with channel router for multi-channel dispatch
- **Slack** integration with Slack SDK (optional dependency)
- **Email** integration with async SMTP via `aiosmtplib` (optional dependency)
- **WhatsApp** integration for messaging workflows

#### Observability
- **OpenTelemetry** tracing with automatic span creation
- Structured **JSON logging** with configurable log levels
- **Metrics** collection for request counts, latencies, and error rates
- FastAPI observability **middleware** for automatic request instrumentation
- **Prompt logging** for LLM input/output audit trails

#### Security
- **Rate limiter** with configurable per-client and global limits
- **Audit logging** for security-sensitive operations
- Input **sanitizer** for prompt injection mitigation

#### REST API
- **FastAPI** application with versioned REST API endpoints
- API key authentication middleware
- Health check endpoints for liveness and readiness probes
- CORS configuration for cross-origin requests

#### Database
- Database module with async PostgreSQL support via `asyncpg`
- Connection pool management and query execution utilities

#### Configuration
- YAML-based configuration with `IAFWK_` environment variable prefix
- Pydantic Settings v2 integration for validated, typed configuration
- Hierarchical config merging (defaults, YAML file, environment variables)

#### Example Agents
- **Customer Support** agent with support tools and conversation memory
- **Document Processor** agent with RAG pipeline and document tools
- **Finance** agent with financial tools and data analysis capabilities

#### Infrastructure
- Docker Compose setup with PostgreSQL 16 + pgvector, Redis 7, and Qdrant v1.12.6
- Production Docker image with multi-stage build
- Production Docker Compose with full stack deployment
- Celery worker and beat scheduler configuration
- GitHub Actions CI pipeline with ruff linting, mypy strict type checking, and pytest
- Makefile with developer convenience commands
- 1339+ unit tests with 90%+ code coverage

### Changed

- Nothing. This is the initial public release.

### Fixed

- Nothing. This is the initial public release.

[1.0.0]: https://github.com/hperezrodal/ia-agent-fwk/releases/tag/v1.0.0
