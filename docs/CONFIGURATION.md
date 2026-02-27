# Configuration Reference

ia-agent-fwk uses a four-layer configuration system with clear precedence. All settings are validated at startup by Pydantic Settings v2.

## Precedence (Highest to Lowest)

1. **Environment variables** -- `IAFWK_` prefix with `__` nesting separator
2. **Environment-specific YAML** -- `config/{environment}.yaml`
3. **Default YAML** -- `config/default.yaml`
4. **Pydantic defaults** -- Hardcoded in the settings models

## Loading Configuration

```python
from ia_agent_fwk.config import load_config

# Auto-detect environment from IAFWK_APP__ENVIRONMENT (default: "development")
settings = load_config()

# Explicit environment
settings = load_config("production")

# Custom config directory
settings = load_config(config_dir="/etc/ia-agent-fwk")
```

## Environment Variable Format

All settings can be overridden via environment variables using the `IAFWK_` prefix and `__` (double underscore) as the nesting separator:

```bash
# app.debug -> IAFWK_APP__DEBUG
export IAFWK_APP__DEBUG=true

# llm.providers.openai.api_key -> IAFWK_LLM__PROVIDERS__OPENAI__API_KEY
export IAFWK_LLM__PROVIDERS__OPENAI__API_KEY="sk-..."

# database.url -> IAFWK_DATABASE__URL
export IAFWK_DATABASE__URL="postgresql+asyncpg://user:pass@host:5432/db"
```

## Environment Profiles

Four profiles are supported: `development`, `testing`, `staging`, `production`.

Set the active profile:
```bash
export IAFWK_APP__ENVIRONMENT=production
```

Each profile can have a corresponding YAML file (e.g., `config/production.yaml`) that overrides `config/default.yaml`.

### Production Validations

In `production` environment, the following semantic rules are enforced:
- `app.debug` must be `false`
- `auth.enabled` must be `true`
- If JWT is enabled, `auth.jwt.secret_key` must be set
- At least one LLM provider must have an `api_key` or `base_url`

## Configuration Sections

### app

Core application identity and runtime settings.

```yaml
app:
  name: "ia-agent-fwk"         # Application name
  version: "0.1.0"             # Application version
  environment: "development"   # development | testing | staging | production
  debug: false                 # Enable debug mode
  log_level: "INFO"            # DEBUG | INFO | WARNING | ERROR | CRITICAL
```

### server

HTTP server (Uvicorn) configuration.

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1                   # Uvicorn workers (production: CPU * 2 + 1)
  reload: false                # Hot-reload (development only)
  cors:
    allow_origins: ["*"]
    allow_methods: ["*"]
    allow_headers: ["*"]
```

### auth

Authentication configuration.

```yaml
auth:
  enabled: true
  api_key:
    enabled: true
    header_name: "X-API-Key"   # Header name for API key auth
  jwt:
    enabled: false
    secret_key: ""             # Set via IAFWK_AUTH__JWT__SECRET_KEY
    algorithm: "HS256"
    expiration_minutes: 60
```

### database

PostgreSQL connection settings.

```yaml
database:
  url: ""                      # Set via IAFWK_DATABASE__URL
  pool_size: 10
  max_overflow: 20
  echo: false                  # SQL echo logging
```

### redis

Redis connection settings.

```yaml
redis:
  url: "redis://localhost:6380/0"
  max_connections: 20
```

### llm

LLM provider configuration. Each provider is configured under `providers` with its own settings.

```yaml
llm:
  default_provider: "openai"   # Default provider for new agents
  providers:
    openai:
      api_key: ""              # Set via IAFWK_LLM__PROVIDERS__OPENAI__API_KEY
      default_model: "gpt-4o"
      temperature: 0.7
      max_tokens: 4096
      timeout: 60               # Request timeout in seconds
      retry:
        max_attempts: 3
        backoff_base: 2.0       # Exponential backoff base
        backoff_max: 60.0       # Maximum backoff delay
      circuit_breaker:
        enabled: true
        failure_threshold: 5    # Failures before circuit opens
        recovery_timeout: 30.0  # Seconds before retry after circuit opens

    anthropic:
      api_key: ""
      default_model: "claude-sonnet-4-20250514"
      temperature: 0.7
      max_tokens: 4096
      timeout: 60

    ollama:
      base_url: "http://localhost:11434"
      default_model: "llama3.1"
      temperature: 0.7
      max_tokens: 4096
      timeout: 120

    vllm:
      base_url: "http://localhost:8000/v1"
      default_model: "meta-llama/Llama-3.1-8B-Instruct"
      temperature: 0.7
      max_tokens: 4096
      timeout: 120

    huggingface:
      base_url: "cpu"          # Device: cpu | cuda | auto
      default_model: "gpt2"
      temperature: 0.7
      max_tokens: 256
      timeout: 300
```

### memory

Memory backend configuration.

```yaml
memory:
  default_backend: "in_memory"  # in_memory | conversation | pgvector | qdrant | structured | weaviate
  embedding:
    provider: "openai"
    model: "text-embedding-3-small"
    api_key: ""
  backends:
    in_memory:
      max_items: 1000
    conversation:
      database_url: ""
      max_history: 100
    pgvector:
      database_url: ""
      collection_name: "memory_vectors"
      embedding_dimensions: 1536
    qdrant:
      url: "http://localhost:6333"
      collection_name: "agent_memory"
      embedding_dimensions: 1536
    structured:
      database_url: ""
      table_name: "memory_kv"
      default_ttl_seconds: 0    # 0 = no expiration
    weaviate:
      url: "http://localhost:8080"
      collection_name: "AgentMemory"
      embedding_dimensions: 1536
```

### rag

RAG (Retrieval-Augmented Generation) pipeline settings.

```yaml
rag:
  default_embedding_provider: "openai"
  chunking:
    strategy: "recursive"       # fixed | recursive
    chunk_size: 1000            # Characters per chunk
    chunk_overlap: 200          # Overlap between chunks
  retrieval:
    top_k: 5                    # Number of results to retrieve
    score_threshold: 0.7        # Minimum similarity score
    strategy: "mmr"             # top_k | mmr (Maximal Marginal Relevance)
  storage_backend: "pgvector"   # pgvector | qdrant
```

### execution

Background task execution via Celery.

```yaml
execution:
  mode: "sync"                  # sync | async
  worker_shutdown_timeout: 300  # Seconds to drain in-flight tasks on SIGTERM
  celery:
    broker_url: "redis://localhost:6380/0"
    result_backend: "redis://localhost:6380/0"
    task_timeout: 300           # Max wall-clock time per task
    max_retries: 3
    worker_concurrency: 4       # Concurrent tasks per worker
    result_expires: 86400       # Result TTL in seconds (24h)
    worker_prefetch_multiplier: 1
    task_serializer: "json"
  scheduler:
    enabled: false
    timezone: "UTC"
```

### streaming

Real-time streaming configuration.

```yaml
streaming:
  sse:
    enabled: true
    heartbeat_interval: 15      # Seconds between SSE heartbeats
  websocket:
    enabled: true
    ping_interval: 30           # Seconds between WebSocket pings
    max_connections: 100
```

### plugins

Plugin system configuration.

```yaml
plugins:
  enabled: true
  discovery_method: "entry_points"  # entry_points | directory
  plugin_dir: "./plugins"           # Default plugin directory
  plugin_dirs: []                   # Additional directories to scan
  auto_load: true
  plugins: []                       # Per-plugin config:
    # - name: "my-plugin"
    #   enabled: true
    #   settings:
    #     key: "value"
```

### integrations

Channel integration settings.

```yaml
integrations:
  slack:
    enabled: false
    bot_token: ""               # Set via IAFWK_INTEGRATIONS__SLACK__BOT_TOKEN
    signing_secret: ""
    default_agent: ""           # Agent type to handle Slack messages

  email:
    enabled: false
    smtp:
      host: "smtp.gmail.com"
      port: 587
      username: ""
      password: ""
      use_tls: true
    imap:
      host: "imap.gmail.com"
      port: 993
      username: ""
      password: ""
      use_ssl: true
    default_agent: ""

  whatsapp:
    enabled: false
    api_url: "https://graph.facebook.com/v18.0"
    phone_number_id: ""
    access_token: ""
    verify_token: ""
    default_agent: ""
```

### agents

Agent definitions (optional; agents can also be created programmatically).

```yaml
agents:
  default_agent: ""             # Default agent type name
  agents:
    my_agent:
      name: "my-agent"
      agent_type: "my_agent"
      system_prompt: "You are helpful."
      provider_name: "openai"
      model: null               # Uses provider default_model if null
      max_iterations: 10
      execution_timeout: 300    # Seconds
      max_tokens_per_response: 4096
      tools: []                 # List of tool names
      context_window: null      # Defaults to 8192
```

### observability

Logging, tracing, and metrics configuration.

```yaml
observability:
  logging:
    format: "json"              # json | text
    level: "INFO"
    include_timestamp: true
    include_correlation_id: true
    redact_pii: true
    redact_patterns: []         # Additional regex patterns for PII redaction

  tracing:
    enabled: false
    exporter: "otlp"           # otlp | jaeger | console
    endpoint: "http://localhost:4317"
    service_name: "ia-agent-fwk"
    sample_rate: 1.0

  metrics:
    enabled: false
    endpoint: "/metrics"
    include_agent_metrics: true
    include_tool_metrics: true
    include_llm_metrics: true

  prompt_logging:
    enabled: true
    log_inputs: true
    log_outputs: true
    log_tokens: true
    log_latency: true
    redact_sensitive: true
```

### tools

Global tool system settings.

```yaml
tools:
  default_timeout: 30.0         # Seconds per tool execution
  default_permission_mode: "allow_all"  # allow_all | allow_list | deny_list | require_confirmation
  builtin_tools_enabled: true   # Auto-register built-in tools
  max_retries: 3                # Reserved for V2
```

### orchestration

Multi-agent orchestration settings.

```yaml
orchestration:
  max_delegation_depth: 3       # Maximum nesting depth for agent delegation
  default_workflow_timeout: 600 # Seconds
  max_parallel_agents: 10       # Maximum concurrent agents in parallel workflows
```

### security

Security controls.

```yaml
security:
  rate_limiting:
    enabled: true
    default_rate: "60/minute"   # Format: "{count}/{period}" (minute, hour, day)
    per_endpoint: {}            # Override rates for specific endpoints

  tool_sandboxing:
    enabled: true
    max_execution_time: 30      # Seconds
    allowed_domains: []         # Allowlist for HTTP tools
    blocked_domains: []         # Blocklist for HTTP tools

  cost_controls:
    enabled: true
    max_tokens_per_execution: 50000
    max_tokens_per_day: 1000000
    alert_threshold_percentage: 80
```

## Full Example: config/default.yaml

See the full default configuration file at [`config/default.yaml`](../config/default.yaml) in the repository.
