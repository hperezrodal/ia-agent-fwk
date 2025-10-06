"""Pydantic Settings models for ia-agent-fwk configuration.

All framework configuration is represented as typed, validated Pydantic models.
The root ``AppSettings`` class supports loading from YAML files and environment
variables with the ``IAFWK_`` prefix and ``__`` nesting separator.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ia_agent_fwk.tools.config import ToolsConfig

# ==============================================================================
# Sub-models (plain BaseModel -- not directly loaded from env vars)
# ==============================================================================


class CorsSettings(BaseModel):
    """CORS configuration for the API server."""

    allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])


class ServerSettings(BaseModel):
    """HTTP server configuration."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors: CorsSettings = Field(default_factory=CorsSettings)


class ApiKeyAuthSettings(BaseModel):
    """API key authentication settings."""

    enabled: bool = True
    header_name: str = "X-API-Key"


class JwtSettings(BaseModel):
    """JWT authentication settings."""

    enabled: bool = False
    secret_key: str = ""
    algorithm: str = "HS256"
    expiration_minutes: int = 60


class AuthSettings(BaseModel):
    """Authentication configuration."""

    enabled: bool = True
    api_key: ApiKeyAuthSettings = Field(default_factory=ApiKeyAuthSettings)
    jwt: JwtSettings = Field(default_factory=JwtSettings)


class DatabaseSettings(BaseModel):
    """PostgreSQL database configuration."""

    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5434/ia_agent_fwk"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


class RedisSettings(BaseModel):
    """Redis configuration."""

    url: str = "redis://localhost:6380/0"
    max_connections: int = 20


class RetrySettings(BaseModel):
    """Retry policy for LLM provider calls."""

    max_attempts: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0


class CircuitBreakerSettings(BaseModel):
    """Circuit breaker configuration for an LLM provider."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 30.0


class LLMProviderSettings(BaseModel):
    """Configuration for a single LLM provider."""

    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    default_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    retry: RetrySettings = Field(default_factory=RetrySettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)


class LLMSettings(BaseModel):
    """LLM provider configuration."""

    default_provider: str = "openai"
    providers: dict[str, LLMProviderSettings] = Field(default_factory=dict)


class InMemoryBackendSettings(BaseModel):
    """In-memory backend settings."""

    max_items: int = 1000


class ConversationBackendSettings(BaseModel):
    """Conversation history backend settings."""

    database_url: str = ""
    max_history: int = 100


class VectorBackendSettings(BaseModel):
    """Vector store backend settings (pgvector / qdrant)."""

    url: str = ""
    database_url: str = ""
    collection_name: str = "agent_memory"
    embedding_dimensions: int = 1536


class StructuredBackendSettings(BaseModel):
    """Structured memory backend settings."""

    database_url: str = ""
    table_name: str = "memory_kv"
    default_ttl_seconds: int = 0


class EmbeddingSettings(BaseModel):
    """Embedding provider configuration for memory backends."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = "http://localhost:11434"


class MemoryBackendsSettings(BaseModel):
    """Memory backend configurations."""

    in_memory: InMemoryBackendSettings = Field(default_factory=InMemoryBackendSettings)
    conversation: ConversationBackendSettings = Field(default_factory=ConversationBackendSettings)
    pgvector: VectorBackendSettings = Field(default_factory=VectorBackendSettings)
    qdrant: VectorBackendSettings = Field(default_factory=lambda: VectorBackendSettings(url="http://localhost:6333"))
    structured: StructuredBackendSettings = Field(default_factory=StructuredBackendSettings)
    weaviate: VectorBackendSettings = Field(
        default_factory=lambda: VectorBackendSettings(url="http://localhost:8080", collection_name="AgentMemory"),
    )


class MemorySettings(BaseModel):
    """Memory system configuration."""

    default_backend: str = "in_memory"
    backends: MemoryBackendsSettings = Field(default_factory=MemoryBackendsSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)


class ChunkingSettings(BaseModel):
    """RAG chunking strategy settings."""

    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class RetrievalSettings(BaseModel):
    """RAG retrieval settings."""

    top_k: int = 5
    score_threshold: float = 0.7
    strategy: str = "mmr"


class RAGSettings(BaseModel):
    """RAG pipeline configuration."""

    default_embedding_provider: str = "openai"
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    storage_backend: str = "pgvector"


class CelerySettings(BaseModel):
    """Celery worker configuration."""

    broker_url: str = "redis://localhost:6380/0"
    result_backend: str = "redis://localhost:6380/0"
    task_timeout: int = 300
    max_retries: int = 3
    worker_concurrency: int = 4
    result_expires: int = 86400
    worker_prefetch_multiplier: int = 1
    task_serializer: str = "json"


class SchedulerSettings(BaseModel):
    """Celery Beat scheduler configuration."""

    enabled: bool = False
    timezone: str = "UTC"


class ExecutionSettings(BaseModel):
    """Task execution configuration."""

    mode: str = "sync"
    worker_shutdown_timeout: int = 300
    celery: CelerySettings = Field(default_factory=CelerySettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)


class SSESettings(BaseModel):
    """Server-Sent Events configuration."""

    enabled: bool = True
    heartbeat_interval: int = 15


class WebSocketSettings(BaseModel):
    """WebSocket configuration."""

    enabled: bool = True
    ping_interval: int = 30
    max_connections: int = 100


class StreamingSettings(BaseModel):
    """Streaming configuration."""

    sse: SSESettings = Field(default_factory=SSESettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)


class PluginConfigSettings(BaseModel):
    """Per-plugin configuration within the main settings."""

    name: str
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)


class PluginsSettings(BaseModel):
    """Plugin system configuration."""

    enabled: bool = True
    discovery_method: str = "entry_points"
    plugin_dir: str = "./plugins"
    plugin_dirs: list[str] = Field(default_factory=list)
    auto_load: bool = True
    plugins: list[PluginConfigSettings] = Field(default_factory=list)


class SlackIntegrationSettings(BaseModel):
    """Slack integration configuration."""

    enabled: bool = False
    bot_token: str = ""
    signing_secret: str = ""
    default_agent: str = ""


class SmtpSettings(BaseModel):
    """SMTP email sending configuration."""

    host: str = "smtp.gmail.com"
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True


class ImapSettings(BaseModel):
    """IMAP email receiving configuration."""

    host: str = "imap.gmail.com"
    port: int = 993
    username: str = ""
    password: str = ""
    use_ssl: bool = True


class EmailIntegrationSettings(BaseModel):
    """Email integration configuration."""

    enabled: bool = False
    smtp: SmtpSettings = Field(default_factory=SmtpSettings)
    imap: ImapSettings = Field(default_factory=ImapSettings)
    default_agent: str = ""


class WhatsAppIntegrationSettings(BaseModel):
    """WhatsApp integration configuration."""

    enabled: bool = False
    api_url: str = "https://graph.facebook.com/v18.0"
    phone_number_id: str = ""
    access_token: str = ""
    verify_token: str = ""
    default_agent: str = ""


class IntegrationsSettings(BaseModel):
    """External integrations configuration."""

    slack: SlackIntegrationSettings = Field(default_factory=SlackIntegrationSettings)
    email: EmailIntegrationSettings = Field(default_factory=EmailIntegrationSettings)
    whatsapp: WhatsAppIntegrationSettings = Field(default_factory=WhatsAppIntegrationSettings)


class LoggingSettings(BaseModel):
    """Logging configuration."""

    format: str = "json"
    level: str = "INFO"
    include_timestamp: bool = True
    include_correlation_id: bool = True
    redact_pii: bool = True
    redact_patterns: list[str] = Field(default_factory=list)


class TracingSettings(BaseModel):
    """OpenTelemetry tracing configuration."""

    enabled: bool = False
    exporter: str = "otlp"
    endpoint: str = "http://localhost:4317"
    service_name: str = "ia-agent-fwk"
    sample_rate: float = 1.0


class MetricsSettings(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = False
    endpoint: str = "/metrics"
    include_agent_metrics: bool = True
    include_tool_metrics: bool = True
    include_llm_metrics: bool = True


class PromptLoggingSettings(BaseModel):
    """LLM prompt/response logging configuration."""

    enabled: bool = True
    log_inputs: bool = True
    log_outputs: bool = True
    log_tokens: bool = True
    log_latency: bool = True
    redact_sensitive: bool = True


class ObservabilitySettings(BaseModel):
    """Observability configuration."""

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    prompt_logging: PromptLoggingSettings = Field(default_factory=PromptLoggingSettings)


class RateLimitingSettings(BaseModel):
    """Rate limiting configuration."""

    enabled: bool = True
    default_rate: str = "60/minute"
    per_endpoint: dict[str, str] = Field(default_factory=dict)


class ToolSandboxingSettings(BaseModel):
    """Tool sandboxing configuration."""

    enabled: bool = True
    max_execution_time: int = 30
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)


class CostControlSettings(BaseModel):
    """LLM cost control configuration."""

    enabled: bool = True
    max_tokens_per_execution: int = 50000
    max_tokens_per_day: int = 1000000
    alert_threshold_percentage: int = 80


class SecuritySettings(BaseModel):
    """Security configuration."""

    rate_limiting: RateLimitingSettings = Field(default_factory=RateLimitingSettings)
    tool_sandboxing: ToolSandboxingSettings = Field(default_factory=ToolSandboxingSettings)
    cost_controls: CostControlSettings = Field(default_factory=CostControlSettings)


class OrchestrationSettings(BaseModel):
    """Orchestration configuration."""

    max_delegation_depth: int = 3
    default_workflow_timeout: int = 600
    max_parallel_agents: int = 10


class AgentConfigSettings(BaseModel):
    """Configuration for a single agent (used within AgentSettings).

    When ``name`` or ``agent_type`` are provided they must be non-empty,
    matching the validation rules of ``AgentConfig`` from
    ``agents/config.py``.
    """

    name: str = ""
    agent_type: str = ""
    system_prompt: str = ""
    provider_name: str = "openai"
    model: str | None = None
    max_iterations: int = 10
    execution_timeout: int = 300
    max_tokens_per_response: int = 4096
    tools: list[str] = Field(default_factory=list)
    memory: dict[str, Any] | None = None
    context_window: int | None = None

    @model_validator(mode="after")
    def _require_name_and_type(self) -> AgentConfigSettings:
        """Ensure ``name`` and ``agent_type`` are both set or both empty."""
        has_name = bool(self.name)
        has_type = bool(self.agent_type)
        if has_name != has_type:
            msg = "Both 'name' and 'agent_type' must be provided together"
            raise ValueError(msg)
        return self


class AgentSettings(BaseModel):
    """Agent system configuration."""

    default_agent: str = ""
    agents: dict[str, AgentConfigSettings] = Field(default_factory=dict)


# ==============================================================================
# Root settings (loaded from env vars + YAML)
# ==============================================================================


class AppCoreSettings(BaseModel):
    """Core application identity and runtime settings.

    These are nested under the ``app:`` key in YAML configuration files
    and are accessed as ``settings.app.name``, ``settings.app.debug``, etc.
    Environment variables use the ``IAFWK_APP__`` prefix, e.g.
    ``IAFWK_APP__DEBUG=true``.
    """

    name: str = "ia-agent-fwk"
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"


class AppSettings(BaseSettings):
    """Root application settings.

    Loads configuration from YAML files and environment variables.
    Environment variables use the ``IAFWK_`` prefix with ``__`` as the
    nesting separator. For example: ``IAFWK_DATABASE__URL``,
    ``IAFWK_APP__DEBUG``.
    """

    model_config = SettingsConfigDict(
        env_prefix="IAFWK_",
        env_nested_delimiter="__",
        frozen=True,
        extra="ignore",
    )

    # -- App-level settings (nested under "app" in YAML) --
    app: AppCoreSettings = Field(default_factory=AppCoreSettings)

    # -- Nested settings sections --
    server: ServerSettings = Field(default_factory=ServerSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)
    plugins: PluginsSettings = Field(default_factory=PluginsSettings)
    integrations: IntegrationsSettings = Field(default_factory=IntegrationsSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    orchestration: OrchestrationSettings = Field(default_factory=OrchestrationSettings)
