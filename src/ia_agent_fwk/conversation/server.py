"""Generic conversational RAG server — ready to run, zero project code needed.

Reads all configuration from env vars + PostgreSQL. No hardcoded domain logic.

Usage:
    python -m ia_agent_fwk.conversation.server
    python -m ia_agent_fwk.conversation.server --port 8090

Env vars:
    DATABASE_URL        PostgreSQL connection (required for production)
    TENANT_ID           Tenant identifier (default: "default")
    LLM_PROVIDER        "openai" or "ollama" (default: "ollama")
    OPENAI_API_KEY      Required if LLM_PROVIDER=openai
    OPENAI_MODEL        Default: "gpt-4o-mini"
    OLLAMA_URL          Default: "http://localhost:11434"
    OLLAMA_MODEL        Default: "llama3.1:8b"
    EMBEDDING_PROVIDER  "ollama" or "openai" (default: "ollama")
    RERANK_MODEL        Cross-encoder model or "none" (default: "BAAI/bge-reranker-v2-m3")
    RERANK_DEVICE       "cpu" or "cuda" (default: "cpu")
    RATE_LIMIT          e.g. "30/minute" (default: "30/minute")
    REQUIRE_API_KEY     "true" or "false" (default: "false")
    API_KEYS            Comma-separated API keys
    INPUT_MAX_LENGTH    Max message length (default: 500)
    BUDGET_DAILY_USD    Daily spend limit (default: 0 = unlimited)
    BUDGET_MONTHLY_USD  Monthly spend limit (default: 0 = unlimited)
    TEMPO_ENDPOINT      OTel trace endpoint (empty = disabled)
    SERVICE_NAME        Service name for traces/metrics (default: "ia-agent")
    LOG_FORMAT          "json" or "text" (default: "json")
    LOG_LEVEL           Default: "INFO"
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI app. Everything from env vars + DB."""
    app = FastAPI(
        title=os.environ.get("APP_TITLE", "Conversational RAG Agent"),
        version="2.0.0",
    )

    # Observability (metrics, tracing, logging, /metrics endpoint)
    from ia_agent_fwk.conversation.metrics import setup_observability  # noqa: PLC0415

    setup_observability(app)

    # Security middleware
    from ia_agent_fwk.security import (  # noqa: PLC0415
        BudgetTracker,
        InputGuard,
        OutputGuard,
        SecurityConfig,
        SecurityMiddleware,
    )

    rate_limit = os.environ.get("RATE_LIMIT", "30/minute")
    require_api_key = os.environ.get("REQUIRE_API_KEY", "false").lower() == "true"
    api_keys = {k.strip(): f"client-{i}" for i, k in enumerate(os.environ.get("API_KEYS", "").split(",")) if k.strip()}

    app.add_middleware(
        SecurityMiddleware,
        config=SecurityConfig(
            rate_limit=rate_limit,
            require_api_key=require_api_key,
            api_keys=api_keys,
        ),
    )

    input_guard = InputGuard(max_length=int(os.environ.get("INPUT_MAX_LENGTH", "500")))
    output_guard = OutputGuard(redact_filenames=True, check_prompt_leak=True)
    _budget = BudgetTracker(
        daily_limit_usd=float(os.environ.get("BUDGET_DAILY_USD", "0")),
        monthly_limit_usd=float(os.environ.get("BUDGET_MONTHLY_USD", "0")),
    )

    # Startup: load config, build pipeline, create agent, mount endpoints
    @app.on_event("startup")
    async def _startup() -> None:
        database_url = os.environ.get("DATABASE_URL", "")
        tenant_id = os.environ.get("TENANT_ID", "default")
        embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "ollama")
        rerank_device = os.environ.get("RERANK_DEVICE", "cpu")

        # Suppress ONNX warning during import
        import os as _os  # noqa: PLC0415

        devnull = _os.open(_os.devnull, _os.O_WRONLY)
        old = _os.dup(2)
        _os.dup2(devnull, 2)
        from ia_agent_fwk.ingestion.providers import build_embedding_store  # noqa: PLC0415
        from ia_agent_fwk.ingestion.query_pipeline import QueryPipeline  # noqa: PLC0415

        _os.dup2(old, 2)
        _os.close(devnull)
        _os.close(old)

        from ia_agent_fwk.conversation import (  # noqa: PLC0415
            ConversationalRAGAgent,
            SessionManager,
            mount_chat_endpoints,
        )
        from ia_agent_fwk.conversation.llm_client import LLMClient  # noqa: PLC0415
        from ia_agent_fwk.ingestion.config_provider import RagConfigProvider  # noqa: PLC0415
        from ia_agent_fwk.ingestion.query_expansion import QueryExpander  # noqa: PLC0415

        # 1. Config from PostgreSQL
        config = RagConfigProvider(database_url=database_url, tenant_id=tenant_id)
        if database_url:
            await config.ensure_table()
            await config.load()
            logger.info("Config loaded: %d keys (tenant=%s)", len(config.get_all()), tenant_id)
        else:
            logger.warning("DATABASE_URL not set — using fallback defaults")

        # 2. LLM client
        llm = LLMClient.from_env()
        logger.info("LLM provider: %s (%s)", llm.provider, llm.model)

        # 3. RAG pipeline
        store = build_embedding_store(embedding_provider)
        rerank_model: str | None = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        if rerank_model and rerank_model.lower() == "none":
            rerank_model = None
        pipeline = QueryPipeline(
            store,
            rerank_model=rerank_model,
            rerank_device=rerank_device,
            query_expander=QueryExpander(
                synonyms=config.get("query_expansion.synonyms", {}),
            ).expand,
        )

        # 4. Session manager
        session = SessionManager(database_url=database_url, tenant_id=tenant_id)

        # 5. Agent
        agent = ConversationalRAGAgent(
            config=config,
            query_pipeline=pipeline,
            session_manager=session,
            llm_chat=llm.chat,
            llm_stream=llm.stream,
            llm_quick=llm.quick,
            input_guard=input_guard,
            output_guard=output_guard,
        )

        # 6. Mount endpoints
        mount_chat_endpoints(app, agent)
        logger.info("Server ready (tenant=%s)", tenant_id)

    return app


# Module-level app for uvicorn
app = create_app()

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Conversational RAG Server")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
