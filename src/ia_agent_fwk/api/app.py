"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from ia_agent_fwk.agents.exceptions import AgentConfigError
from ia_agent_fwk.api.middleware import RequestIdMiddleware, get_request_id
from ia_agent_fwk.api.models import ErrorDetail, ErrorResponse
from ia_agent_fwk.api.routes import (
    agents,
    conversations,
    health,
    integrations,
    jobs,
    metrics,
    rag,
    schedules,
    streaming,
    triggers,
)
from ia_agent_fwk.execution.celery_app import get_celery_app
from ia_agent_fwk.execution.exceptions import (
    ExecutionError,
    InvalidCronExpressionError,
    JobNotFoundError,
    ScheduleNotFoundError,
    TriggerNotFoundError,
)
from ia_agent_fwk.execution.manager import JobManager
from ia_agent_fwk.execution.scheduler import ScheduleManager
from ia_agent_fwk.execution.triggers import TriggerManager
from ia_agent_fwk.llm.exceptions import LLMProviderError
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.exceptions import MemoryRetrieveError, MemoryStoreError
from ia_agent_fwk.memory.factory import MemoryFactory
from ia_agent_fwk.observability.middleware import ObservabilityMiddleware
from ia_agent_fwk.security.audit import AuditLogger
from ia_agent_fwk.security.exceptions import RateLimitExceededError
from ia_agent_fwk.security.rate_limiter import SlidingWindowRateLimiter, parse_rate
from ia_agent_fwk.security.sanitizer import sanitize_error_message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ia_agent_fwk.config.settings import AppSettings

logger = logging.getLogger(__name__)


def _register_example_agents() -> None:
    """Register built-in example agent types in the AgentRegistry."""
    from ia_agent_fwk.agents.base import Agent  # noqa: PLC0415
    from ia_agent_fwk.agents.registry import AgentRegistry  # noqa: PLC0415

    class _CustomerSupportAgent(Agent):
        @property
        def agent_type(self) -> str:
            return "customer_support"

    class _DocumentProcessorAgent(Agent):
        @property
        def agent_type(self) -> str:
            return "document_processor"

    class _FinanceAgent(Agent):
        @property
        def agent_type(self) -> str:
            return "finance"

    for name, cls in [
        ("customer_support", _CustomerSupportAgent),
        ("document_processor", _DocumentProcessorAgent),
        ("finance", _FinanceAgent),
    ]:
        AgentRegistry.register(name, cls, replace=True)  # type: ignore[type-abstract]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager.

    Startup: create memory backends, store on app.state.
    Shutdown: close backends, log clean exit.
    """
    settings: AppSettings = app.state.settings

    # Activate structured JSON logging (must happen in lifespan, after
    # uvicorn has configured its own logging, so we can override it)
    from ia_agent_fwk.observability.logging import setup_logging  # noqa: PLC0415

    setup_logging(settings.observability.logging)

    # Initialize distributed tracing (OTel → Tempo)
    from ia_agent_fwk.observability.tracing import TracingManager  # noqa: PLC0415

    tracing_manager = TracingManager(settings.observability.tracing)
    tracing_manager.setup()
    app.state.tracing_manager = tracing_manager

    # Register example agent types so they are available via the API.
    _register_example_agents()

    logger.info(
        "Starting %s v%s (%s)",
        settings.app.name,
        settings.app.version,
        settings.app.environment,
    )

    # Create memory backend
    memory_backend = MemoryFactory.create(settings.memory)
    app.state.memory_backend = memory_backend

    # Create conversation backend
    conversation_backend = ConversationMemoryBackend(
        max_history=settings.memory.backends.conversation.max_history,
    )
    app.state.conversation_backend = conversation_backend

    # Create job manager (Celery + Redis for job index)
    celery = get_celery_app(settings)
    redis_client = None
    broker_url = settings.execution.celery.broker_url
    if broker_url and broker_url.startswith("redis://"):
        try:
            import redis  # noqa: PLC0415

            redis_client = redis.Redis.from_url(broker_url, decode_responses=True)
            redis_client.ping()
            logger.info("Redis client connected for job index")
        except Exception:  # noqa: BLE001
            logger.warning("Failed to connect Redis for job index; listing disabled")
            redis_client = None
    job_manager = JobManager(celery_app=celery, redis_client=redis_client)
    app.state.job_manager = job_manager

    # Create rate limiter from security settings
    rate_cfg = settings.security.rate_limiting
    if rate_cfg.enabled:
        limit, window = parse_rate(rate_cfg.default_rate)
        rate_limiter = SlidingWindowRateLimiter(
            default_limit=limit,
            default_window_seconds=window,
        )
    else:
        rate_limiter = None
    app.state.rate_limiter = rate_limiter

    # Create audit logger
    app.state.audit_logger = AuditLogger()

    # Create schedule manager
    schedule_manager = ScheduleManager()
    app.state.schedule_manager = schedule_manager

    # Create trigger manager
    trigger_manager = TriggerManager(job_manager=job_manager)
    app.state.trigger_manager = trigger_manager

    yield

    # Shutdown
    tracing_manager.shutdown()
    await memory_backend.close()
    await conversation_backend.close()
    logger.info("Application shut down cleanly")


def _make_error_response(
    status_code: int,
    code: str,
    message: str,
    detail: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    """Build a JSON error response with the standard envelope."""
    error = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            detail=detail,
            request_id=get_request_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(),
        headers={"X-Request-ID": get_request_id()},
    )


def _register_exception_handlers(app: FastAPI) -> None:  # noqa: C901
    """Register all exception handlers on the FastAPI application."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,  # noqa: ARG001
        exc: RequestValidationError,
    ) -> JSONResponse:
        detail_list: list[dict[str, Any]] = [
            {
                "loc": list(err.get("loc", [])),
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in exc.errors()
        ]
        return _make_error_response(
            status_code=422,
            code="VALIDATION_ERROR",
            message="Request validation failed",
            detail=detail_list,
        )

    @app.exception_handler(AgentConfigError)
    async def agent_config_error_handler(
        request: Request,  # noqa: ARG001
        exc: AgentConfigError,
    ) -> JSONResponse:
        return _make_error_response(
            status_code=404,
            code="RESOURCE_NOT_FOUND",
            message=str(exc),
        )

    @app.exception_handler(MemoryRetrieveError)
    async def memory_retrieve_error_handler(
        request: Request,  # noqa: ARG001
        exc: MemoryRetrieveError,
    ) -> JSONResponse:
        return _make_error_response(
            status_code=404,
            code="RESOURCE_NOT_FOUND",
            message=str(exc),
        )

    @app.exception_handler(MemoryStoreError)
    async def memory_store_error_handler(
        request: Request,  # noqa: ARG001
        exc: MemoryStoreError,  # noqa: ARG001
    ) -> JSONResponse:
        return _make_error_response(
            status_code=500,
            code="MEMORY_STORE_ERROR",
            message="Memory store operation failed",
        )

    @app.exception_handler(LLMProviderError)
    async def llm_error_handler(
        request: Request,  # noqa: ARG001
        exc: LLMProviderError,  # noqa: ARG001
    ) -> JSONResponse:
        return _make_error_response(
            status_code=502,
            code="LLM_PROVIDER_ERROR",
            message="LLM provider error",
        )

    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(
        request: Request,  # noqa: ARG001
        exc: JobNotFoundError,
    ) -> JSONResponse:
        return _make_error_response(
            status_code=404,
            code="RESOURCE_NOT_FOUND",
            message=str(exc),
        )

    @app.exception_handler(ScheduleNotFoundError)
    async def schedule_not_found_handler(
        request: Request,  # noqa: ARG001
        exc: ScheduleNotFoundError,
    ) -> JSONResponse:
        return _make_error_response(
            status_code=404,
            code="RESOURCE_NOT_FOUND",
            message=str(exc),
        )

    @app.exception_handler(InvalidCronExpressionError)
    async def invalid_cron_handler(
        request: Request,  # noqa: ARG001
        exc: InvalidCronExpressionError,
    ) -> JSONResponse:
        return _make_error_response(
            status_code=422,
            code="VALIDATION_ERROR",
            message=str(exc),
        )

    @app.exception_handler(TriggerNotFoundError)
    async def trigger_not_found_handler(
        request: Request,  # noqa: ARG001
        exc: TriggerNotFoundError,
    ) -> JSONResponse:
        return _make_error_response(
            status_code=404,
            code="RESOURCE_NOT_FOUND",
            message=str(exc),
        )

    @app.exception_handler(ExecutionError)
    async def execution_error_handler(
        request: Request,  # noqa: ARG001
        exc: ExecutionError,
    ) -> JSONResponse:
        return _make_error_response(
            status_code=500,
            code="EXECUTION_ERROR",
            message=str(exc),
        )

    @app.exception_handler(RateLimitExceededError)
    async def rate_limit_error_handler(
        request: Request,  # noqa: ARG001
        exc: RateLimitExceededError,
    ) -> JSONResponse:
        error = ErrorResponse(
            error=ErrorDetail(
                code="RATE_LIMIT_EXCEEDED",
                message="Too many requests. Please retry later.",
                request_id=get_request_id(),
                timestamp=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            )
        )
        return JSONResponse(
            status_code=429,
            content=error.model_dump(),
            headers={
                "X-Request-ID": get_request_id(),
                "Retry-After": str(exc.retry_after),
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(
        request: Request,  # noqa: ARG001
        exc: Exception,
    ) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return _make_error_response(
            status_code=500,
            code="INTERNAL_ERROR",
            message=sanitize_error_message(exc),
        )


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure a FastAPI application.

    Parameters
    ----------
    settings:
        Application settings. If ``None``, loads via ``load_config()``.

    """
    if settings is None:
        from ia_agent_fwk.config.loader import load_config  # noqa: PLC0415

        settings = load_config()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        lifespan=lifespan,
    )

    # Store settings on app.state for dependency injection
    app.state.settings = settings

    # Add request ID middleware (outermost)
    app.add_middleware(RequestIdMiddleware)

    # Add observability middleware (metrics + tracing)
    app.add_middleware(ObservabilityMiddleware)

    # Add CORS middleware.
    # Disable allow_credentials when allow_origins is a wildcard to prevent
    # insecure reflected-origin behaviour (see F-010).
    origins = settings.server.cors.allow_origins
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=settings.server.cors.allow_methods,
        allow_headers=settings.server.cors.allow_headers,
        allow_credentials=allow_credentials,
    )

    # Register exception handlers
    _register_exception_handlers(app)

    # ------------------------------------------------------------------
    # Include routers
    # ------------------------------------------------------------------

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(agents.router)
    app.include_router(conversations.router)
    app.include_router(jobs.router)
    app.include_router(schedules.router)
    app.include_router(triggers.router)
    app.include_router(streaming.router)
    app.include_router(integrations.router)
    app.include_router(rag.router)

    return app
