"""Health check endpoints."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from ia_agent_fwk.api.dependencies import get_memory_backend
from ia_agent_fwk.api.models import HealthResponse, ReadinessResponse
from ia_agent_fwk.memory.base import MemoryBackend  # noqa: TC001
from ia_agent_fwk.observability.metrics import get_metrics_collector

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Liveness probe. Returns 200 if the process is alive."""
    collector = get_metrics_collector()
    collector.increment("api_health_checks_total", labels={"type": "liveness", "status": "healthy"})
    return HealthResponse(status="healthy")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(
    memory: Annotated[MemoryBackend, Depends(get_memory_backend)],
) -> JSONResponse:
    """Readiness probe. Checks memory backend health."""
    collector = get_metrics_collector()
    start = time.monotonic()

    memory_healthy = await memory.health_check()

    duration_ms = (time.monotonic() - start) * 1000
    checks = {"memory": "healthy" if memory_healthy else "unhealthy"}

    status = "ready" if memory_healthy else "not_ready"
    collector.increment("api_health_checks_total", labels={"type": "readiness", "status": status})
    collector.observe("api_health_check_duration_seconds", duration_ms / 1000)

    if memory_healthy:
        response = ReadinessResponse(status="ready", checks=checks)
        return JSONResponse(content=response.model_dump(), status_code=200)

    response = ReadinessResponse(status="not_ready", checks=checks)
    return JSONResponse(content=response.model_dump(), status_code=503)
