"""Celery task definitions for background agent execution."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ia_agent_fwk.execution.celery_app import celery_app
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


@celery_app.task(
    name="ia_agent_fwk.execute_agent",
    bind=True,
    acks_late=True,
    max_retries=0,
)
def execute_agent_task(
    self: Any,
    agent_type: str,
    prompt: str,
    conversation_id: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute an agent in the background.

    This task bridges the async ``Agent.run()`` method into the synchronous
    Celery worker process by creating a fresh asyncio event loop via
    ``asyncio.run()``.

    Parameters
    ----------
    self:
        Bound Celery task instance (``bind=True``).
    agent_type:
        Registered agent type name (e.g. ``"conversational"``).
    prompt:
        User input text.
    conversation_id:
        Optional conversation ID for continuity.
    config_overrides:
        Optional execution options.

    Returns
    -------
    dict
        The ``AgentResult`` as a JSON-compatible dict.

    """
    collector = get_metrics_collector()
    job_id = str(self.request.id)

    self.update_state(state="RUNNING", meta={"agent_type": agent_type})

    start_time = time.monotonic()
    logger.info(
        "Starting agent execution: type=%s, job_id=%s",
        agent_type,
        job_id,
        extra={
            "execution_data": {
                "event": "task_started",
                "job_id": job_id,
                "agent_type": agent_type,
            }
        },
    )

    with _tracer.start_as_current_span(
        "execution.task.run",
        attributes={
            "execution.agent_type": agent_type,
            "execution.job_id": job_id,
        },
    ) as span:
        try:
            result_dict: dict[str, Any] = asyncio.run(
                _run_agent(agent_type, prompt, conversation_id, config_overrides),
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            span.set_attribute("execution.duration_ms", duration_ms)
            span.set_attribute("execution.outcome", "error")
            span.record_exception(exc)
            collector.increment(
                "execution_task_completed_total",
                labels={"agent_type": agent_type, "outcome": "error"},
            )
            collector.observe(
                "execution_task_duration_seconds",
                duration_ms / 1000,
            )
            logger.exception(
                "Agent execution failed: type=%s, job_id=%s, duration_ms=%.0f",
                agent_type,
                job_id,
                duration_ms,
                extra={
                    "execution_data": {
                        "event": "task_failed",
                        "job_id": job_id,
                        "agent_type": agent_type,
                        "duration_ms": round(duration_ms, 1),
                        "error": str(exc),
                    }
                },
            )
            return {
                "output": "",
                "state": "FAILED",
                "error": str(exc),
                "duration_ms": duration_ms,
                "iterations": 0,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        duration_ms = (time.monotonic() - start_time) * 1000
        iterations = result_dict.get("iterations", 0)
        total_tokens = 0
        usage = result_dict.get("usage")
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens", 0)

        span.set_attribute("execution.duration_ms", duration_ms)
        span.set_attribute("execution.outcome", "success")
        span.set_attribute("execution.iterations", iterations)
        span.set_attribute("execution.total_tokens", total_tokens)
        collector.increment(
            "execution_task_completed_total",
            labels={"agent_type": agent_type, "outcome": "success"},
        )
        collector.observe(
            "execution_task_duration_seconds",
            duration_ms / 1000,
        )
        logger.info(
            "Agent execution completed: type=%s, job_id=%s, duration_ms=%.0f",
            agent_type,
            job_id,
            duration_ms,
            extra={
                "execution_data": {
                    "event": "task_completed",
                    "job_id": job_id,
                    "agent_type": agent_type,
                    "duration_ms": round(duration_ms, 1),
                    "iterations": iterations,
                    "total_tokens": total_tokens,
                }
            },
        )

    return result_dict


_agents_registered = False


def _ensure_agents_registered() -> None:
    """Register agent types in the worker process (idempotent)."""
    global _agents_registered  # noqa: PLW0603
    if _agents_registered:
        return

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

    _agents_registered = True
    logger.info("Agent types registered in worker process")


async def _run_agent(
    agent_type: str,
    prompt: str,
    conversation_id: str | None,  # noqa: ARG001
    config_overrides: dict[str, Any] | None,  # noqa: ARG001
) -> dict[str, Any]:
    """Async helper that creates and runs the agent.

    Imports are deferred to avoid circular dependencies and to ensure
    each worker process gets fresh instances.
    """
    from ia_agent_fwk.agents.config import AgentConfig  # noqa: PLC0415
    from ia_agent_fwk.agents.factory import AgentFactory  # noqa: PLC0415
    from ia_agent_fwk.agents.registry import AgentRegistry  # noqa: PLC0415
    from ia_agent_fwk.config.loader import load_config  # noqa: PLC0415

    settings = load_config()

    # Ensure agent types are registered in the worker process.
    # The API registers them in its lifespan; workers must do it lazily.
    _ensure_agents_registered()

    # Validate agent type (raises AgentConfigError if unknown)
    AgentRegistry.get(agent_type)

    agent_config = AgentConfig(
        name=f"{agent_type}-worker",
        agent_type=agent_type,
        provider_name=settings.llm.default_provider,
    )

    agent = AgentFactory.create(agent_config, settings.llm)
    result = await agent.run(prompt)

    return result.model_dump(mode="json")
