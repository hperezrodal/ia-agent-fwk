"""Job lifecycle management for background agent execution."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.execution.models import JobInfo, JobStatus
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

if TYPE_CHECKING:
    from celery import Celery

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

# Mapping from Celery states to framework job statuses.
_STATE_MAP: dict[str, JobStatus] = {
    "PENDING": JobStatus.PENDING,
    "RECEIVED": JobStatus.PENDING,
    "STARTED": JobStatus.STARTED,
    "RUNNING": JobStatus.STARTED,
    "SUCCESS": JobStatus.SUCCESS,
    "FAILURE": JobStatus.FAILURE,
    "REVOKED": JobStatus.REVOKED,
    "REJECTED": JobStatus.FAILURE,
    "RETRY": JobStatus.PENDING,
}

_JOB_INDEX_KEY = "ia_agent_fwk:jobs"


def _meta_key(job_id: str) -> str:
    return f"ia_agent_fwk:job:{job_id}:meta"


class JobManager:
    """Manage async job submission, status tracking, and result retrieval.

    Uses Celery ``AsyncResult`` for individual job status and Redis sorted
    sets for job listing.

    Parameters
    ----------
    celery_app:
        Configured Celery application instance.
    redis_client:
        Optional Redis client for the job index.  When ``None`` the
        ``list_jobs`` method returns an empty list.

    """

    def __init__(
        self,
        celery_app: Celery,
        redis_client: Any | None = None,
    ) -> None:
        self._celery_app = celery_app
        self._redis: Any | None = redis_client
        # Track jobs whose completion metrics have already been emitted
        # to avoid double-counting when get_result() is called multiple times.
        self._tracked_completions: set[str] = set()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(
        self,
        agent_type: str,
        prompt: str,
        conversation_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> str:
        """Submit an agent execution task and return the job ID.

        Parameters
        ----------
        agent_type:
            Registered agent type name.
        prompt:
            User input text.
        conversation_id:
            Optional conversation ID.
        config_overrides:
            Optional execution overrides.

        Returns
        -------
        str
            The Celery task ID serving as the job ID.

        """
        from ia_agent_fwk.execution.tasks import execute_agent_task  # noqa: PLC0415

        collector = get_metrics_collector()
        start = time.monotonic()

        with _tracer.start_as_current_span(
            "execution.job.submit",
            attributes={"execution.agent_type": agent_type},
        ) as span:
            result = execute_agent_task.apply_async(
                args=[agent_type, prompt],
                kwargs={
                    "conversation_id": conversation_id,
                    "config_overrides": config_overrides,
                },
            )
            job_id = str(result.id)
            span.set_attribute("execution.job_id", job_id)

            # Store in job index if Redis is available
            if self._redis is not None:
                self._redis.zadd(_JOB_INDEX_KEY, {result.id: time.time()})
                self._redis.hset(
                    _meta_key(result.id),
                    mapping={
                        "agent_type": agent_type,
                        "prompt": prompt[:200],
                        "created_at": str(time.time()),
                    },
                )

            duration_ms = (time.monotonic() - start) * 1000
            collector.increment(
                "execution_job_submissions_total",
                labels={"agent_type": agent_type},
            )
            collector.observe("execution_job_submission_duration_seconds", duration_ms / 1000)

            logger.info(
                "Job submitted: job_id=%s, agent_type=%s (%.1fms)",
                job_id,
                agent_type,
                duration_ms,
                extra={
                    "execution_data": {
                        "event": "job_submitted",
                        "job_id": job_id,
                        "agent_type": agent_type,
                        "duration_ms": round(duration_ms, 1),
                        "redis_tracking": self._redis is not None,
                    }
                },
            )

        return job_id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, job_id: str) -> JobInfo:
        """Return the current status of a job.

        Raises
        ------
        JobNotFoundError
            If the job ID is not found.

        """
        collector = get_metrics_collector()

        with _tracer.start_as_current_span(
            "execution.job.get_status",
            attributes={"execution.job_id": job_id},
        ) as span:
            async_result = self._celery_app.AsyncResult(job_id)
            celery_state: str = str(async_result.state)

            status = _STATE_MAP.get(celery_state, JobStatus.UNKNOWN)
            span.set_attribute("execution.status", status.value)

            # Attempt to get metadata from Redis
            agent_type: str | None = None
            created_at: str | None = None
            if self._redis is not None:
                meta = self._redis.hgetall(_meta_key(job_id))
                if meta:
                    agent_type = meta.get("agent_type") or meta.get(b"agent_type", b"").decode(
                        "utf-8", errors="replace"
                    )
                    created_at = meta.get("created_at") or meta.get(b"created_at", b"").decode(
                        "utf-8", errors="replace"
                    )

            # Extract error from result if failed
            error: str | None = None
            if status == JobStatus.FAILURE:
                result = async_result.result
                error = str(result) if result else None

            collector.increment(
                "execution_job_status_queries_total",
                labels={"status": status.value},
            )

            logger.debug(
                "Job status queried: job_id=%s, status=%s",
                job_id,
                status.value,
                extra={
                    "execution_data": {
                        "event": "job_status_queried",
                        "job_id": job_id,
                        "status": status.value,
                        "agent_type": agent_type,
                    }
                },
            )

        return JobInfo(
            job_id=job_id,
            agent_type=agent_type,
            status=status,
            created_at=created_at,
            error=error,
        )

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        """Return the task result if available, else ``None``."""
        collector = get_metrics_collector()

        with _tracer.start_as_current_span(
            "execution.job.get_result",
            attributes={"execution.job_id": job_id},
        ) as span:
            async_result = self._celery_app.AsyncResult(job_id)
            if async_result.ready():
                result: Any = async_result.result
                if isinstance(result, dict):
                    span.set_attribute("execution.result_found", True)  # noqa: FBT003
                    collector.increment(
                        "execution_job_result_queries_total",
                        labels={"found": "true"},
                    )

                    # Emit task completion metrics from the API side.
                    # The Celery worker runs in a separate process so its
                    # in-memory metrics never reach the /metrics endpoint.
                    self._track_task_completion(job_id, result, collector)

                    logger.debug(
                        "Job result retrieved: job_id=%s",
                        job_id,
                        extra={
                            "execution_data": {
                                "event": "job_result_retrieved",
                                "job_id": job_id,
                                "found": True,
                            }
                        },
                    )
                    return dict(result)

            span.set_attribute("execution.result_found", False)  # noqa: FBT003
            collector.increment(
                "execution_job_result_queries_total",
                labels={"found": "false"},
            )
            return None

    def _track_task_completion(
        self,
        job_id: str,
        result: dict[str, Any],
        collector: Any,
    ) -> None:
        """Emit task-level metrics from a completed job result.

        Called from ``get_result()`` in the API process so Prometheus
        can scrape task completion data.  Each job is only tracked once.
        """
        if job_id in self._tracked_completions:
            return
        self._tracked_completions.add(job_id)

        # Determine agent_type from Redis metadata
        agent_type = "unknown"
        if self._redis is not None:
            meta = self._redis.hgetall(_meta_key(job_id))
            if meta:
                agent_type = meta.get("agent_type", "unknown")

        state = result.get("state", "")
        outcome = "success" if state in {"COMPLETED", "completed"} else "error"
        duration_ms = result.get("duration_ms", 0)
        iterations = result.get("iterations", 0)
        total_tokens = 0
        usage = result.get("usage")
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens", 0)

        collector.increment(
            "execution_task_completed_total",
            labels={"agent_type": agent_type, "outcome": outcome},
        )
        if duration_ms > 0:
            collector.observe(
                "execution_task_duration_seconds",
                duration_ms / 1000,
            )

        logger.info(
            "Task completion tracked: job_id=%s, agent_type=%s, outcome=%s, duration_ms=%.0f",
            job_id,
            agent_type,
            outcome,
            duration_ms,
            extra={
                "execution_data": {
                    "event": "task_completion_tracked",
                    "job_id": job_id,
                    "agent_type": agent_type,
                    "outcome": outcome,
                    "duration_ms": round(duration_ms, 1),
                    "iterations": iterations,
                    "total_tokens": total_tokens,
                }
            },
        )

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Cancel a running or pending job.

        Returns
        -------
        bool
            ``True`` if the revocation signal was sent, ``False`` if
            the job is already in a terminal state.

        """
        collector = get_metrics_collector()

        with _tracer.start_as_current_span(
            "execution.job.cancel",
            attributes={"execution.job_id": job_id},
        ) as span:
            async_result = self._celery_app.AsyncResult(job_id)
            celery_state: str = str(async_result.state)

            if celery_state in {"SUCCESS", "FAILURE", "REVOKED"}:
                span.set_attribute("execution.cancel_outcome", "already_terminal")
                collector.increment(
                    "execution_job_cancellations_total",
                    labels={"outcome": "already_terminal"},
                )
                logger.info(
                    "Job cancel skipped (terminal): job_id=%s, state=%s",
                    job_id,
                    celery_state,
                    extra={
                        "execution_data": {
                            "event": "job_cancel_skipped",
                            "job_id": job_id,
                            "celery_state": celery_state,
                        }
                    },
                )
                return False

            async_result.revoke(terminate=True, signal="SIGTERM")
            span.set_attribute("execution.cancel_outcome", "cancelled")
            collector.increment(
                "execution_job_cancellations_total",
                labels={"outcome": "cancelled"},
            )
            logger.info(
                "Job cancelled: job_id=%s, previous_state=%s",
                job_id,
                celery_state,
                extra={
                    "execution_data": {
                        "event": "job_cancelled",
                        "job_id": job_id,
                        "previous_state": celery_state,
                    }
                },
            )
            return True

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_jobs(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[JobInfo], int]:
        """List recent jobs from the Redis job index.

        Returns
        -------
        tuple[list[JobInfo], int]
            A tuple of ``(job_summaries, total_count)``.

        """
        collector = get_metrics_collector()

        with _tracer.start_as_current_span(
            "execution.job.list",
            attributes={"execution.limit": limit, "execution.offset": offset},
        ) as span:
            if self._redis is None:
                span.set_attribute("execution.redis_available", False)  # noqa: FBT003
                collector.increment("execution_job_list_queries_total")
                return [], 0

            span.set_attribute("execution.redis_available", True)  # noqa: FBT003
            total: int = self._redis.zcard(_JOB_INDEX_KEY)

            # Retrieve job IDs in reverse chronological order
            raw_ids: list[Any] = self._redis.zrevrange(
                _JOB_INDEX_KEY,
                offset,
                offset + limit - 1,
            )

            jobs: list[JobInfo] = []
            for raw_id in raw_ids:
                job_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
                try:
                    info = self.get_status(job_id)
                except Exception:  # noqa: BLE001
                    info = JobInfo(job_id=job_id, status=JobStatus.UNKNOWN)
                jobs.append(info)

            span.set_attribute("execution.total_jobs", total)
            span.set_attribute("execution.returned_jobs", len(jobs))
            collector.increment("execution_job_list_queries_total")

            logger.debug(
                "Job list queried: total=%d, returned=%d, limit=%d, offset=%d",
                total,
                len(jobs),
                limit,
                offset,
                extra={
                    "execution_data": {
                        "event": "job_list_queried",
                        "total": total,
                        "returned": len(jobs),
                        "limit": limit,
                        "offset": offset,
                    }
                },
            )

        return jobs, int(total)
