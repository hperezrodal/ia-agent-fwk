"""Schedule management for periodic agent execution via Celery Beat."""

from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.execution.exceptions import (
    InvalidCronExpressionError,
    ScheduleNotFoundError,
)
from ia_agent_fwk.observability.metrics import get_metrics_collector

if TYPE_CHECKING:
    from ia_agent_fwk.execution.models import ScheduleDefinition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cron expression validation
# ---------------------------------------------------------------------------

# Matches standard 5-field cron: minute hour day-of-month month day-of-week.
# Each field accepts: *, digits, ranges (1-5), steps (*/5), lists (1,3,5),
# and combinations thereof.
_CRON_FIELD = r"(?:\*(?:/\d+)?|\d+(?:-\d+)?(?:/\d+)?(?:,\d+(?:-\d+)?(?:/\d+)?)*)"
_CRON_PATTERN = re.compile(
    rf"^\s*{_CRON_FIELD}\s+{_CRON_FIELD}\s+{_CRON_FIELD}\s+{_CRON_FIELD}\s+{_CRON_FIELD}\s*$",
)


def validate_cron_expression(expression: str) -> None:
    """Validate a 5-field cron expression.

    Raises
    ------
    InvalidCronExpressionError
        If the expression does not match the expected format.

    """
    if not _CRON_PATTERN.match(expression):
        msg = f"Invalid cron expression: {expression!r}"
        raise InvalidCronExpressionError(msg)


# ---------------------------------------------------------------------------
# ScheduleManager
# ---------------------------------------------------------------------------


class ScheduleManager:
    """Manage cron-based schedules for periodic agent execution.

    Stores schedule definitions in-memory (V1). Generates a Celery Beat
    compatible schedule dictionary for integration with ``celery_app.conf``.
    """

    def __init__(self) -> None:
        self._schedules: dict[str, ScheduleDefinition] = {}
        self._schedule_ids: dict[str, str] = {}  # schedule_id -> name (for ordering)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_schedule(self, definition: ScheduleDefinition) -> str:
        """Register a new cron schedule.

        Parameters
        ----------
        definition:
            The schedule definition to register.

        Returns
        -------
        str
            The generated schedule ID.

        Raises
        ------
        InvalidCronExpressionError
            If the cron expression is invalid.

        """
        validate_cron_expression(definition.cron_expression)

        schedule_id = str(uuid.uuid4())
        self._schedules[schedule_id] = definition
        self._schedule_ids[schedule_id] = definition.name

        collector = get_metrics_collector()
        collector.increment(
            "execution_schedule_operations_total",
            labels={"operation": "add"},
        )
        logger.info(
            "Schedule added: id=%s, name=%s, cron=%s",
            schedule_id,
            definition.name,
            definition.cron_expression,
            extra={
                "execution_data": {
                    "event": "schedule_added",
                    "schedule_id": schedule_id,
                    "name": definition.name,
                    "agent_type": definition.agent_type,
                    "cron_expression": definition.cron_expression,
                    "enabled": definition.enabled,
                    "total_schedules": len(self._schedules),
                }
            },
        )
        return schedule_id

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule by ID.

        Returns
        -------
        bool
            ``True`` if the schedule was found and removed.

        """
        if schedule_id in self._schedules:
            definition = self._schedules.pop(schedule_id)
            self._schedule_ids.pop(schedule_id, None)
            collector = get_metrics_collector()
            collector.increment(
                "execution_schedule_operations_total",
                labels={"operation": "remove"},
            )
            logger.info(
                "Schedule removed: id=%s, name=%s",
                schedule_id,
                definition.name,
                extra={
                    "execution_data": {
                        "event": "schedule_removed",
                        "schedule_id": schedule_id,
                        "name": definition.name,
                        "total_schedules": len(self._schedules),
                    }
                },
            )
            return True
        return False

    def get_schedule(self, schedule_id: str) -> ScheduleDefinition | None:
        """Return a schedule definition by ID, or ``None``."""
        return self._schedules.get(schedule_id)

    def list_schedules(self) -> list[tuple[str, ScheduleDefinition]]:
        """Return all schedules as ``(schedule_id, definition)`` pairs."""
        return list(self._schedules.items())

    def update_schedule(self, schedule_id: str, definition: ScheduleDefinition) -> None:
        """Update an existing schedule.

        Raises
        ------
        ScheduleNotFoundError
            If the schedule ID does not exist.
        InvalidCronExpressionError
            If the new cron expression is invalid.

        """
        if schedule_id not in self._schedules:
            msg = f"Schedule not found: {schedule_id}"
            raise ScheduleNotFoundError(msg)

        validate_cron_expression(definition.cron_expression)

        self._schedules[schedule_id] = definition
        self._schedule_ids[schedule_id] = definition.name

        collector = get_metrics_collector()
        collector.increment(
            "execution_schedule_operations_total",
            labels={"operation": "update"},
        )
        logger.info(
            "Schedule updated: id=%s, name=%s",
            schedule_id,
            definition.name,
            extra={
                "execution_data": {
                    "event": "schedule_updated",
                    "schedule_id": schedule_id,
                    "name": definition.name,
                    "agent_type": definition.agent_type,
                    "cron_expression": definition.cron_expression,
                }
            },
        )

    # ------------------------------------------------------------------
    # Celery Beat integration
    # ------------------------------------------------------------------

    def generate_beat_schedule(self) -> dict[str, Any]:
        """Generate a Celery Beat-compatible schedule dictionary.

        Only enabled schedules are included. Each entry maps to the
        ``ia_agent_fwk.execute_agent`` task with the schedule's agent_type
        and prompt as arguments.

        Returns
        -------
        dict[str, Any]
            Dictionary suitable for ``celery_app.conf.beat_schedule``.

        """
        from celery.schedules import crontab  # noqa: PLC0415

        beat_schedule: dict[str, Any] = {}

        for schedule_id, definition in self._schedules.items():
            if not definition.enabled:
                continue

            parts = definition.cron_expression.strip().split()
            if len(parts) != 5:  # noqa: PLR2004
                continue  # skip malformed (should not happen after validation)

            beat_schedule[f"schedule-{schedule_id}"] = {
                "task": "ia_agent_fwk.execute_agent",
                "schedule": crontab(
                    minute=parts[0],
                    hour=parts[1],
                    day_of_month=parts[2],
                    month_of_year=parts[3],
                    day_of_week=parts[4],
                ),
                "args": [definition.agent_type, definition.prompt],
                "kwargs": {
                    "config_overrides": definition.config_overrides,
                },
            }

        return beat_schedule
