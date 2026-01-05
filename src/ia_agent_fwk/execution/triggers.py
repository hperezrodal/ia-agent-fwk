"""Event trigger management for webhook-driven agent execution."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

if TYPE_CHECKING:
    from ia_agent_fwk.execution.manager import JobManager
    from ia_agent_fwk.execution.models import EventTrigger

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class TriggerManager:
    """Manage event triggers that fire agent executions on incoming events.

    Each trigger watches for a specific ``event_type``. When
    ``fire_trigger()`` is called with a matching event, the trigger's
    ``prompt_template`` is rendered with the event payload and an agent
    execution job is submitted through ``JobManager``.

    Parameters
    ----------
    job_manager:
        The ``JobManager`` instance used to submit agent execution jobs.

    """

    def __init__(self, job_manager: JobManager) -> None:
        self._job_manager = job_manager
        self._triggers: dict[str, EventTrigger] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_trigger(self, trigger: EventTrigger) -> str:
        """Register a new event trigger.

        Parameters
        ----------
        trigger:
            The trigger definition.

        Returns
        -------
        str
            The generated trigger ID.

        """
        trigger_id = str(uuid.uuid4())
        self._triggers[trigger_id] = trigger

        collector = get_metrics_collector()
        collector.increment(
            "execution_trigger_operations_total",
            labels={"operation": "register"},
        )
        logger.info(
            "Trigger registered: id=%s, name=%s, event_type=%s",
            trigger_id,
            trigger.name,
            trigger.event_type,
            extra={
                "execution_data": {
                    "event": "trigger_registered",
                    "trigger_id": trigger_id,
                    "name": trigger.name,
                    "agent_type": trigger.agent_type,
                    "event_type": trigger.event_type,
                    "total_triggers": len(self._triggers),
                }
            },
        )
        return trigger_id

    def unregister_trigger(self, trigger_id: str) -> bool:
        """Unregister a trigger by ID.

        Returns
        -------
        bool
            ``True`` if the trigger was found and removed.

        """
        if trigger_id in self._triggers:
            trigger = self._triggers.pop(trigger_id)
            collector = get_metrics_collector()
            collector.increment(
                "execution_trigger_operations_total",
                labels={"operation": "unregister"},
            )
            logger.info(
                "Trigger unregistered: id=%s, name=%s",
                trigger_id,
                trigger.name,
                extra={
                    "execution_data": {
                        "event": "trigger_unregistered",
                        "trigger_id": trigger_id,
                        "name": trigger.name,
                        "total_triggers": len(self._triggers),
                    }
                },
            )
            return True
        return False

    def list_triggers(self) -> list[tuple[str, EventTrigger]]:
        """Return all triggers as ``(trigger_id, trigger)`` pairs."""
        return list(self._triggers.items())

    def get_trigger(self, trigger_id: str) -> EventTrigger | None:
        """Return a trigger by ID, or ``None``."""
        return self._triggers.get(trigger_id)

    # ------------------------------------------------------------------
    # Fire
    # ------------------------------------------------------------------

    def fire_trigger(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> tuple[str, str] | None:
        """Find a matching trigger and submit an agent execution job.

        Iterates over registered triggers and fires the **first** one
        whose ``event_type`` matches. The prompt is produced by rendering
        the trigger's ``prompt_template`` with the serialised payload.

        Parameters
        ----------
        event_type:
            The event type string to match against registered triggers.
        payload:
            The event payload data passed to the prompt template.

        Returns
        -------
        tuple[str, str] | None
            ``(trigger_id, job_id)`` if a matching trigger was found and
            the job was submitted, otherwise ``None``.

        """
        collector = get_metrics_collector()
        start = time.monotonic()

        with _tracer.start_as_current_span(
            "execution.trigger.fire",
            attributes={"execution.event_type": event_type},
        ) as span:
            for trigger_id, trigger in self._triggers.items():
                if trigger.event_type == event_type:
                    prompt = self._render_prompt(trigger.prompt_template, payload)

                    job_id = self._job_manager.submit(
                        agent_type=trigger.agent_type,
                        prompt=prompt,
                        config_overrides=trigger.config_overrides,
                    )

                    duration_ms = (time.monotonic() - start) * 1000
                    span.set_attribute("execution.matched", True)  # noqa: FBT003
                    span.set_attribute("execution.trigger_id", trigger_id)
                    span.set_attribute("execution.job_id", job_id)
                    span.set_attribute("execution.duration_ms", duration_ms)
                    collector.increment(
                        "execution_trigger_fires_total",
                        labels={"event_type": event_type, "matched": "true"},
                    )
                    collector.observe(
                        "execution_trigger_fire_duration_seconds",
                        duration_ms / 1000,
                    )
                    logger.info(
                        "Trigger fired: trigger_id=%s, event_type=%s, job_id=%s (%.1fms)",
                        trigger_id,
                        event_type,
                        job_id,
                        duration_ms,
                        extra={
                            "execution_data": {
                                "event": "trigger_fired",
                                "trigger_id": trigger_id,
                                "trigger_name": trigger.name,
                                "event_type": event_type,
                                "agent_type": trigger.agent_type,
                                "job_id": job_id,
                                "duration_ms": round(duration_ms, 1),
                            }
                        },
                    )
                    return trigger_id, job_id

            span.set_attribute("execution.matched", False)  # noqa: FBT003
            collector.increment(
                "execution_trigger_fires_total",
                labels={"event_type": event_type, "matched": "false"},
            )
            logger.debug(
                "No trigger matched event_type=%s",
                event_type,
                extra={
                    "execution_data": {
                        "event": "trigger_no_match",
                        "event_type": event_type,
                        "registered_triggers": len(self._triggers),
                    }
                },
            )
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_prompt(template: str, payload: dict[str, Any]) -> str:
        """Render a prompt template with the event payload.

        Uses simple ``str.format_map`` substitution. If the template
        contains no format placeholders, the serialised payload is
        appended.
        """
        try:
            rendered = template.format_map(payload)
        except (KeyError, IndexError, ValueError):
            # Fallback: append the JSON payload
            rendered = f"{template}\n\nEvent payload:\n{json.dumps(payload)}"
        return rendered
