"""Structured audit logging for security-relevant events.

Uses a dedicated Python logger (``ia_agent_fwk.audit``) to emit structured
JSON records for authentication, agent execution, tool execution, and other
security-relevant events.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.observability.metrics import get_metrics_collector


class AuditEventType(str, Enum):
    """Supported audit event types."""

    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AGENT_EXECUTION = "agent_execution"
    TOOL_EXECUTION = "tool_execution"
    CONFIG_CHANGE = "config_change"
    RATE_LIMIT_HIT = "rate_limit_hit"


class AuditEvent(BaseModel):
    """Immutable structured audit event.

    Attributes
    ----------
    event_type:
        The type of security event.
    timestamp:
        ISO 8601 timestamp of the event.
    actor:
        Identifier of the actor (hashed API key, "anonymous", etc.).
    resource:
        The resource being accessed (endpoint path, agent type, tool name).
    action:
        The action performed (e.g. "authenticate", "execute", "read").
    result:
        Outcome of the action (e.g. "success", "failure", "denied").
    metadata:
        Additional structured data about the event.

    """

    model_config = ConfigDict(frozen=True)

    event_type: AuditEventType
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    )
    actor: str
    resource: str
    action: str
    result: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for use as an audit actor identifier.

    Uses SHA-256 and returns the first 16 hex characters to provide
    a consistent, non-reversible identifier.

    Parameters
    ----------
    api_key:
        The raw API key to hash.

    Returns
    -------
    str:
        A truncated SHA-256 hex digest (16 chars).

    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


class AuditLogger:
    """Logs audit events as structured JSON to a dedicated logger.

    Parameters
    ----------
    logger_name:
        Name of the Python logger to use. Defaults to ``ia_agent_fwk.audit``.

    """

    def __init__(self, logger_name: str = "ia_agent_fwk.audit") -> None:
        self._logger = logging.getLogger(logger_name)

    def log_event(self, event: AuditEvent) -> None:
        """Log a structured audit event.

        Events are logged at INFO level as JSON-serializable dictionaries.

        Parameters
        ----------
        event:
            The audit event to log.

        """
        collector = get_metrics_collector()
        collector.increment(
            "audit_events_total",
            labels={"event_type": event.event_type.value, "result": event.result},
        )

        data = event.model_dump()
        self._logger.info(
            "audit_event: %s",
            data,
            extra={"audit_event": data},
        )

    def log_auth_success(self, api_key: str, resource: str) -> None:
        """Log a successful authentication event."""
        self.log_event(
            AuditEvent(
                event_type=AuditEventType.AUTH_SUCCESS,
                actor=hash_api_key(api_key),
                resource=resource,
                action="authenticate",
                result="success",
            )
        )

    def log_auth_failure(self, resource: str, reason: str = "invalid_key") -> None:
        """Log a failed authentication event."""
        self.log_event(
            AuditEvent(
                event_type=AuditEventType.AUTH_FAILURE,
                actor="anonymous",
                resource=resource,
                action="authenticate",
                result="failure",
                metadata={"reason": reason},
            )
        )

    def log_agent_execution(
        self,
        api_key: str,
        agent_type: str,
        *,
        result: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an agent execution event."""
        self.log_event(
            AuditEvent(
                event_type=AuditEventType.AGENT_EXECUTION,
                actor=hash_api_key(api_key),
                resource=agent_type,
                action="execute",
                result=result,
                metadata=metadata or {},
            )
        )

    def log_tool_execution(
        self,
        api_key: str,
        tool_name: str,
        *,
        result: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a tool execution event."""
        self.log_event(
            AuditEvent(
                event_type=AuditEventType.TOOL_EXECUTION,
                actor=hash_api_key(api_key),
                resource=tool_name,
                action="execute",
                result=result,
                metadata=metadata or {},
            )
        )

    def log_rate_limit_hit(self, key: str, resource: str) -> None:
        """Log a rate limit hit event.

        Parameters
        ----------
        key:
            The rate-limit key (already hashed by the caller).
        resource:
            The resource path that was rate-limited.

        """
        self.log_event(
            AuditEvent(
                event_type=AuditEventType.RATE_LIMIT_HIT,
                actor=key,
                resource=resource,
                action="request",
                result="denied",
                metadata={"reason": "rate_limit_exceeded"},
            )
        )
