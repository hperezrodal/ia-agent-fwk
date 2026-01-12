"""Tests for structured audit logging."""

from __future__ import annotations

import logging

import pytest

from ia_agent_fwk.security.audit import (
    AuditEvent,
    AuditEventType,
    hash_api_key,
)


@pytest.mark.unit
class TestAuditEventModel:
    def test_audit_event_creation(self):
        event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            actor="abc123",
            resource="/api/v1/agents",
            action="authenticate",
            result="success",
        )
        assert event.event_type == AuditEventType.AUTH_SUCCESS
        assert event.actor == "abc123"
        assert event.resource == "/api/v1/agents"
        assert event.action == "authenticate"
        assert event.result == "success"
        assert event.metadata == {}
        assert event.timestamp  # auto-generated

    def test_audit_event_frozen(self):
        event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            actor="abc",
            resource="/test",
            action="read",
            result="success",
        )
        with pytest.raises(Exception):  # noqa: B017
            event.actor = "modified"  # type: ignore[misc]

    def test_audit_event_with_metadata(self):
        event = AuditEvent(
            event_type=AuditEventType.RATE_LIMIT_HIT,
            actor="key-hash",
            resource="/api/v1/agents",
            action="request",
            result="denied",
            metadata={"reason": "rate_limit_exceeded", "limit": 60},
        )
        assert event.metadata["reason"] == "rate_limit_exceeded"
        assert event.metadata["limit"] == 60

    def test_audit_event_model_dump(self):
        event = AuditEvent(
            event_type=AuditEventType.AUTH_FAILURE,
            actor="anonymous",
            resource="/api/v1/agents",
            action="authenticate",
            result="failure",
        )
        data = event.model_dump()
        assert data["event_type"] == "auth_failure"
        assert data["actor"] == "anonymous"
        assert "timestamp" in data

    def test_all_event_types_valid(self):
        expected = {
            "auth_success",
            "auth_failure",
            "agent_execution",
            "tool_execution",
            "config_change",
            "rate_limit_hit",
        }
        actual = {e.value for e in AuditEventType}
        assert actual == expected


@pytest.mark.unit
class TestHashApiKey:
    def test_hash_api_key_returns_hex(self):
        result = hash_api_key("test-api-key-123")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_api_key_deterministic(self):
        assert hash_api_key("my-key") == hash_api_key("my-key")

    def test_hash_api_key_different_keys(self):
        assert hash_api_key("key-1") != hash_api_key("key-2")


@pytest.mark.unit
class TestAuditLogger:
    def test_log_event_structured(self, audit_logger, caplog):
        event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            actor="abc123",
            resource="/api/v1/agents",
            action="authenticate",
            result="success",
        )
        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.audit"):
            audit_logger.log_event(event)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "audit_event" in record.message
        assert record.audit_event["event_type"] == "auth_success"
        assert record.audit_event["actor"] == "abc123"

    def test_auth_failure_logged(self, audit_logger, caplog):
        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.audit"):
            audit_logger.log_auth_failure(
                resource="/api/v1/agents",
                reason="missing_key",
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.audit_event["event_type"] == "auth_failure"
        assert record.audit_event["actor"] == "anonymous"
        assert record.audit_event["metadata"]["reason"] == "missing_key"

    def test_auth_success_logged(self, audit_logger, caplog):
        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.audit"):
            audit_logger.log_auth_success(
                api_key="test-key-abc",
                resource="/api/v1/agents",
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.audit_event["event_type"] == "auth_success"
        # Actor should be a hash, not the raw key
        assert record.audit_event["actor"] != "test-key-abc"
        assert len(record.audit_event["actor"]) == 16

    def test_agent_execution_logged(self, audit_logger, caplog):
        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.audit"):
            audit_logger.log_agent_execution(
                api_key="test-key",
                agent_type="customer_support",
                result="success",
                metadata={"iterations": 3, "duration_ms": 1500.0},
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.audit_event["event_type"] == "agent_execution"
        assert record.audit_event["resource"] == "customer_support"
        assert record.audit_event["metadata"]["iterations"] == 3

    def test_tool_execution_logged(self, audit_logger, caplog):
        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.audit"):
            audit_logger.log_tool_execution(
                api_key="test-key",
                tool_name="http_request",
                result="success",
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.audit_event["event_type"] == "tool_execution"
        assert record.audit_event["resource"] == "http_request"

    def test_rate_limit_hit_logged(self, audit_logger, caplog):
        with caplog.at_level(logging.INFO, logger="ia_agent_fwk.audit"):
            audit_logger.log_rate_limit_hit(
                key="hashed-key-abc",
                resource="/api/v1/agents/test/run",
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.audit_event["event_type"] == "rate_limit_hit"
        assert record.audit_event["result"] == "denied"
