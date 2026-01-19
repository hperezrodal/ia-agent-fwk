"""Tests for streaming event models."""

from __future__ import annotations

import pytest

from ia_agent_fwk.streaming.models import AgentStreamEvent, StreamEvent


@pytest.mark.unit
class TestStreamEvent:
    def test_create_start_event(self):
        event = StreamEvent(event_type="start", data={"agent": "test"})
        assert event.event_type == "start"
        assert event.data == {"agent": "test"}
        assert event.timestamp  # non-empty ISO string

    def test_create_token_event(self):
        event = StreamEvent(event_type="token", data={"delta": "Hello"})
        assert event.event_type == "token"
        assert event.data["delta"] == "Hello"

    def test_create_complete_event(self):
        event = StreamEvent(event_type="complete", data={"output": "done"})
        assert event.event_type == "complete"

    def test_create_error_event(self):
        event = StreamEvent(event_type="error", data={"message": "fail"})
        assert event.event_type == "error"

    def test_frozen_model(self):
        event = StreamEvent(event_type="start")
        with pytest.raises(Exception):  # noqa: B017, PT011
            event.event_type = "token"  # type: ignore[misc]

    def test_default_data_is_empty(self):
        event = StreamEvent(event_type="start")
        assert event.data == {}

    def test_all_event_types(self):
        for etype in ("start", "token", "tool_call", "tool_result", "thinking", "complete", "error"):
            event = StreamEvent(event_type=etype)
            assert event.event_type == etype

    def test_serialization(self):
        event = StreamEvent(event_type="token", data={"delta": "hi"})
        d = event.model_dump(mode="json")
        assert d["event_type"] == "token"
        assert d["data"]["delta"] == "hi"
        assert "timestamp" in d


@pytest.mark.unit
class TestAgentStreamEvent:
    def test_create_start_event(self):
        event = AgentStreamEvent(event="start", agent_type="test")
        assert event.event == "start"
        assert event.agent_type == "test"
        assert event.content is None
        assert event.usage is None

    def test_create_complete_event_with_usage(self):
        event = AgentStreamEvent(
            event="complete",
            agent_type="test",
            content="Hello world",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        assert event.content == "Hello world"
        assert event.usage is not None
        assert event.usage["total_tokens"] == 30

    def test_create_error_event(self):
        event = AgentStreamEvent(
            event="error",
            agent_type="test",
            content="Something failed",
        )
        assert event.event == "error"
        assert event.content == "Something failed"

    def test_metadata_default_empty(self):
        event = AgentStreamEvent(event="start")
        assert event.metadata == {}

    def test_metadata_with_values(self):
        event = AgentStreamEvent(
            event="complete",
            metadata={"iterations": 3, "duration_ms": 100.0},
        )
        assert event.metadata["iterations"] == 3

    def test_frozen_model(self):
        event = AgentStreamEvent(event="start")
        with pytest.raises(Exception):  # noqa: B017, PT011
            event.event = "error"  # type: ignore[misc]

    def test_serialization_roundtrip(self):
        event = AgentStreamEvent(
            event="complete",
            agent_type="test",
            content="output",
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )
        d = event.model_dump(mode="json")
        restored = AgentStreamEvent.model_validate(d)
        assert restored.event == "complete"
        assert restored.content == "output"

    def test_timestamp_present(self):
        event = AgentStreamEvent(event="start")
        assert event.timestamp
        assert "T" in event.timestamp  # ISO format
