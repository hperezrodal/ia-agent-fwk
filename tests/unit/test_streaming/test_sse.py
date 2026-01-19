"""Tests for SSE stream utility."""

from __future__ import annotations

import json

import pytest

from ia_agent_fwk.streaming.sse import _DEFAULT_HEARTBEAT_INTERVAL, sse_stream


@pytest.mark.unit
class TestSSEStream:
    async def test_sse_stream_format(self, stream_agent):
        """Each SSE chunk must be 'data: {json}\\n\\n'."""
        events = []
        async for chunk in sse_stream(stream_agent, "Hello"):
            events.append(chunk)

        assert len(events) == 2  # start + complete
        for chunk in events:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")
            # Payload between 'data: ' and '\n\n' must be valid JSON
            payload = chunk[6:-2]
            json.loads(payload)  # should not raise

    async def test_sse_stream_events_order(self, stream_agent):
        """Events must be: start, then complete."""
        events = []
        async for chunk in sse_stream(stream_agent, "test prompt"):
            payload = json.loads(chunk[6:-2])
            events.append(payload)

        assert len(events) == 2
        assert events[0]["event"] == "start"
        assert events[1]["event"] == "complete"

    async def test_sse_start_event_has_agent_type(self, stream_agent):
        events = []
        async for chunk in sse_stream(stream_agent, "test"):
            events.append(json.loads(chunk[6:-2]))

        assert events[0]["agent_type"] == "stream_test"

    async def test_sse_complete_event_has_output(self, stream_agent):
        events = []
        async for chunk in sse_stream(stream_agent, "my input"):
            events.append(json.loads(chunk[6:-2]))

        complete = events[1]
        assert complete["event"] == "complete"
        assert complete["content"] == "Streamed: my input"

    async def test_sse_complete_event_has_usage(self, stream_agent):
        events = []
        async for chunk in sse_stream(stream_agent, "hello"):
            events.append(json.loads(chunk[6:-2]))

        usage = events[1]["usage"]
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == 15
        assert usage["total_tokens"] == 20

    async def test_sse_stream_error_event(self, error_agent):
        """When agent raises, we should get start + error events."""
        events = []
        async for chunk in sse_stream(error_agent, "fail"):
            events.append(json.loads(chunk[6:-2]))

        assert len(events) == 2
        assert events[0]["event"] == "start"
        assert events[1]["event"] == "error"
        assert "Intentional test error" in events[1]["content"]

    async def test_sse_stream_with_conversation_id(self, stream_agent):
        events = []
        async for chunk in sse_stream(
            stream_agent,
            "hello",
            conversation_id="conv-123",
        ):
            events.append(json.loads(chunk[6:-2]))

        start = events[0]
        assert start["metadata"]["conversation_id"] == "conv-123"
        complete = events[1]
        assert complete["metadata"]["conversation_id"] == "conv-123"

    async def test_sse_stream_without_conversation_id(self, stream_agent):
        events = []
        async for chunk in sse_stream(stream_agent, "hello"):
            events.append(json.loads(chunk[6:-2]))

        start = events[0]
        assert start["metadata"] == {}

    async def test_sse_complete_event_has_metadata(self, stream_agent):
        events = []
        async for chunk in sse_stream(stream_agent, "hello"):
            events.append(json.loads(chunk[6:-2]))

        meta = events[1]["metadata"]
        assert "iterations" in meta
        assert "duration_ms" in meta

    async def test_sse_heartbeat_emitted_for_slow_agent(self, slow_agent):
        """Heartbeat events are emitted while the agent runs longer than the interval."""
        events = []
        async for chunk in sse_stream(slow_agent, "hello", heartbeat_interval=0.05):
            events.append(json.loads(chunk[6:-2]))

        event_types = [e["event"] for e in events]
        assert event_types[0] == "start"
        assert "heartbeat" in event_types
        assert event_types[-1] == "complete"

    async def test_sse_no_heartbeat_when_disabled(self, slow_agent):
        """No heartbeat events when heartbeat_interval=0."""
        events = []
        async for chunk in sse_stream(slow_agent, "hello", heartbeat_interval=0):
            events.append(json.loads(chunk[6:-2]))

        event_types = [e["event"] for e in events]
        assert "heartbeat" not in event_types
        assert event_types == ["start", "complete"]

    async def test_sse_default_heartbeat_interval(self):
        """Default heartbeat interval is 15 seconds."""
        assert _DEFAULT_HEARTBEAT_INTERVAL == 15.0
