"""Tests for WebSocket handler."""

from __future__ import annotations

import json

import pytest

from ia_agent_fwk.config.settings import AppSettings, AuthSettings
from ia_agent_fwk.streaming.websocket import WebSocketHandler, get_active_connections

from .conftest import ErrorTestAgent, StreamTestAgent, _MockLLMProvider


def _make_handler(auth_enabled=True, monkeypatch=None):  # noqa: ARG001, FBT002
    """Create a WebSocketHandler with mock settings."""
    settings = AppSettings(auth=AuthSettings(enabled=auth_enabled))
    mock_provider = _MockLLMProvider()

    from ia_agent_fwk.agents.config import AgentConfig

    def agent_factory(agent_type):
        if agent_type == "stream_test":
            config = AgentConfig(name="ws-test", agent_type="stream_test")
            return StreamTestAgent(config=config, provider=mock_provider)
        if agent_type == "error_test":
            config = AgentConfig(name="ws-error", agent_type="error_test")
            return ErrorTestAgent(config=config, provider=mock_provider)
        msg = f"Unknown agent type: {agent_type}"
        raise ValueError(msg)

    return WebSocketHandler(settings=settings, agent_factory=agent_factory)


class _FakeWebSocket:
    """Minimal fake WebSocket for unit tests."""

    def __init__(self, messages=None, query_params=None):
        self._messages = list(messages or [])
        self._sent = []
        self._accepted = False
        self._closed = False
        self._close_code = None
        self.query_params = query_params or {}
        self._msg_index = 0

    async def accept(self):
        self._accepted = True

    async def receive_text(self):
        if self._msg_index >= len(self._messages):
            # Simulate disconnect
            msg = "WebSocket disconnected"
            raise RuntimeError(msg)
        msg = self._messages[self._msg_index]
        self._msg_index += 1
        return msg

    async def send_text(self, data):
        self._sent.append(data)

    async def close(self, code=1000):
        self._closed = True
        self._close_code = code


@pytest.mark.unit
class TestWebSocketHandler:
    async def test_websocket_connect_and_receive(self, monkeypatch):
        """WebSocket handler accepts, auths, processes prompt, and sends events."""
        monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
        handler = _make_handler(auth_enabled=True)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"prompt": "Hello", "agent_type": "stream_test"}),
            ],
            query_params={"api_key": "test-key-1"},
        )

        await handler.handle(ws)

        assert ws._accepted
        # Should have sent start + complete events
        assert len(ws._sent) >= 2
        events = [json.loads(s) for s in ws._sent]
        assert events[0]["event"] == "start"
        assert events[1]["event"] == "complete"
        assert events[1]["content"] == "Streamed: Hello"

    async def test_websocket_auth_required(self, monkeypatch):
        """WebSocket closes with 4001 when auth fails."""
        monkeypatch.setenv("IAFWK_API_KEYS", "valid-key")
        handler = _make_handler(auth_enabled=True)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"api_key": "wrong-key"}),
            ],
            query_params={},
        )

        await handler.handle(ws)

        assert ws._closed
        assert ws._close_code == 4001
        # Should have sent an error message
        assert len(ws._sent) >= 1
        err = json.loads(ws._sent[0])
        assert err["event"] == "error"
        assert "Authentication" in err["content"]

    async def test_websocket_auth_via_query_param(self, monkeypatch):
        """WebSocket authenticates via query param."""
        monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
        handler = _make_handler(auth_enabled=True)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"prompt": "Hi", "agent_type": "stream_test"}),
            ],
            query_params={"api_key": "test-key-1"},
        )

        await handler.handle(ws)

        assert ws._accepted
        events = [json.loads(s) for s in ws._sent]
        assert any(e["event"] == "complete" for e in events)

    async def test_websocket_auth_via_first_message(self, monkeypatch):
        """WebSocket authenticates via api_key in first message."""
        monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
        handler = _make_handler(auth_enabled=True)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"api_key": "test-key-1"}),
                json.dumps({"prompt": "Hello", "agent_type": "stream_test"}),
            ],
            query_params={},
        )

        await handler.handle(ws)

        assert ws._accepted
        events = [json.loads(s) for s in ws._sent]
        assert any(e["event"] == "complete" for e in events)

    async def test_websocket_auth_disabled(self, monkeypatch):
        """WebSocket skips auth when auth is disabled."""
        handler = _make_handler(auth_enabled=False)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"prompt": "Hello", "agent_type": "stream_test"}),
            ],
        )

        await handler.handle(ws)

        assert ws._accepted
        events = [json.loads(s) for s in ws._sent]
        assert any(e["event"] == "complete" for e in events)

    async def test_websocket_multi_turn(self, monkeypatch):
        """WebSocket supports multiple prompts in one connection."""
        monkeypatch.setenv("IAFWK_API_KEYS", "test-key-1")
        handler = _make_handler(auth_enabled=True)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"prompt": "First", "agent_type": "stream_test"}),
                json.dumps({"prompt": "Second", "agent_type": "stream_test"}),
            ],
            query_params={"api_key": "test-key-1"},
        )

        await handler.handle(ws)

        events = [json.loads(s) for s in ws._sent]
        # Should have 4 events: start+complete for each prompt
        complete_events = [e for e in events if e["event"] == "complete"]
        assert len(complete_events) == 2
        assert complete_events[0]["content"] == "Streamed: First"
        assert complete_events[1]["content"] == "Streamed: Second"

    async def test_websocket_missing_prompt(self, monkeypatch):
        """WebSocket returns error when prompt is missing."""
        handler = _make_handler(auth_enabled=False)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"agent_type": "stream_test"}),
            ],
        )

        await handler.handle(ws)

        events = [json.loads(s) for s in ws._sent]
        assert any("Missing 'prompt'" in e.get("content", "") for e in events)

    async def test_websocket_missing_agent_type(self, monkeypatch):
        """WebSocket returns error when agent_type is missing."""
        handler = _make_handler(auth_enabled=False)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"prompt": "Hello"}),
            ],
        )

        await handler.handle(ws)

        events = [json.loads(s) for s in ws._sent]
        assert any("Missing 'agent_type'" in e.get("content", "") for e in events)

    async def test_websocket_invalid_json(self, monkeypatch):
        """WebSocket returns error for invalid JSON."""
        handler = _make_handler(auth_enabled=False)
        ws = _FakeWebSocket(
            messages=["not-json"],
        )

        await handler.handle(ws)

        events = [json.loads(s) for s in ws._sent]
        assert any("Invalid JSON" in e.get("content", "") for e in events)

    async def test_websocket_agent_error(self, monkeypatch):
        """WebSocket returns error event when agent raises."""
        handler = _make_handler(auth_enabled=False)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"prompt": "fail", "agent_type": "error_test"}),
            ],
        )

        await handler.handle(ws)

        events = [json.loads(s) for s in ws._sent]
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert "Intentional test error" in error_events[0]["content"]

    async def test_websocket_connection_limit(self, monkeypatch):
        """WebSocket rejects connections when max_connections is reached."""
        import ia_agent_fwk.streaming.websocket as ws_mod

        handler = _make_handler(auth_enabled=False)
        handler._max_connections = 0  # Set limit to 0 to force rejection

        ws = _FakeWebSocket(messages=[])
        old_count = ws_mod._active_connections
        try:
            await handler.handle(ws)
        finally:
            ws_mod._active_connections = old_count

        assert ws._accepted
        assert ws._closed
        events = [json.loads(s) for s in ws._sent]
        assert any("Too many connections" in e.get("content", "") for e in events)

    async def test_websocket_active_connections_tracked(self, monkeypatch):
        """Active connections counter increments/decrements correctly."""
        import ia_agent_fwk.streaming.websocket as ws_mod

        handler = _make_handler(auth_enabled=False)
        ws = _FakeWebSocket(
            messages=[
                json.dumps({"prompt": "Hello", "agent_type": "stream_test"}),
            ],
        )

        before = ws_mod._active_connections
        await handler.handle(ws)
        after = ws_mod._active_connections

        # Should be back to original count after handle() completes
        assert after == before

    async def test_get_active_connections(self):
        """get_active_connections returns current count."""
        count = get_active_connections()
        assert isinstance(count, int)
        assert count >= 0
