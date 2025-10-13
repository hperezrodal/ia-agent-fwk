"""Tests for the canonical Pydantic v2 LLM models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ia_agent_fwk.llm.models import (
    ChatResponse,
    CompletionResponse,
    FinishReason,
    HealthStatus,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)


class TestMessage:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant", "tool"):
            m = Message(role=role, content="hi")
            assert m.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(role="invalid", content="x")

    def test_optional_fields(self):
        m = Message(role="user", content="hi")
        assert m.tool_calls is None
        assert m.tool_call_id is None
        assert m.metadata is None

    def test_with_metadata(self):
        m = Message(role="user", content="hi", metadata={"trace_id": "abc123"})
        assert m.metadata == {"trace_id": "abc123"}

    def test_with_tool_calls(self):
        tc = ToolCall(id="1", name="func", arguments='{"a": 1}')
        m = Message(role="assistant", content="ok", tool_calls=[tc])
        assert m.tool_calls is not None
        assert len(m.tool_calls) == 1

    def test_json_round_trip(self):
        m = Message(role="user", content="hello")
        data = m.model_dump_json()
        restored = Message.model_validate_json(data)
        assert restored == m


class TestToolCall:
    def test_parse_arguments(self):
        tc = ToolCall(id="1", name="func", arguments='{"a": 1, "b": "x"}')
        result = tc.parse_arguments()
        assert result == {"a": 1, "b": "x"}

    def test_parse_arguments_empty_object(self):
        tc = ToolCall(id="2", name="func", arguments="{}")
        assert tc.parse_arguments() == {}

    def test_frozen(self):
        tc = ToolCall(id="1", name="func", arguments="{}")
        with pytest.raises(ValidationError):
            tc.id = "2"  # type: ignore[misc]


class TestFinishReason:
    def test_enum_values(self):
        expected = {"stop", "tool_calls", "length", "content_filter", "error"}
        actual = {e.value for e in FinishReason}
        assert actual == expected

    def test_str_comparison(self):
        assert FinishReason.stop == "stop"


class TestTokenUsage:
    def test_auto_compute_total(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
        assert usage.total_tokens == 30

    def test_explicit_total(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=50)
        assert usage.total_tokens == 50

    def test_zero_defaults(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_json_round_trip(self):
        usage = TokenUsage(prompt_tokens=5, completion_tokens=10)
        data = usage.model_dump_json()
        restored = TokenUsage.model_validate_json(data)
        assert restored.total_tokens == 15


class TestChatResponse:
    def test_construction(self, sample_chat_response):
        assert sample_chat_response.message.role == "assistant"
        assert sample_chat_response.finish_reason == FinishReason.stop

    def test_json_round_trip(self, sample_chat_response):
        data = sample_chat_response.model_dump_json()
        restored = ChatResponse.model_validate_json(data)
        assert restored.model == sample_chat_response.model


class TestCompletionResponse:
    def test_construction(self):
        r = CompletionResponse(
            text="hello",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2),
            model="m",
        )
        assert r.text == "hello"
        assert r.finish_reason == FinishReason.stop


class TestStreamChunk:
    def test_defaults(self):
        c = StreamChunk()
        assert c.delta == ""
        assert c.finish_reason is None
        assert c.usage is None

    def test_with_usage(self):
        c = StreamChunk(
            delta="hi",
            finish_reason=FinishReason.stop,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )
        assert c.usage is not None
        assert c.usage.total_tokens == 2


class TestHealthStatus:
    def test_healthy(self):
        h = HealthStatus(status="healthy", latency_ms=42.0)
        assert h.status == "healthy"

    def test_unhealthy_with_message(self):
        h = HealthStatus(status="unhealthy", message="Connection refused")
        assert h.message == "Connection refused"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            HealthStatus(status="unknown")
