"""Tests for API request/response model validation."""

from __future__ import annotations

import pytest

from ia_agent_fwk.api.models import (
    AgentRunRequest,
    ConversationCreateRequest,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    TokenUsageResponse,
)


@pytest.mark.unit
class TestApiModels:
    def test_agent_run_request_valid(self):
        req = AgentRunRequest(prompt="Hello, world!")
        assert req.prompt == "Hello, world!"
        assert req.conversation_id is None
        assert req.options is None

    def test_agent_run_request_empty_prompt(self):
        with pytest.raises(Exception):  # noqa: B017
            AgentRunRequest(prompt="")

    def test_token_usage_response(self):
        usage = TokenUsageResponse(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_error_detail_model(self):
        detail = ErrorDetail(
            code="TEST_ERROR",
            message="Something went wrong",
            request_id="req-123",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert detail.code == "TEST_ERROR"
        assert detail.message == "Something went wrong"
        assert detail.detail is None

    def test_error_response_envelope(self):
        response = ErrorResponse(
            error=ErrorDetail(
                code="TEST_ERROR",
                message="Something went wrong",
                request_id="req-123",
                timestamp="2026-01-01T00:00:00Z",
            )
        )
        assert response.error.code == "TEST_ERROR"
        data = response.model_dump()
        assert "error" in data
        assert data["error"]["code"] == "TEST_ERROR"

    def test_health_response_model(self):
        resp = HealthResponse(status="healthy")
        assert resp.status == "healthy"

    def test_conversation_create_request(self):
        req = ConversationCreateRequest(agent_type="test-agent")
        assert req.agent_type == "test-agent"
        assert req.title is None

        req_with_title = ConversationCreateRequest(agent_type="test", title="My Chat")
        assert req_with_title.title == "My Chat"
