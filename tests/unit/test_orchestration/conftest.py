"""Shared fixtures for orchestration tests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig, AgentResult
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.base import LLMProvider
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
from ia_agent_fwk.orchestration.models import WorkflowStep

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# MockLLMProvider for orchestration tests
# ---------------------------------------------------------------------------


class OrcMockLLMProvider(LLMProvider):
    """Lightweight mock LLM provider for orchestration tests."""

    def __init__(
        self,
        responses: list[ChatResponse] | None = None,
        *,
        delay: float = 0.0,
    ) -> None:
        self.responses = list(responses or [])
        self._call_index = 0
        self._delay = delay
        self.provider_name = "mock"

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._call_index >= len(self.responses):
            msg = f"No response configured for call index {self._call_index}"
            raise IndexError(msg)
        response = self.responses[self._call_index]
        self._call_index += 1
        return response

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError
        yield  # type: ignore[misc]

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text) // 4

    async def health_check(self) -> HealthStatus:
        return HealthStatus(status="healthy")

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# MockAgent
# ---------------------------------------------------------------------------


class MockAgent(Agent):
    """Test agent with configurable output."""

    _mock_output: str
    _should_fail: bool
    _delay: float

    def __init__(  # noqa: PLR0913
        self,
        config: AgentConfig,
        provider: LLMProvider,
        tool_executor=None,
        *,
        output: str = "mock output",
        should_fail: bool = False,
        delay: float = 0.0,
    ) -> None:
        super().__init__(config=config, provider=provider, tool_executor=tool_executor)
        self._mock_output = output
        self._should_fail = should_fail
        self._delay = delay

    @property
    def agent_type(self) -> str:
        return "mock"

    async def run(self, input_text: str, conversation_history=None, conversation_id=None) -> AgentResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._should_fail:
            return AgentResult(
                output="",
                state=AgentState.FAILED,
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
                iterations=0,
                duration_ms=0.0,
                error="Mock agent failure",
            )

        return AgentResult(
            output=self._mock_output,
            state=AgentState.COMPLETED,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            iterations=1,
            duration_ms=100.0,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chat_response(
    content: str = "Hello!",
    finish_reason: FinishReason = FinishReason.stop,
    tool_calls: list[ToolCall] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> ChatResponse:
    return ChatResponse(
        message=Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        model="mock-model",
        finish_reason=finish_reason,
    )


@pytest.fixture
def mock_provider() -> OrcMockLLMProvider:
    return OrcMockLLMProvider(responses=[make_chat_response()])


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    return AgentConfig(
        name="test-agent",
        agent_type="mock",
        system_prompt="You are a test agent.",
        provider_name="mock",
        max_iterations=3,
        execution_timeout=30,
    )


@pytest.fixture
def simple_workflow_steps() -> list[WorkflowStep]:
    return [
        WorkflowStep(name="step_1", agent_name="mock"),
        WorkflowStep(name="step_2", agent_name="mock"),
        WorkflowStep(name="step_3", agent_name="mock"),
    ]


@pytest.fixture
def mock_agent_factory(mock_provider):
    """Factory that creates MockAgents with configurable behavior.

    Returns a factory function. The factory tracks calls and supports
    per-agent output configuration via _outputs dict.
    """
    _outputs: dict[str, str] = {}
    _failures: set[str] = set()
    _call_count = [0]

    def factory(config: AgentConfig) -> MockAgent:
        _call_count[0] += 1
        output = _outputs.get(config.name, f"output from {config.name}")
        should_fail = config.name in _failures
        return MockAgent(
            config=config,
            provider=mock_provider,
            output=output,
            should_fail=should_fail,
        )

    factory._outputs = _outputs  # type: ignore[attr-defined]
    factory._failures = _failures  # type: ignore[attr-defined]
    factory._call_count = _call_count  # type: ignore[attr-defined]
    return factory
