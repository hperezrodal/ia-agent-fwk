"""Tests for the ReasoningLoop."""

from __future__ import annotations

import asyncio
import logging

import pytest

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.context import AgentContext
from ia_agent_fwk.agents.exceptions import AgentMaxIterationsError
from ia_agent_fwk.agents.protocols import NoOpToolExecutor, ToolResult
from ia_agent_fwk.agents.reasoning import ReasoningLoop
from ia_agent_fwk.llm.exceptions import LLMProviderError
from ia_agent_fwk.llm.models import FinishReason, ToolCall

from .conftest import MockLLMProvider, make_chat_response


def _make_config(**kwargs) -> AgentConfig:
    defaults = {
        "name": "test",
        "agent_type": "test",
        "system_prompt": "You are a test agent.",
        "max_iterations": 10,
        "max_tokens_per_response": 4096,
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _make_context(config: AgentConfig | None = None) -> AgentContext:
    if config is None:
        config = _make_config()
    budget = config.context_window or 8192
    return AgentContext(
        system_prompt=config.system_prompt,
        token_budget=budget,
        token_counter=lambda text: len(text) // 4,
    )


def _make_event() -> asyncio.Event:
    event = asyncio.Event()
    event.set()  # Start resumed
    return event


class TestNoToolCompletion:
    @pytest.mark.asyncio
    async def test_single_iteration_stop(self):
        """AC-15: no-tool completion in 1 iteration."""
        provider = MockLLMProvider(
            responses=[
                make_chat_response(content="Done!", finish_reason=FinishReason.stop),
            ]
        )
        config = _make_config()
        context = _make_context(config)
        context.add_message(
            __import__("ia_agent_fwk.llm.models", fromlist=["Message"]).Message(role="user", content="Hello")
        )

        loop = ReasoningLoop()
        output, iterations, usage = await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert output == "Done!"
        assert iterations == 1
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20


class TestToolCallThenCompletion:
    @pytest.mark.asyncio
    async def test_two_iterations_with_tool(self):
        """AC-14: tool call on iter 1, stop on iter 2."""
        tool_call = ToolCall(id="tc-1", name="search", arguments='{"q": "test"}')
        provider = MockLLMProvider(
            responses=[
                make_chat_response(
                    content="Searching...",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tool_call],
                ),
                make_chat_response(
                    content="Found results.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )
        config = _make_config()
        context = _make_context(config)

        loop = ReasoningLoop()
        output, iterations, usage = await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert output == "Found results."
        assert iterations == 2
        assert usage.prompt_tokens == 20  # 10 + 10
        assert usage.completion_tokens == 40  # 20 + 20


class TestMaxIterations:
    @pytest.mark.asyncio
    async def test_exceeds_max_iterations(self):
        """AC-02: max iterations raises AgentMaxIterationsError."""
        tool_call = ToolCall(id="tc-1", name="loop_tool", arguments="{}")
        responses = [
            make_chat_response(
                content="Again...",
                finish_reason=FinishReason.tool_calls,
                tool_calls=[tool_call],
            )
            for _ in range(5)
        ]
        provider = MockLLMProvider(responses=responses)
        config = _make_config(max_iterations=3)
        context = _make_context(config)

        loop = ReasoningLoop()
        with pytest.raises(AgentMaxIterationsError):
            await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())


class TestFinishReasonLength:
    @pytest.mark.asyncio
    async def test_length_treated_as_completion(self, caplog):
        """finish_reason=length completes without retry, with warning."""
        provider = MockLLMProvider(
            responses=[
                make_chat_response(content="Truncated...", finish_reason=FinishReason.length),
            ]
        )
        config = _make_config()
        context = _make_context(config)

        loop = ReasoningLoop()
        with caplog.at_level(logging.WARNING):
            output, iterations, _ = await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert output == "Truncated..."
        assert iterations == 1
        assert "finish_reason=length" in caplog.text


class TestLLMErrorPropagation:
    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        """LLMProviderError propagates to caller."""
        provider = MockLLMProvider(
            responses=[make_chat_response()],
            error_on_call=0,
        )
        config = _make_config()
        context = _make_context(config)

        loop = ReasoningLoop()
        with pytest.raises(LLMProviderError):
            await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())


class TestTokenUsageAggregation:
    @pytest.mark.asyncio
    async def test_aggregated_across_iterations(self):
        """Token usage is summed across iterations."""
        tool_call = ToolCall(id="tc-1", name="tool", arguments="{}")
        provider = MockLLMProvider(
            responses=[
                make_chat_response(
                    content="Step 1",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tool_call],
                    prompt_tokens=100,
                    completion_tokens=50,
                ),
                make_chat_response(
                    content="Step 2",
                    finish_reason=FinishReason.stop,
                    prompt_tokens=200,
                    completion_tokens=75,
                ),
            ]
        )
        config = _make_config()
        context = _make_context(config)

        loop = ReasoningLoop()
        _, _, usage = await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 125
        assert usage.total_tokens == 425


class TestPauseResume:
    @pytest.mark.asyncio
    async def test_resume_event_controls_loop(self):
        """Event clear blocks; event set continues."""
        provider = MockLLMProvider(
            responses=[
                make_chat_response(content="Done!", finish_reason=FinishReason.stop),
            ]
        )
        config = _make_config()
        context = _make_context(config)
        event = asyncio.Event()

        loop = ReasoningLoop()

        # Start with event cleared -- loop should block
        task = asyncio.create_task(loop.run_loop(context, provider, NoOpToolExecutor(), config, event))

        # Give the task a chance to start and block
        await asyncio.sleep(0.05)
        assert not task.done()

        # Set the event -- loop should proceed
        event.set()
        output, _, _ = await asyncio.wait_for(task, timeout=2.0)
        assert output == "Done!"


class TestEmptyResponseHandling:
    @pytest.mark.asyncio
    async def test_none_content(self):
        """Loop handles ChatResponse with content=None gracefully."""
        provider = MockLLMProvider(
            responses=[
                make_chat_response(content="", finish_reason=FinishReason.stop),
            ]
        )
        config = _make_config()
        context = _make_context(config)

        loop = ReasoningLoop()
        output, iterations, _ = await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert output == ""
        assert iterations == 1


class TestCustomToolExecutor:
    @pytest.mark.asyncio
    async def test_tool_results_added_to_context(self):
        """Tool results are added as tool messages."""
        tool_call = ToolCall(id="tc-42", name="calculator", arguments='{"expr": "2+2"}')

        class CalculatorExecutor:
            async def execute(self, tc: ToolCall) -> ToolResult:
                return ToolResult(output="4", tool_call_id=tc.id)

        provider = MockLLMProvider(
            responses=[
                make_chat_response(
                    content="Let me calculate.",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tool_call],
                ),
                make_chat_response(
                    content="The answer is 4.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )
        config = _make_config()
        context = _make_context(config)

        loop = ReasoningLoop()
        output, _, _ = await loop.run_loop(context, provider, CalculatorExecutor(), config, _make_event())

        assert output == "The answer is 4."
        # Verify tool result was stored
        assert context.intermediate_results["tc-42"] == "4"


class TestMultipleToolCalls:
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_response(self):
        """F-012: multiple tool calls handled in a single iteration."""
        tc1 = ToolCall(id="tc-1", name="search", arguments='{"q": "a"}')
        tc2 = ToolCall(id="tc-2", name="lookup", arguments='{"q": "b"}')

        provider = MockLLMProvider(
            responses=[
                make_chat_response(
                    content="Let me look up both.",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[tc1, tc2],
                ),
                make_chat_response(
                    content="Both results found.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )
        config = _make_config()
        context = _make_context(config)

        loop = ReasoningLoop()
        output, iterations, _ = await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert output == "Both results found."
        assert iterations == 2
        assert "tc-1" in context.intermediate_results
        assert "tc-2" in context.intermediate_results


class TestContentFilterFinishReason:
    @pytest.mark.asyncio
    async def test_content_filter_continues_loop(self):
        """F-012: content_filter finish reason continues to next iteration."""
        provider = MockLLMProvider(
            responses=[
                make_chat_response(
                    content="Filtered",
                    finish_reason=FinishReason.content_filter,
                ),
                make_chat_response(
                    content="OK response.",
                    finish_reason=FinishReason.stop,
                ),
            ]
        )
        config = _make_config()
        context = _make_context(config)

        loop = ReasoningLoop()
        output, iterations, _ = await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert output == "OK response."
        assert iterations == 2


class TestPartialUsageOnError:
    @pytest.mark.asyncio
    async def test_partial_usage_preserved_on_max_iterations(self):
        """F-003: partial token usage available after AgentMaxIterationsError."""
        tool_call = ToolCall(id="tc-1", name="tool", arguments="{}")
        responses = [
            make_chat_response(
                content="Again...",
                finish_reason=FinishReason.tool_calls,
                tool_calls=[tool_call],
                prompt_tokens=50,
                completion_tokens=25,
            )
            for _ in range(5)
        ]
        provider = MockLLMProvider(responses=responses)
        config = _make_config(max_iterations=3)
        context = _make_context(config)

        loop = ReasoningLoop()
        with pytest.raises(AgentMaxIterationsError):
            await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        # Partial usage should reflect 3 completed iterations
        assert loop.partial_usage.prompt_tokens == 150  # 50 * 3
        assert loop.partial_usage.completion_tokens == 75  # 25 * 3
        assert loop.partial_iterations == 3


class TestModelPassthrough:
    @pytest.mark.asyncio
    async def test_model_passed_to_provider(self):
        """F-013: config.model is passed through to provider.chat()."""
        received_kwargs: list[dict[str, object]] = []

        class KwargsCapturingProvider(MockLLMProvider):
            async def chat(self, messages, **kwargs):
                received_kwargs.append(dict(kwargs))
                return await super().chat(messages, **kwargs)

        provider = KwargsCapturingProvider(
            responses=[
                make_chat_response(content="Done", finish_reason=FinishReason.stop),
            ]
        )
        config = _make_config(model="gpt-4-turbo")
        context = _make_context(config)

        loop = ReasoningLoop()
        await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert len(received_kwargs) == 1
        assert received_kwargs[0].get("model") == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_model_not_passed_when_none(self):
        """F-013: model kwarg not sent when config.model is None."""
        received_kwargs: list[dict[str, object]] = []

        class KwargsCapturingProvider(MockLLMProvider):
            async def chat(self, messages, **kwargs):
                received_kwargs.append(dict(kwargs))
                return await super().chat(messages, **kwargs)

        provider = KwargsCapturingProvider(
            responses=[
                make_chat_response(content="Done", finish_reason=FinishReason.stop),
            ]
        )
        config = _make_config(model=None)
        context = _make_context(config)

        loop = ReasoningLoop()
        await loop.run_loop(context, provider, NoOpToolExecutor(), config, _make_event())

        assert "model" not in received_kwargs[0]
