"""Tests for the Agent base class lifecycle and state management."""

from __future__ import annotations

import asyncio

import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig, AgentResult
from ia_agent_fwk.agents.exceptions import AgentError, InvalidStateTransitionError
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.models import FinishReason, ToolCall

from .conftest import MockLLMProvider, make_chat_response


class ConcreteTestAgent(Agent):
    """Minimal concrete Agent subclass for testing."""

    @property
    def agent_type(self) -> str:
        return "test"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks_called: list[str] = []

    async def on_start(self):
        self.hooks_called.append("on_start")

    async def on_complete(self, result):
        self.hooks_called.append("on_complete")

    async def on_error(self, error):
        self.hooks_called.append("on_error")

    async def on_timeout(self):
        self.hooks_called.append("on_timeout")


def _make_config(**kwargs) -> AgentConfig:
    defaults = {
        "name": "test-agent",
        "agent_type": "test",
        "system_prompt": "You are a test agent.",
        "max_iterations": 10,
        "execution_timeout": 300,
        "max_tokens_per_response": 4096,
        "context_window": 8192,
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


class TestAgentABC:
    def test_cannot_instantiate_directly(self):
        """Agent is abstract and cannot be instantiated directly."""
        config = _make_config()
        provider = MockLLMProvider(responses=[make_chat_response()])
        with pytest.raises(TypeError):
            Agent(config=config, provider=provider)  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        config = _make_config()
        provider = MockLLMProvider(responses=[make_chat_response()])
        agent = ConcreteTestAgent(config=config, provider=provider)
        assert agent.state == AgentState.IDLE


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_run_completes(self):
        """AC-01: run() transitions IDLE->RUNNING->COMPLETED."""
        config = _make_config()
        provider = MockLLMProvider(
            responses=[
                make_chat_response(content="Hello!", finish_reason=FinishReason.stop),
            ]
        )
        agent = ConcreteTestAgent(config=config, provider=provider)

        result = await agent.run("Hello")

        assert result.state == AgentState.COMPLETED
        assert result.output == "Hello!"
        assert result.iterations == 1
        assert result.duration_ms > 0
        assert result.error is None
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_result_has_usage(self):
        config = _make_config()
        provider = MockLLMProvider(
            responses=[
                make_chat_response(prompt_tokens=100, completion_tokens=50),
            ]
        )
        agent = ConcreteTestAgent(config=config, provider=provider)
        result = await agent.run("Test")

        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150


class TestHookOrder:
    @pytest.mark.asyncio
    async def test_success_hooks(self):
        """AC-18: on_start called first, then on_complete."""
        config = _make_config()
        provider = MockLLMProvider(responses=[make_chat_response()])
        agent = ConcreteTestAgent(config=config, provider=provider)

        await agent.run("Hello")

        assert agent.hooks_called == ["on_start", "on_complete"]

    @pytest.mark.asyncio
    async def test_error_hooks(self):
        """AC-19: on_start called, then on_error. on_complete NOT called."""
        config = _make_config()
        provider = MockLLMProvider(
            responses=[make_chat_response()],
            error_on_call=0,
        )
        agent = ConcreteTestAgent(config=config, provider=provider)

        result = await agent.run("Hello")

        assert result.state == AgentState.FAILED
        assert "on_start" in agent.hooks_called
        assert "on_error" in agent.hooks_called
        assert "on_complete" not in agent.hooks_called


class TestTimeout:
    @pytest.mark.asyncio
    async def test_execution_timeout(self):
        """AC-03: timeout transitions to FAILED, on_timeout and on_error called."""
        config = _make_config(execution_timeout=1)
        provider = MockLLMProvider(responses=[make_chat_response()], delay=5.0)
        agent = ConcreteTestAgent(config=config, provider=provider)

        result = await agent.run("Hello")

        assert result.state == AgentState.FAILED
        assert result.error is not None
        assert "timed out" in result.error
        assert "on_timeout" in agent.hooks_called
        assert "on_error" in agent.hooks_called
        assert agent.state == AgentState.FAILED


class TestMaxIterations:
    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self):
        """AC-02: max iterations transitions to FAILED."""
        tool_call = ToolCall(id="tc-1", name="tool", arguments="{}")
        responses = [
            make_chat_response(
                content="Again",
                finish_reason=FinishReason.tool_calls,
                tool_calls=[tool_call],
            )
            for _ in range(5)
        ]
        config = _make_config(max_iterations=3)
        provider = MockLLMProvider(responses=responses)
        agent = ConcreteTestAgent(config=config, provider=provider)

        result = await agent.run("Hello")

        assert result.state == AgentState.FAILED
        assert result.error is not None
        assert "max_iterations" in result.error
        assert agent.state == AgentState.FAILED
        assert "on_error" in agent.hooks_called


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_transitions_to_failed(self):
        """AC-12: stop() transitions to FAILED and cancels the running task."""
        config = _make_config()
        provider = MockLLMProvider(responses=[make_chat_response()], delay=5.0)
        agent = ConcreteTestAgent(config=config, provider=provider)

        # Start the agent in a task
        task = asyncio.create_task(agent.run("Hello"))
        await asyncio.sleep(0.05)

        # Stop while it's waiting for the slow LLM
        await agent.stop()

        assert agent.state == AgentState.FAILED
        assert "on_error" in agent.hooks_called

        # The task should complete (stop() cancels it internally)
        result = await asyncio.wait_for(task, timeout=2.0)
        assert result.state == AgentState.FAILED

    @pytest.mark.asyncio
    async def test_stop_uses_agent_error(self):
        """F-010: stop() uses AgentError, not AgentMaxIterationsError."""
        config = _make_config()
        provider = MockLLMProvider(responses=[make_chat_response()], delay=5.0)

        captured_errors: list[Exception] = []

        class ErrorCapturingAgent(Agent):
            @property
            def agent_type(self) -> str:
                return "test"

            async def on_error(self, error: Exception) -> None:
                captured_errors.append(error)

        agent = ErrorCapturingAgent(config=config, provider=provider)
        task = asyncio.create_task(agent.run("Hello"))
        await asyncio.sleep(0.05)
        await agent.stop()
        await asyncio.wait_for(task, timeout=2.0)

        assert len(captured_errors) == 1
        assert isinstance(captured_errors[0], AgentError)
        assert "stopped by user" in str(captured_errors[0])


class TestPauseResume:
    @pytest.mark.asyncio
    async def test_pause_and_resume(self):
        """AC-13: pause -> WAITING_FOR_INPUT -> resume -> RUNNING."""
        config = _make_config()
        provider = MockLLMProvider(
            responses=[
                make_chat_response(content="First", finish_reason=FinishReason.stop),
            ]
        )
        agent = ConcreteTestAgent(config=config, provider=provider)

        # Start, then immediately pause before the loop runs
        agent._transition_to(AgentState.RUNNING)
        agent.pause()

        assert agent.state == AgentState.WAITING_FOR_INPUT

        agent.resume("new input")

        assert agent.state == AgentState.RUNNING


class TestPauseResumeDuringRun:
    @pytest.mark.asyncio
    async def test_pause_resume_during_run(self):
        """F-011: full lifecycle pause/resume with running agent.

        Uses a provider that blocks on the *second* chat call via an
        asyncio.Event, giving us a deterministic window to pause the
        agent. While the second call is blocked, we pause. Then we
        unblock the provider and resume the agent. The loop's event
        check at the top of iteration 3 will block until resume.
        """
        gate = asyncio.Event()  # Controls when second call returns

        # 3 iterations: tool_call -> tool_call -> stop
        provider = MockLLMProvider(
            responses=[
                make_chat_response(
                    content="Step 1",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[ToolCall(id="tc-1", name="tool", arguments="{}")],
                ),
                make_chat_response(
                    content="Step 2",
                    finish_reason=FinishReason.tool_calls,
                    tool_calls=[ToolCall(id="tc-2", name="tool", arguments="{}")],
                ),
                make_chat_response(content="Final answer!", finish_reason=FinishReason.stop),
            ]
        )

        original_chat = provider.chat
        call_count = 0

        async def gated_chat(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Block second call until gate is opened
                await gate.wait()
            return await original_chat(messages, **kwargs)

        provider.chat = gated_chat  # type: ignore[assignment]

        config = _make_config(max_iterations=5)
        agent = ConcreteTestAgent(config=config, provider=provider)

        # Start agent
        task = asyncio.create_task(agent.run("Hello"))
        # Wait for second call to block
        for _ in range(50):
            await asyncio.sleep(0.01)
            if call_count >= 2:
                break

        assert call_count == 2
        assert agent.state == AgentState.RUNNING

        # Pause while second call is blocked. The resume_event is now
        # cleared, so when the loop reaches iteration 3's event check
        # it will block.
        agent.pause()
        assert agent.state == AgentState.WAITING_FOR_INPUT

        # Unblock the second provider call (iteration 2 continues)
        gate.set()
        # Yield so iteration 2 finishes; iteration 3 will block on event
        await asyncio.sleep(0.05)
        assert not task.done()

        # Resume with new input
        agent.resume("new user message")
        assert agent.state == AgentState.RUNNING

        result = await asyncio.wait_for(task, timeout=2.0)
        assert result.state == AgentState.COMPLETED
        assert result.output == "Final answer!"


class TestNoToolMode:
    @pytest.mark.asyncio
    async def test_no_tools_completes(self):
        """AC-15: agent with no tools runs and completes."""
        config = _make_config(tools=[])
        provider = MockLLMProvider(
            responses=[
                make_chat_response(content="No tools needed!", finish_reason=FinishReason.stop),
            ]
        )
        agent = ConcreteTestAgent(config=config, provider=provider)

        result = await agent.run("Hello")

        assert result.state == AgentState.COMPLETED
        assert result.output == "No tools needed!"


class TestInvalidState:
    @pytest.mark.asyncio
    async def test_run_on_non_idle_raises(self):
        """Calling run() on a non-IDLE agent raises InvalidStateTransitionError."""
        config = _make_config()
        provider = MockLLMProvider(responses=[make_chat_response()])
        agent = ConcreteTestAgent(config=config, provider=provider)

        # First run succeeds
        await agent.run("Hello")

        # Second run should fail (agent is now COMPLETED)
        with pytest.raises(InvalidStateTransitionError):
            await agent.run("Hello again")


class TestAgentResultFields:
    @pytest.mark.asyncio
    async def test_all_fields_populated(self):
        config = _make_config()
        provider = MockLLMProvider(
            responses=[
                make_chat_response(
                    content="Result",
                    finish_reason=FinishReason.stop,
                    prompt_tokens=50,
                    completion_tokens=25,
                ),
            ]
        )
        agent = ConcreteTestAgent(config=config, provider=provider)

        result = await agent.run("Input")

        assert isinstance(result, AgentResult)
        assert result.output == "Result"
        assert result.state == AgentState.COMPLETED
        assert result.usage.prompt_tokens == 50
        assert result.usage.completion_tokens == 25
        assert result.iterations == 1
        assert result.duration_ms > 0
        assert result.error is None
