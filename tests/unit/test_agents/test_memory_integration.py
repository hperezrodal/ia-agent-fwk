"""Tests for memory integration in agent execution loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig, AgentMemoryConfig
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.memory.models import ConversationMessage, MemoryResult

from .conftest import MockLLMProvider, make_chat_response

# ---------------------------------------------------------------------------
# Concrete test agent (minimal subclass)
# ---------------------------------------------------------------------------


class _MemoryTestAgent(Agent):
    @property
    def agent_type(self) -> str:
        return "memory_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> AgentConfig:
    defaults = {
        "name": "mem-test",
        "agent_type": "memory_test",
        "system_prompt": "You are a test agent.",
        "provider_name": "mock",
        "max_iterations": 3,
        "execution_timeout": 10,
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _make_memory_backend(search_results: list[MemoryResult] | None = None):
    backend = AsyncMock()
    backend.backend_type = "mock"
    backend.search = AsyncMock(return_value=search_results or [])
    backend.store = AsyncMock()
    return backend


def _make_conversation_backend(messages: list[ConversationMessage] | None = None):
    backend = AsyncMock()
    backend.get_messages = AsyncMock(return_value=messages or [])
    backend.add_message = AsyncMock()
    backend.create_conversation = AsyncMock(
        return_value=MagicMock(conversation_id="conv-123"),
    )
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentMemoryIntegration:
    """Test memory integration in Agent.run()."""

    async def test_run_without_memory_backends(self):
        """Agent works as before when no memory backends are provided."""
        provider = MockLLMProvider(responses=[make_chat_response("Hello!")])
        agent = _MemoryTestAgent(config=_make_config(), provider=provider)
        result = await agent.run("Hi")
        assert result.state == AgentState.COMPLETED
        assert result.output == "Hello!"

    async def test_run_loads_conversation_history(self):
        """Agent loads history from conversation_backend when conversation_id is set."""
        conv_backend = _make_conversation_backend(
            messages=[
                ConversationMessage(
                    id="m1",
                    conversation_id="conv-1",
                    role="user",
                    content="Prior question",
                ),
                ConversationMessage(
                    id="m2",
                    conversation_id="conv-1",
                    role="assistant",
                    content="Prior answer",
                ),
            ]
        )
        provider = MockLLMProvider(responses=[make_chat_response("Follow-up!")])
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            conversation_backend=conv_backend,
        )
        result = await agent.run("New question", conversation_id="conv-1")
        assert result.state == AgentState.COMPLETED
        conv_backend.get_messages.assert_called_once_with("conv-1")

    async def test_run_persists_turn_on_success(self):
        """After successful run, user and assistant messages are stored."""
        conv_backend = _make_conversation_backend()
        mem_backend = _make_memory_backend()
        provider = MockLLMProvider(responses=[make_chat_response("Answer!")])
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            memory_backend=mem_backend,
            conversation_backend=conv_backend,
        )
        result = await agent.run("Question?", conversation_id="conv-42")
        assert result.state == AgentState.COMPLETED

        # Conversation backend: user + assistant messages persisted
        assert conv_backend.add_message.call_count == 2
        calls = conv_backend.add_message.call_args_list
        assert calls[0].kwargs["role"] == "user"
        assert calls[0].kwargs["content"] == "Question?"
        assert calls[1].kwargs["role"] == "assistant"
        assert calls[1].kwargs["content"] == "Answer!"

        # Vector memory: turn stored
        mem_backend.store.assert_called_once()
        store_kwargs = mem_backend.store.call_args.kwargs
        assert "Question?" in store_kwargs["value"]
        assert "Answer!" in store_kwargs["value"]

    async def test_run_semantic_search_injects_context(self):
        """Agent searches vector memory and injects results into context."""
        mem_backend = _make_memory_backend(
            search_results=[
                MemoryResult(key="k1", value="Previous important fact", score=0.9),
                MemoryResult(key="k2", value="Another relevant detail", score=0.7),
            ]
        )
        provider = MockLLMProvider(responses=[make_chat_response("Used memory!")])
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            memory_backend=mem_backend,
        )
        result = await agent.run("What was that fact?")
        assert result.state == AgentState.COMPLETED
        mem_backend.search.assert_called_once()
        assert result.metadata is not None
        assert result.metadata["memories_retrieved"] == 2

    async def test_run_semantic_search_filters_by_threshold(self):
        """Results below score_threshold are filtered out."""
        mem_backend = _make_memory_backend(
            search_results=[
                MemoryResult(key="k1", value="Relevant", score=0.9),
                MemoryResult(key="k2", value="Not relevant", score=0.1),
            ]
        )
        provider = MockLLMProvider(responses=[make_chat_response("Ok!")])
        config = _make_config(memory=AgentMemoryConfig(semantic_search_score_threshold=0.5))
        agent = _MemoryTestAgent(
            config=config,
            provider=provider,
            memory_backend=mem_backend,
        )
        result = await agent.run("Test")
        assert result.metadata is not None
        assert result.metadata["memories_retrieved"] == 1

    async def test_run_semantic_search_no_results(self):
        """Agent handles empty search results gracefully."""
        mem_backend = _make_memory_backend(search_results=[])
        provider = MockLLMProvider(responses=[make_chat_response("No context!")])
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            memory_backend=mem_backend,
        )
        result = await agent.run("Something new")
        assert result.state == AgentState.COMPLETED
        assert result.metadata is not None
        assert result.metadata["memories_retrieved"] == 0

    async def test_run_semantic_search_disabled(self):
        """Memory search is skipped when semantic_search_enabled=False."""
        mem_backend = _make_memory_backend()
        provider = MockLLMProvider(responses=[make_chat_response("No search!")])
        config = _make_config(memory=AgentMemoryConfig(semantic_search_enabled=False))
        agent = _MemoryTestAgent(
            config=config,
            provider=provider,
            memory_backend=mem_backend,
        )
        result = await agent.run("Test")
        assert result.state == AgentState.COMPLETED
        mem_backend.search.assert_not_called()

    async def test_run_memory_disabled(self):
        """All memory ops skipped when memory.enabled=False."""
        mem_backend = _make_memory_backend()
        conv_backend = _make_conversation_backend()
        provider = MockLLMProvider(responses=[make_chat_response("No memory!")])
        config = _make_config(memory=AgentMemoryConfig(enabled=False))
        agent = _MemoryTestAgent(
            config=config,
            provider=provider,
            memory_backend=mem_backend,
            conversation_backend=conv_backend,
        )
        result = await agent.run("Test", conversation_id="conv-1")
        assert result.state == AgentState.COMPLETED
        mem_backend.search.assert_not_called()
        mem_backend.store.assert_not_called()
        conv_backend.get_messages.assert_not_called()
        conv_backend.add_message.assert_not_called()

    async def test_memory_backend_error_non_fatal(self):
        """Memory errors are logged but don't crash the agent."""
        mem_backend = _make_memory_backend()
        mem_backend.search = AsyncMock(side_effect=RuntimeError("Connection lost"))
        mem_backend.store = AsyncMock(side_effect=RuntimeError("Connection lost"))
        provider = MockLLMProvider(responses=[make_chat_response("Still works!")])
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            memory_backend=mem_backend,
        )
        result = await agent.run("Test")
        assert result.state == AgentState.COMPLETED
        assert result.output == "Still works!"

    async def test_conversation_backend_error_non_fatal(self):
        """Conversation backend errors don't crash the agent."""
        conv_backend = _make_conversation_backend()
        conv_backend.get_messages = AsyncMock(side_effect=RuntimeError("DB down"))
        conv_backend.add_message = AsyncMock(side_effect=RuntimeError("DB down"))
        provider = MockLLMProvider(responses=[make_chat_response("Resilient!")])
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            conversation_backend=conv_backend,
        )
        result = await agent.run("Test", conversation_id="conv-fail")
        assert result.state == AgentState.COMPLETED
        assert result.output == "Resilient!"

    async def test_explicit_history_takes_precedence(self):
        """When both conversation_history and conversation_backend are set,
        the explicit history takes precedence."""
        from ia_agent_fwk.llm.models import Message

        conv_backend = _make_conversation_backend(
            messages=[
                ConversationMessage(id="m1", conversation_id="c1", role="user", content="Backend msg"),
            ]
        )
        provider = MockLLMProvider(responses=[make_chat_response("Got it!")])
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            conversation_backend=conv_backend,
        )
        explicit = [Message(role="user", content="Explicit msg")]
        result = await agent.run("New", conversation_history=explicit, conversation_id="c1")
        assert result.state == AgentState.COMPLETED
        # conversation_backend.get_messages should NOT be called
        conv_backend.get_messages.assert_not_called()

    async def test_metadata_includes_conversation_id(self):
        """AgentResult.metadata includes conversation_id."""
        provider = MockLLMProvider(responses=[make_chat_response("Done!")])
        mem_backend = _make_memory_backend()
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            memory_backend=mem_backend,
        )
        result = await agent.run("Test", conversation_id="my-conv-id")
        assert result.metadata is not None
        assert result.metadata["conversation_id"] == "my-conv-id"

    async def test_auto_generates_conversation_id(self):
        """When no conversation_id is given but backends exist, one is auto-generated."""
        mem_backend = _make_memory_backend()
        provider = MockLLMProvider(responses=[make_chat_response("Auto!")])
        agent = _MemoryTestAgent(
            config=_make_config(),
            provider=provider,
            memory_backend=mem_backend,
        )
        result = await agent.run("Test")
        assert result.metadata is not None
        assert result.metadata["conversation_id"] is not None
        assert len(result.metadata["conversation_id"]) == 32  # uuid hex


@pytest.mark.unit
class TestFormatMemories:
    def test_format_memories_as_context(self):
        memories = [
            MemoryResult(key="k1", value="Fact A", score=0.9),
            MemoryResult(key="k2", value="Fact B", score=0.7),
        ]
        text = _MemoryTestAgent._format_memories_as_context(memories)
        assert "[Relevant context" in text
        assert "Fact A" in text
        assert "0.90" in text
        assert "Fact B" in text

    def test_format_empty_memories(self):
        text = _MemoryTestAgent._format_memories_as_context([])
        assert "[Relevant context" in text
