"""Tests for ConversationMemoryBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ia_agent_fwk.memory.exceptions import MemoryStoreError
from ia_agent_fwk.memory.models import ConversationInfo, ConversationMessage

if TYPE_CHECKING:
    from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend


@pytest.mark.unit
class TestConversationCreation:
    async def test_create_conversation(self, conversation_backend: ConversationMemoryBackend):
        info = await conversation_backend.create_conversation(agent_namespace="agent-1")
        assert isinstance(info, ConversationInfo)
        assert info.agent_namespace == "agent-1"
        assert info.conversation_id  # auto-generated UUID
        assert info.message_count == 0

    async def test_create_conversation_with_id(self, conversation_backend: ConversationMemoryBackend):
        info = await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="my-conv-1",
        )
        assert info.conversation_id == "my-conv-1"

    async def test_create_conversation_with_title(self, conversation_backend: ConversationMemoryBackend):
        info = await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            title="Test Conversation",
        )
        assert info.title == "Test Conversation"


@pytest.mark.unit
class TestConversationRetrieval:
    async def test_get_conversation(self, conversation_backend: ConversationMemoryBackend):
        created = await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        retrieved = await conversation_backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.conversation_id == created.conversation_id

    async def test_get_conversation_not_found(self, conversation_backend: ConversationMemoryBackend):
        result = await conversation_backend.get_conversation("nonexistent")
        assert result is None


@pytest.mark.unit
class TestConversationListing:
    async def test_list_conversations(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(agent_namespace="agent-1")
        await conversation_backend.create_conversation(agent_namespace="agent-2")
        conversations, total = await conversation_backend.list_conversations()
        assert total == 2
        assert len(conversations) == 2

    async def test_list_conversations_by_namespace(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(agent_namespace="agent-1")
        await conversation_backend.create_conversation(agent_namespace="agent-1")
        await conversation_backend.create_conversation(agent_namespace="agent-2")
        conversations, total = await conversation_backend.list_conversations(agent_namespace="agent-1")
        assert total == 2
        assert len(conversations) == 2
        assert all(c.agent_namespace == "agent-1" for c in conversations)

    async def test_list_conversations_pagination(self, conversation_backend: ConversationMemoryBackend):
        for i in range(5):
            await conversation_backend.create_conversation(
                agent_namespace="agent-1",
                conversation_id=f"conv-{i}",
            )
        page1, total = await conversation_backend.list_conversations(limit=2, offset=0)
        assert total == 5
        assert len(page1) == 2

        page2, total2 = await conversation_backend.list_conversations(limit=2, offset=2)
        assert total2 == 5
        assert len(page2) == 2

        page3, total3 = await conversation_backend.list_conversations(limit=2, offset=4)
        assert total3 == 5
        assert len(page3) == 1


@pytest.mark.unit
class TestMessageOperations:
    async def test_add_message(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        msg = await conversation_backend.add_message(
            conversation_id="conv-1",
            role="user",
            content="Hello!",
        )
        assert isinstance(msg, ConversationMessage)
        assert msg.conversation_id == "conv-1"
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.id  # auto-generated UUID

    async def test_add_message_nonexistent_conversation_raises(self, conversation_backend: ConversationMemoryBackend):
        with pytest.raises(MemoryStoreError, match="does not exist"):
            await conversation_backend.add_message(
                conversation_id="nonexistent",
                role="user",
                content="Orphan message",
            )

    async def test_add_message_updates_count(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        await conversation_backend.add_message(
            conversation_id="conv-1",
            role="user",
            content="Hello!",
        )
        info = await conversation_backend.get_conversation("conv-1")
        assert info is not None
        assert info.message_count == 1

        await conversation_backend.add_message(
            conversation_id="conv-1",
            role="assistant",
            content="Hi there!",
        )
        info = await conversation_backend.get_conversation("conv-1")
        assert info is not None
        assert info.message_count == 2

    async def test_add_message_updates_last_message_at(self, conversation_backend: ConversationMemoryBackend):
        info = await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        assert info.last_message_at is None

        await conversation_backend.add_message(
            conversation_id="conv-1",
            role="user",
            content="Hello!",
        )
        info = await conversation_backend.get_conversation("conv-1")
        assert info is not None
        assert info.last_message_at is not None

    async def test_get_messages(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        await conversation_backend.add_message(conversation_id="conv-1", role="user", content="First")
        await conversation_backend.add_message(conversation_id="conv-1", role="assistant", content="Second")
        msgs = await conversation_backend.get_messages("conv-1")
        assert len(msgs) == 2
        assert msgs[0].content == "First"
        assert msgs[1].content == "Second"

    async def test_get_messages_with_limit(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        for i in range(5):
            await conversation_backend.add_message(
                conversation_id="conv-1",
                role="user",
                content=f"Message {i}",
            )
        msgs = await conversation_backend.get_messages("conv-1", limit=3)
        assert len(msgs) == 3
        # Should return the 3 most recent messages (newest)
        assert msgs[0].content == "Message 2"
        assert msgs[2].content == "Message 4"

    async def test_max_history_eviction(self, conversation_backend: ConversationMemoryBackend):
        # max_history=5 from fixture
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        for i in range(7):
            await conversation_backend.add_message(
                conversation_id="conv-1",
                role="user",
                content=f"Message {i}",
            )
        msgs = await conversation_backend.get_messages("conv-1")
        assert len(msgs) == 5
        # Oldest messages (0, 1) should be evicted; messages 2-6 remain
        assert msgs[0].content == "Message 2"
        assert msgs[4].content == "Message 6"


@pytest.mark.unit
class TestConversationDeletion:
    async def test_delete_conversation(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        await conversation_backend.add_message(conversation_id="conv-1", role="user", content="Hello!")
        deleted = await conversation_backend.delete_conversation("conv-1")
        assert deleted is True
        assert await conversation_backend.get_conversation("conv-1") is None
        msgs = await conversation_backend.get_messages("conv-1")
        assert msgs == []

    async def test_delete_conversation_not_found(self, conversation_backend: ConversationMemoryBackend):
        deleted = await conversation_backend.delete_conversation("nonexistent")
        assert deleted is False


@pytest.mark.unit
class TestNamespaceIsolation:
    async def test_namespace_isolation(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        await conversation_backend.create_conversation(
            agent_namespace="agent-2",
            conversation_id="conv-2",
        )
        await conversation_backend.add_message(conversation_id="conv-1", role="user", content="Agent 1 message")
        await conversation_backend.add_message(conversation_id="conv-2", role="user", content="Agent 2 message")

        convs_1, total_1 = await conversation_backend.list_conversations(agent_namespace="agent-1")
        assert total_1 == 1
        assert convs_1[0].conversation_id == "conv-1"

        msgs_1 = await conversation_backend.get_messages("conv-1")
        assert len(msgs_1) == 1
        assert msgs_1[0].content == "Agent 1 message"


@pytest.mark.unit
class TestMemoryBackendInterface:
    async def test_store_interface(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        await conversation_backend.store("conv-1", {"role": "user", "content": "Hello via store!"})
        msgs = await conversation_backend.get_messages("conv-1")
        assert len(msgs) == 1
        assert msgs[0].content == "Hello via store!"

    async def test_retrieve_interface(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        await conversation_backend.add_message(conversation_id="conv-1", role="user", content="Test")
        result = await conversation_backend.retrieve("conv-1")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].content == "Test"

    async def test_search_interface(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        await conversation_backend.add_message(
            conversation_id="conv-1",
            role="user",
            content="The weather is sunny today",
        )
        results = await conversation_backend.search("sunny")
        assert len(results) == 1
        assert results[0].value == "The weather is sunny today"

    async def test_clear_interface(self, conversation_backend: ConversationMemoryBackend):
        await conversation_backend.create_conversation(
            agent_namespace="agent-1",
            conversation_id="conv-1",
        )
        await conversation_backend.add_message(conversation_id="conv-1", role="user", content="Test")
        await conversation_backend.clear()
        assert await conversation_backend.get_conversation("conv-1") is None
        _convs, total = await conversation_backend.list_conversations()
        assert total == 0

    async def test_health_check(self, conversation_backend: ConversationMemoryBackend):
        result = await conversation_backend.health_check()
        assert result is True

    async def test_backend_type(self, conversation_backend: ConversationMemoryBackend):
        assert conversation_backend.backend_type == "conversation"
