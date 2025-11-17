"""In-memory conversation history backend.

Stores messages per conversation with agent namespace isolation.
Messages are evicted (oldest first) when ``max_history`` is exceeded.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.exceptions import MemoryStoreError
from ia_agent_fwk.memory.models import ConversationInfo, ConversationMessage, MemoryResult
from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class ConversationMemoryBackend(MemoryBackend):
    """In-memory conversation history backend.

    Parameters
    ----------
    max_history:
        Maximum messages per conversation. When exceeded, the oldest
        messages are dropped.

    """

    def __init__(self, max_history: int = 100) -> None:
        self._max_history = max_history
        self._conversations: dict[str, ConversationInfo] = {}
        self._messages: dict[str, list[ConversationMessage]] = {}

    @property
    def backend_type(self) -> str:
        """Return ``'conversation'``."""
        return "conversation"

    # ------------------------------------------------------------------
    # Conversation-specific operations
    # ------------------------------------------------------------------

    async def create_conversation(
        self,
        agent_namespace: str,
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> ConversationInfo:
        """Create a new conversation.

        Parameters
        ----------
        agent_namespace:
            Logical namespace for the owning agent.
        conversation_id:
            Explicit ID. If ``None``, a UUID4 is generated.
        title:
            Optional conversation title.

        """
        cid = conversation_id or str(uuid.uuid4())
        info = ConversationInfo(
            conversation_id=cid,
            agent_namespace=agent_namespace,
            title=title,
        )
        self._conversations[cid] = info
        self._messages[cid] = []
        collector = get_metrics_collector()
        collector.increment("conversation_operations_total", labels={"operation": "create"})
        logger.info(
            "Conversation created: id=%s, namespace=%s",
            cid,
            agent_namespace,
            extra={
                "memory_data": {
                    "event": "conversation_created",
                    "conversation_id": cid,
                    "agent_namespace": agent_namespace,
                }
            },
        )
        return info

    async def get_conversation(self, conversation_id: str) -> ConversationInfo | None:
        """Retrieve conversation info by ID. Returns ``None`` if not found."""
        return self._conversations.get(conversation_id)

    async def list_conversations(
        self,
        agent_namespace: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ConversationInfo], int]:
        """List conversations with optional namespace filter and pagination.

        Returns
        -------
        tuple[list[ConversationInfo], int]
            A page of conversations and the total count matching the filter.

        """
        conversations = list(self._conversations.values())
        if agent_namespace is not None:
            conversations = [c for c in conversations if c.agent_namespace == agent_namespace]

        total = len(conversations)
        page = conversations[offset : offset + limit]
        return page, total

    async def add_message(  # noqa: PLR0913
        self,
        conversation_id: str,
        role: str,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Add a message to an existing conversation.

        Auto-generates a UUID4 message ID. Evicts the oldest messages
        when ``max_history`` is exceeded.

        Raises
        ------
        MemoryStoreError
            If the conversation does not exist.

        """
        if conversation_id not in self._conversations:
            msg = f"Conversation {conversation_id} does not exist"
            raise MemoryStoreError(msg)

        now = datetime.now(timezone.utc)  # noqa: UP017
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            token_count=token_count,
            metadata=metadata or {},
            created_at=now,
        )

        msgs = self._messages[conversation_id]
        msgs.append(message)

        collector = get_metrics_collector()
        # Evict oldest messages when max_history is exceeded
        if len(msgs) > self._max_history:
            evicted = len(msgs) - self._max_history
            self._messages[conversation_id] = msgs[-self._max_history :]
            collector.increment("conversation_messages_evicted_total", value=evicted)

        # Update conversation info
        info = self._conversations[conversation_id]
        current_msgs = self._messages[conversation_id]
        self._conversations[conversation_id] = info.model_copy(
            update={
                "message_count": len(current_msgs),
                "last_message_at": now,
            }
        )

        collector.increment("conversation_messages_total", labels={"role": role})
        return message

    async def get_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[ConversationMessage]:
        """Return messages in chronological order, optionally limited.

        When *limit* is provided, the **most recent** (newest) N messages
        are returned instead of the oldest, which is more useful for
        conversation context windows and chat UIs.
        """
        msgs = self._messages.get(conversation_id, [])
        if limit is not None:
            return msgs[-limit:]
        return list(msgs)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages. Returns ``True`` if it existed."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            self._messages.pop(conversation_id, None)
            collector = get_metrics_collector()
            collector.increment("conversation_operations_total", labels={"operation": "delete"})
            logger.info(
                "Conversation deleted: id=%s",
                conversation_id,
                extra={
                    "memory_data": {
                        "event": "conversation_deleted",
                        "conversation_id": conversation_id,
                    }
                },
            )
            return True
        return False

    # ------------------------------------------------------------------
    # MemoryBackend interface
    # ------------------------------------------------------------------

    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Delegate to ``add_message()``.

        *key* is the ``conversation_id``, *value* is expected to be a
        dict with at least a ``role`` field (and optionally ``content``).
        """
        if isinstance(value, dict):
            await self.add_message(
                conversation_id=key,
                role=value.get("role", "user"),
                content=value.get("content"),
                tool_calls=value.get("tool_calls"),
                tool_call_id=value.get("tool_call_id"),
                token_count=value.get("token_count"),
                metadata=metadata,
            )
        else:
            await self.add_message(
                conversation_id=key,
                role="user",
                content=str(value),
                metadata=metadata,
            )

    async def retrieve(self, key: str) -> list[ConversationMessage] | None:
        """Return the message list for a conversation, or ``None`` if it doesn't exist."""
        if key not in self._conversations:
            return None
        return await self.get_messages(key)

    async def search(self, query: str, top_k: int = 5) -> list[MemoryResult]:
        """Substring search on message content across all conversations."""
        results: list[MemoryResult] = []
        query_lower = query.lower()

        for conversation_id, msgs in self._messages.items():
            for msg in msgs:
                if msg.content and query_lower in msg.content.lower():
                    results.append(
                        MemoryResult(
                            key=conversation_id,
                            value=msg.content,
                            score=0.5,
                            metadata={"message_id": msg.id, "role": msg.role},
                        )
                    )
                    if len(results) >= top_k:
                        return results

        return results[:top_k]

    async def delete(self, key: str) -> bool:
        """Delegate to ``delete_conversation()``."""
        return await self.delete_conversation(key)

    async def clear(self) -> None:
        """Remove all conversations and messages."""
        self._conversations.clear()
        self._messages.clear()

    async def health_check(self) -> bool:
        """Return ``True`` (in-memory backend is always healthy)."""
        return True
