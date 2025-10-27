"""Agent context management.

``AgentContext`` is a mutable state container that manages conversation
history, current task, intermediate tool results, token budget tracking,
and sliding window eviction.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.llm.models import Message
from ia_agent_fwk.observability.metrics import get_metrics_collector

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Safety margin: 20% of token budget (10% spec safety + 10% overflow buffer)
_SAFETY_MARGIN_RATIO = 0.20

# Default context window when none is configured
_DEFAULT_CONTEXT_WINDOW = 8192


def _message_text(message: Message) -> str:
    """Extract countable text from a message."""
    parts: list[str] = []
    if message.content:
        parts.append(message.content)
    if message.tool_calls:
        for tc in message.tool_calls:
            parts.append(tc.name)
            parts.append(tc.arguments)
    return " ".join(parts) if parts else ""


class AgentContext:
    """Mutable context container for an agent's execution.

    Parameters
    ----------
    system_prompt:
        The system prompt text. Measured once at init for token budgeting.
    token_budget:
        Total tokens available (model context window or configured limit).
    token_counter:
        Callable that counts tokens in a string (e.g. ``provider.count_tokens``).

    """

    def __init__(
        self,
        system_prompt: str,
        token_budget: int,
        token_counter: Callable[[str], int],
    ) -> None:
        self._system_prompt = system_prompt
        self._token_budget = token_budget
        self._token_counter = token_counter

        # Measure system prompt tokens once
        self._system_prompt_tokens = token_counter(system_prompt) if system_prompt else 0

        # Safety margin
        self._safety_margin = int(token_budget * _SAFETY_MARGIN_RATIO)

        # Available budget for conversation history
        self._history_budget = token_budget - self._system_prompt_tokens - self._safety_margin

        if self._history_budget <= 0:
            logger.warning(
                "Token budget (%d) is too small for system prompt (%d tokens) "
                "+ safety margin (%d). No conversation history will be retained.",
                token_budget,
                self._system_prompt_tokens,
                self._safety_margin,
            )

        # Conversation history (excluding system prompt) -- deque for O(1) popleft
        self._history: deque[Message] = deque()

        # Running token total for O(1) budget checks
        self._total_token_count: int = 0

        # Per-message token cache: keyed by hash(message) (frozen Pydantic models
        # are hashable) for a deterministic, GC-safe cache key.
        self._token_cache: dict[int, int] = {}

        # Eviction tracking
        self._eviction_count: int = 0

        # Current task tracking
        self._current_task: str | None = None

        # Intermediate tool results
        self.intermediate_results: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_task(self) -> str | None:
        """Get the current task string."""
        return self._current_task

    @current_task.setter
    def current_task(self, value: str | None) -> None:
        """Set the current task string."""
        self._current_task = value

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt

    @property
    def token_budget(self) -> int:
        """Get the total token budget."""
        return self._token_budget

    @property
    def history_budget(self) -> int:
        """Get the token budget available for conversation history."""
        return self._history_budget

    @property
    def message_count(self) -> int:
        """Return number of messages in history (excluding system prompt)."""
        return len(self._history)

    @property
    def eviction_count(self) -> int:
        """Return total number of messages evicted by sliding window."""
        return self._eviction_count

    @property
    def token_utilization(self) -> float:
        """Return token utilization ratio (0.0 to 1.0) of the history budget."""
        if self._history_budget <= 0:
            return 0.0
        return min(self._total_token_count / self._history_budget, 1.0)

    def get_stats(self) -> dict[str, Any]:
        """Return context statistics for observability."""
        return {
            "message_count": len(self._history),
            "token_budget": self._token_budget,
            "history_budget": self._history_budget,
            "tokens_used": self._total_token_count,
            "system_prompt_tokens": self._system_prompt_tokens,
            "safety_margin": self._safety_margin,
            "token_utilization": round(self.token_utilization, 3),
            "eviction_count": self._eviction_count,
        }

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    @staticmethod
    def _message_cache_key(message: Message) -> int:
        """Compute a deterministic, GC-safe cache key for a Message.

        We cannot use ``hash(message)`` because frozen Pydantic models with
        list fields (e.g. ``tool_calls``) are not hashable. Instead we hash
        a tuple of the primitive fields and the JSON-serialized model.
        """
        return hash(message.model_dump_json())

    def _count_message_tokens(self, message: Message) -> int:
        """Count tokens for a message, using cache when possible."""
        key = self._message_cache_key(message)
        if key not in self._token_cache:
            text = _message_text(message)
            self._token_cache[key] = self._token_counter(text)
        return self._token_cache[key]

    def _total_history_tokens(self) -> int:
        """Return the running token total for all history messages."""
        return self._total_token_count

    def _apply_sliding_window(self) -> None:
        """Drop oldest non-system messages until history fits within budget."""
        collector = get_metrics_collector()
        while self._history and self._total_token_count > self._history_budget:
            removed = self._history.popleft()
            removed_tokens = self._count_message_tokens(removed)
            self._total_token_count -= removed_tokens
            self._eviction_count += 1
            collector.increment("agent_context_evictions_total")
            logger.debug(
                "Context sliding window evicted %s message (%d tokens, total evictions: %d)",
                removed.role,
                removed_tokens,
                self._eviction_count,
            )

    def add_message(self, message: Message) -> None:
        """Append a message to history and enforce the token budget."""
        tokens = self._count_message_tokens(message)
        self._total_token_count += tokens
        self._history.append(message)
        self._apply_sliding_window()

    def get_messages(self) -> list[Message]:
        """Build and return the full message list: system prompt + history."""
        messages: list[Message] = []
        if self._system_prompt:
            messages.append(Message(role="system", content=self._system_prompt))
        messages.extend(self._history)
        return messages

    def inject_memory_context(self, memory_text: str) -> None:
        """Insert retrieved memory context at the beginning of history.

        The memory message is added as a ``system`` role message so the LLM
        sees relevant past context before the conversation messages. It
        participates in the sliding window and will be evicted first when
        the context is too large (which is the correct behavior -- recent
        conversation messages are more important than old memories).
        """
        if not memory_text:
            return
        memory_message = Message(role="system", content=memory_text)
        tokens = self._count_message_tokens(memory_message)
        self._total_token_count += tokens
        self._history.appendleft(memory_message)
        self._apply_sliding_window()

    def clear(self) -> None:
        """Clear conversation history (not the system prompt)."""
        self._history.clear()
        self._token_cache.clear()
        self._total_token_count = 0
        self.intermediate_results.clear()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to a dictionary."""
        return {
            "system_prompt": self._system_prompt,
            "token_budget": self._token_budget,
            "current_task": self._current_task,
            "intermediate_results": dict(self.intermediate_results),
            "history": [m.model_dump() for m in self._history],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        token_counter: Callable[[str], int],
    ) -> AgentContext:
        """Deserialize context from a dictionary."""
        ctx = cls(
            system_prompt=data["system_prompt"],
            token_budget=data["token_budget"],
            token_counter=token_counter,
        )
        ctx._current_task = data.get("current_task")
        ctx.intermediate_results = dict(data.get("intermediate_results", {}))
        for msg_data in data.get("history", []):
            msg = Message(**msg_data)
            ctx.add_message(msg)
        return ctx
