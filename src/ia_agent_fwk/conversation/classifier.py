"""Message classifier — SEARCH vs CHAT with query rewriting."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassifyResult:
    """Result of message classification."""

    mode: str  # "search" or "chat"
    search_query: str = ""
    filters: dict[str, str] | None = None
    cached_context: str | None = None


class MessageClassifier:
    """Classify user messages as SEARCH or CHAT and rewrite queries.

    All prompts and parameters come from config. No hardcoded text.

    Parameters
    ----------
    llm_quick:
        Async callable(prompt, max_tokens) -> str for fast LLM calls.
    classify_prompt:
        Prompt template with {history} and {message} placeholders.
    max_tokens:
        Max tokens for the classify+rewrite LLM call.
    history_window:
        Number of recent messages to include in the prompt.

    """

    def __init__(
        self,
        llm_quick: Callable[..., Awaitable[str]],
        classify_prompt: str,
        max_tokens: int = 80,
        history_window: int = 4,
    ) -> None:
        self._llm_quick = llm_quick
        self._classify_prompt = classify_prompt
        self._max_tokens = max_tokens
        self._history_window = history_window

    async def classify(
        self,
        message: str,
        history: list[dict[str, str]],
        cached_context: str | None = None,
    ) -> ClassifyResult:
        """Classify a message and optionally rewrite the search query.

        Parameters
        ----------
        message:
            The user's message.
        history:
            List of {"role": "user"|"assistant", "content": "..."} dicts.
        cached_context:
            Last RAG context for this session. Returned if mode=CHAT.

        """
        # Format recent history (empty string if no history)
        recent = history[-self._history_window :]
        history_text = "\n".join(
            f"{'Usuario' if m['role'] == 'user' else 'Asesor'}: {m['content'][:150]}" for m in recent
        )

        prompt = self._classify_prompt.format(
            history=history_text,
            message=message,
        )
        result = await self._llm_quick(prompt, self._max_tokens)
        result = result.strip()
        logger.info("Classify+rewrite: '%s' → '%s'", message[:50], result[:100])

        if result.upper().startswith("CHAT"):
            return ClassifyResult(
                mode="chat",
                cached_context=cached_context or "",
            )

        # Parse SEARCH result: extract query and optional filters
        filters = None
        lines = result.split("\n")
        query_line = lines[0]
        for line in lines[1:]:
            stripped = line.strip().upper()
            if ":" in stripped:
                key, _, val = line.strip().partition(":")
                key = key.strip().upper()
                val = val.strip()
                if key == "INSURER" and val:
                    filters = {"insurer": val}
                    logger.info("Filter detected: insurer=%s", val)

        # Strip SEARCH: prefix
        query = query_line
        for prefix in ("SEARCH:", "SEARCH"):
            if query.upper().startswith(prefix):
                query = query[len(prefix) :].strip()
                break

        return ClassifyResult(
            mode="search",
            search_query=query or message,
            filters=filters,
        )
