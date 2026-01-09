"""Prompt logging for LLM interactions.

``PromptLogger`` captures all LLM calls (input messages, output,
tokens, latency, model, provider) via a dedicated structured logger.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import PromptLoggingSettings

# Dedicated logger name for prompt audit trail
_PROMPT_LOGGER_NAME = "ia_agent_fwk.prompts"


class PromptLogger:
    """Log LLM prompt/response interactions with structured data.

    Parameters
    ----------
    settings:
        Prompt logging configuration from
        ``ObservabilitySettings.prompt_logging``.

    """

    def __init__(self, settings: PromptLoggingSettings) -> None:
        self._settings = settings
        self._logger = logging.getLogger(_PROMPT_LOGGER_NAME)

    @property
    def enabled(self) -> bool:
        """Return ``True`` if prompt logging is enabled."""
        return self._settings.enabled

    def log_prompt(  # noqa: PLR0913
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        response: str,
        duration_ms: float,
        usage: dict[str, int] | None = None,
        agent: str = "",
        iteration: int = 0,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log a single LLM interaction.

        Parameters
        ----------
        provider:
            LLM provider name (e.g. ``"openai"``).
        model:
            Model identifier (e.g. ``"gpt-4o"``).
        messages:
            Input messages sent to the LLM.
        response:
            LLM response text.
        duration_ms:
            Wall-clock time in milliseconds.
        usage:
            Token usage dict (``prompt_tokens``, ``completion_tokens``,
            ``total_tokens``).
        agent:
            Agent type identifier (e.g. ``"document_processor"``).
        iteration:
            Current reasoning loop iteration number.
        tool_calls:
            List of tool call dicts from the LLM response, if any.

        """
        if not self._settings.enabled:
            return

        record: dict[str, Any] = {
            "event": "llm_prompt",
            "provider": provider,
            "model": model,
        }

        if agent:
            record["agent"] = agent

        if iteration:
            record["iteration"] = iteration

        if self._settings.log_inputs:
            record["messages"] = messages

        if self._settings.log_outputs:
            record["response"] = response

        if tool_calls:
            record["tool_calls"] = tool_calls

        if self._settings.log_latency:
            record["duration_ms"] = duration_ms

        if self._settings.log_tokens and usage:
            record["usage"] = usage

        self._logger.info(
            "LLM call: agent=%s provider=%s model=%s duration=%.1fms tokens=%s tools=%d",
            agent or "unknown",
            provider,
            model,
            duration_ms,
            usage.get("total_tokens", 0) if usage else 0,
            len(tool_calls) if tool_calls else 0,
            extra={"prompt_data": record},
        )
