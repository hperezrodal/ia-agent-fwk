"""Anthropic provider implementation.

Uses the official ``anthropic`` Python SDK.  SDK-level retries are disabled
(``max_retries=0``) in favour of the framework retry layer.

System messages are extracted from the message list and passed via the
``system`` parameter as required by the Anthropic Messages API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import anthropic

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ia_agent_fwk.config.settings import LLMProviderSettings
from ia_agent_fwk.llm.base import LLMProvider
from ia_agent_fwk.llm.circuit_breaker import CircuitBreaker
from ia_agent_fwk.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigError,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamError,
    LLMTimeoutError,
)
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
from ia_agent_fwk.llm.retry import with_retry
from ia_agent_fwk.llm.streaming import buffered_stream
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


def _map_finish_reason(stop_reason: str | None) -> FinishReason:
    mapping: dict[str | None, FinishReason] = {
        "end_turn": FinishReason.stop,
        "stop_sequence": FinishReason.stop,
        "tool_use": FinishReason.tool_calls,
        "max_tokens": FinishReason.length,
    }
    if stop_reason is not None and stop_reason not in mapping:
        logger.warning("Unmapped Anthropic stop_reason '%s'; treating as error", stop_reason)
        return FinishReason.error
    return mapping.get(stop_reason, FinishReason.stop)


def _map_anthropic_error(exc: anthropic.AnthropicError) -> LLMProviderError:
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMAuthenticationError(str(exc))
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError(str(exc))
    return LLMProviderError(str(exc))


def _extract_system_and_messages(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Separate system messages from non-system messages.

    Returns
    -------
    system_text:
        Concatenation of all system-role contents
        (or ``None`` if there are none).
    anthropic_messages:
        List of dicts in Anthropic's expected format.

    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
        else:
            converted.append(_to_anthropic_message(msg))

    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, converted


def _to_anthropic_message(msg: Message) -> dict[str, Any]:
    role = "assistant" if msg.role == "assistant" else "user"
    content: Any

    if msg.role == "tool":
        content = [
            {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id or "",
                "content": msg.content or "",
            }
        ]
        role = "user"
    elif msg.tool_calls:
        blocks: list[dict[str, Any]] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        blocks.extend(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": json.loads(tc.arguments),
            }
            for tc in msg.tool_calls
        )
        content = blocks
    else:
        content = msg.content or ""

    return {"role": role, "content": content}


def _process_stream_event(
    event: Any,
    collected: str,
    input_tokens: int,
    output_tokens: int,
) -> tuple[StreamChunk | None, str, int, int]:
    """Process a single Anthropic stream event into an optional StreamChunk."""
    if not hasattr(event, "type"):
        return None, collected, input_tokens, output_tokens

    if event.type == "content_block_delta":
        delta_text = event.delta.text if hasattr(event.delta, "text") else ""
        collected += delta_text
        return StreamChunk(delta=delta_text), collected, input_tokens, output_tokens

    if event.type == "message_start":
        if hasattr(event, "message") and hasattr(event.message, "usage"):
            input_tokens = event.message.usage.input_tokens
        return None, collected, input_tokens, output_tokens

    if event.type == "message_delta":
        if hasattr(event, "usage") and event.usage is not None:
            output_tokens = event.usage.output_tokens
        sr = _extract_stop_reason(event)
        chunk = StreamChunk(
            delta="",
            finish_reason=_map_finish_reason(sr),
            usage=TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            ),
        )
        return chunk, collected, input_tokens, output_tokens

    return None, collected, input_tokens, output_tokens


def _extract_stop_reason(event: Any) -> str | None:
    """Safely extract stop_reason from a message_delta event."""
    if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
        return event.delta.stop_reason  # type: ignore[no-any-return]
    return None


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider (Claude 4, Claude 3.5 Sonnet, ...)."""

    def __init__(self, settings: LLMProviderSettings, provider_name: str, **_kwargs: Any) -> None:
        super().__init__(settings, provider_name)
        api_key = settings.api_key.get_secret_value()
        # F-008: Validate API key availability early.
        if not api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            msg = (
                "Anthropic API key is not configured. "
                "Set 'api_key' in provider settings or the ANTHROPIC_API_KEY environment variable."
            )
            raise LLMConfigError(msg)
        client_kwargs: dict[str, Any] = {
            "max_retries": 0,
            # F-012: Pass timeout to match configured value.
            "timeout": float(settings.timeout),
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = anthropic.AsyncAnthropic(**client_kwargs)
        self._circuit_breaker = CircuitBreaker(
            provider_name=provider_name,
            settings=settings.circuit_breaker,
        )

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    def format_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool schemas to Anthropic format.

        Anthropic expects: ``{"name": ..., "description": ..., "input_schema": ...}``
        OpenAI provides: ``{"type": "function", "function": {"name", "description", "parameters"}}``
        """
        anthropic_tools: list[dict[str, Any]] = []
        for schema in schemas:
            func = schema.get("function", {})
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                }
            )
        return anthropic_tools

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        collector = get_metrics_collector()
        model = kwargs.get("model", self.settings.default_model)
        start = time.monotonic()

        async def _do_chat() -> ChatResponse:
            # Copy kwargs to avoid mutating the outer dict on retry (F-003).
            kw = dict(kwargs)
            model = kw.pop("model", self.settings.default_model)
            system_text, anthropic_messages = _extract_system_and_messages(messages)

            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": kw.pop("max_tokens", self.settings.max_tokens),
                "temperature": kw.pop("temperature", self.settings.temperature),
            }
            if system_text is not None:
                create_kwargs["system"] = system_text
            create_kwargs.update(kw)

            try:
                response = await self._client.messages.create(**create_kwargs)
            except anthropic.AnthropicError as exc:
                raise _map_anthropic_error(exc) from exc

            # Extract text content and tool calls from content blocks.
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=json.dumps(block.input),
                        )
                    )

            return ChatResponse(
                message=Message(
                    role="assistant",
                    content="".join(text_parts) if text_parts else None,
                    tool_calls=tool_calls or None,
                ),
                usage=TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                ),
                model=response.model,
                finish_reason=_map_finish_reason(response.stop_reason),
            )

        with _tracer.start_as_current_span(
            "llm.chat",
            attributes={"llm.provider": "anthropic", "llm.model": model},
        ) as span:
            try:
                result = await self._circuit_breaker.call(
                    with_retry,
                    _do_chat,
                    retry_settings=self.settings.retry,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.record_exception(exc)
                collector.increment(
                    "llm_requests_total",
                    labels={
                        "provider": "anthropic",
                        "model": model,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    },
                )
                collector.observe(
                    "llm_request_duration_seconds", duration_ms / 1000, labels={"provider": "anthropic", "model": model}
                )
                logger.exception(
                    "Anthropic chat failed: model=%s (%.1fms)",
                    model,
                    duration_ms,
                    extra={
                        "llm_data": {
                            "event": "chat_error",
                            "provider": "anthropic",
                            "model": model,
                            "duration_ms": round(duration_ms, 1),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    },
                )
                raise
            else:
                duration_ms = (time.monotonic() - start) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                span.set_attribute("llm.prompt_tokens", result.usage.prompt_tokens)
                span.set_attribute("llm.completion_tokens", result.usage.completion_tokens)
                span.set_attribute("llm.finish_reason", result.finish_reason.value)
                collector.increment(
                    "llm_requests_total",
                    labels={"provider": "anthropic", "model": result.model, "status": "success", "error_type": ""},
                )
                collector.observe(
                    "llm_request_duration_seconds",
                    duration_ms / 1000,
                    labels={"provider": "anthropic", "model": result.model},
                )
                collector.observe(
                    "llm_prompt_tokens",
                    result.usage.prompt_tokens,
                    labels={"provider": "anthropic", "model": result.model},
                )
                collector.observe(
                    "llm_completion_tokens",
                    result.usage.completion_tokens,
                    labels={"provider": "anthropic", "model": result.model},
                )
                logger.info(
                    "Anthropic chat completed: model=%s, tokens=%d/%d (%.1fms)",
                    result.model,
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    duration_ms,
                    extra={
                        "llm_data": {
                            "event": "chat_completed",
                            "provider": "anthropic",
                            "model": result.model,
                            "prompt_tokens": result.usage.prompt_tokens,
                            "completion_tokens": result.usage.completion_tokens,
                            "total_tokens": result.usage.total_tokens,
                            "finish_reason": result.finish_reason.value,
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
                return result

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = await self.chat(
            [Message(role="user", content=prompt)],
            **kwargs,
        )
        return CompletionResponse(
            text=response.message.content or "",
            usage=response.usage,
            model=response.model,
            finish_reason=response.finish_reason,
        )

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        # Copy kwargs to avoid mutation.
        kw = dict(kwargs)
        model = kw.pop("model", self.settings.default_model)
        collector = get_metrics_collector()
        start = time.monotonic()

        collector.increment(
            "llm_stream_requests_total",
            labels={"provider": "anthropic", "model": model},
        )

        system_text, anthropic_messages = _extract_system_and_messages(messages)

        create_kwargs: dict[str, Any] = {
            "max_tokens": kw.pop("max_tokens", self.settings.max_tokens),
            "temperature": kw.pop("temperature", self.settings.temperature),
        }
        if system_text is not None:
            create_kwargs["system"] = system_text
        # Remove 'stream' if caller passed it; we control streaming here.
        kw.pop("stream", None)
        create_kwargs.update(kw)

        # F-005: Wrap stream connection in retry + circuit breaker.
        async def _open_stream() -> Any:
            try:
                return self._client.messages.stream(
                    model=model,
                    messages=anthropic_messages,  # type: ignore[arg-type]
                    **create_kwargs,
                )
            except anthropic.AnthropicError as exc:
                raise _map_anthropic_error(exc) from exc

        stream_cm = await self._circuit_breaker.call(
            with_retry,
            _open_stream,
            retry_settings=self.settings.retry,
        )

        async def _iterate() -> AsyncIterator[StreamChunk]:
            collected = ""
            input_tokens = 0
            output_tokens = 0
            try:
                async with stream_cm as stream:
                    async for event in stream:
                        chunk, collected, input_tokens, output_tokens = _process_stream_event(
                            event,
                            collected,
                            input_tokens,
                            output_tokens,
                        )
                        if chunk is not None:
                            yield chunk
            except anthropic.AnthropicError as exc:
                collector.increment(
                    "llm_stream_errors_total",
                    labels={"provider": "anthropic", "model": model},
                )
                raise LLMStreamError(str(exc), partial_content=collected) from exc

        # F-006: Apply bounded backpressure buffering.
        async for chunk in buffered_stream(_iterate()):
            yield chunk

        duration_ms = (time.monotonic() - start) * 1000
        collector.observe(
            "llm_stream_duration_seconds", duration_ms / 1000, labels={"provider": "anthropic", "model": model}
        )
        logger.info(
            "Anthropic stream completed: model=%s (%.1fms)",
            model,
            duration_ms,
            extra={
                "llm_data": {
                    "event": "stream_completed",
                    "provider": "anthropic",
                    "model": model,
                    "duration_ms": round(duration_ms, 1),
                }
            },
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:  # noqa: ARG002
        # Fallback heuristic -- Anthropic does not expose a public tokenizer
        # that can be called synchronously without API overhead.
        return max(1, len(text) // 4)

    async def health_check(self) -> HealthStatus:
        collector = get_metrics_collector()
        start = time.monotonic()
        try:
            # F-009: Use count_tokens endpoint to verify connectivity without
            # incurring generation token costs.
            await self._client.messages.count_tokens(
                model=self.settings.default_model,
                messages=[{"role": "user", "content": "ping"}],
            )
            elapsed = (time.monotonic() - start) * 1000
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "anthropic", "status": "healthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "anthropic"})
            return HealthStatus(status="healthy", latency_ms=elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.monotonic() - start) * 1000
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "anthropic", "status": "unhealthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "anthropic"})
            return HealthStatus(
                status="unhealthy",
                message=str(exc),
                latency_ms=elapsed,
            )

    async def close(self) -> None:
        await self._client.close()
