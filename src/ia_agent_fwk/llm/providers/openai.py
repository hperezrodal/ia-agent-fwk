"""OpenAI provider implementation.

Uses the official ``openai`` Python SDK (v1+).  SDK-level retries are
disabled (``max_retries=0``) in favour of the framework retry layer.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import openai
import tiktoken

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


def _map_finish_reason(reason: str | None) -> FinishReason:
    mapping: dict[str | None, FinishReason] = {
        "stop": FinishReason.stop,
        "tool_calls": FinishReason.tool_calls,
        "function_call": FinishReason.tool_calls,
        "length": FinishReason.length,
        "content_filter": FinishReason.content_filter,
    }
    if reason is not None and reason not in mapping:
        logger.warning("Unmapped OpenAI finish_reason '%s'; treating as error", reason)
        return FinishReason.error
    return mapping.get(reason, FinishReason.stop)


def _map_openai_error(exc: openai.OpenAIError) -> LLMProviderError:
    """Convert an OpenAI SDK error to a framework exception."""
    if isinstance(exc, openai.AuthenticationError):
        return LLMAuthenticationError(str(exc))
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError(str(exc))
    return LLMProviderError(str(exc))


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider (GPT-4o, GPT-4, GPT-3.5-turbo, ...)."""

    def __init__(self, settings: LLMProviderSettings, provider_name: str, **_kwargs: Any) -> None:
        super().__init__(settings, provider_name)
        api_key = settings.api_key.get_secret_value()
        # F-008: Validate API key availability early.
        if not api_key and not os.environ.get("OPENAI_API_KEY"):
            msg = (
                "OpenAI API key is not configured. "
                "Set 'api_key' in provider settings or the OPENAI_API_KEY environment variable."
            )
            raise LLMConfigError(msg)
        client_kwargs: dict[str, Any] = {"max_retries": 0, "timeout": float(settings.timeout)}
        if api_key:
            client_kwargs["api_key"] = api_key
        if settings.base_url:
            client_kwargs["base_url"] = settings.base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._circuit_breaker = CircuitBreaker(
            provider_name=provider_name,
            settings=settings.circuit_breaker,
        )

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        collector = get_metrics_collector()
        model = kwargs.get("model", self.settings.default_model)
        start = time.monotonic()

        async def _do_chat() -> ChatResponse:
            # Copy kwargs to avoid mutating the outer dict on retry (F-002).
            kw = dict(kwargs)
            try:
                response = await self._client.chat.completions.create(
                    model=kw.pop("model", self.settings.default_model),
                    messages=[self._to_openai_message(m) for m in messages],  # type: ignore[misc]
                    temperature=kw.pop("temperature", self.settings.temperature),
                    max_tokens=kw.pop("max_tokens", self.settings.max_tokens),
                    **kw,
                )
            except openai.OpenAIError as exc:
                raise _map_openai_error(exc) from exc

            choice = response.choices[0]
            tool_calls: list[ToolCall] | None = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,  # type: ignore[union-attr]
                        arguments=tc.function.arguments,  # type: ignore[union-attr]
                    )
                    for tc in choice.message.tool_calls
                ]

            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=choice.message.content,
                    tool_calls=tool_calls,
                ),
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                ),
                model=response.model,
                finish_reason=_map_finish_reason(choice.finish_reason),
            )

        with _tracer.start_as_current_span(
            "llm.chat",
            attributes={"llm.provider": "openai", "llm.model": model},
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
                    labels={"provider": "openai", "model": model, "status": "error", "error_type": type(exc).__name__},
                )
                collector.observe(
                    "llm_request_duration_seconds", duration_ms / 1000, labels={"provider": "openai", "model": model}
                )
                logger.exception(
                    "OpenAI chat failed: model=%s (%.1fms)",
                    model,
                    duration_ms,
                    extra={
                        "llm_data": {
                            "event": "chat_error",
                            "provider": "openai",
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
                    labels={"provider": "openai", "model": result.model, "status": "success", "error_type": ""},
                )
                collector.observe(
                    "llm_request_duration_seconds",
                    duration_ms / 1000,
                    labels={"provider": "openai", "model": result.model},
                )
                collector.observe(
                    "llm_prompt_tokens",
                    result.usage.prompt_tokens,
                    labels={"provider": "openai", "model": result.model},
                )
                collector.observe(
                    "llm_completion_tokens",
                    result.usage.completion_tokens,
                    labels={"provider": "openai", "model": result.model},
                )
                logger.info(
                    "OpenAI chat completed: model=%s, tokens=%d/%d (%.1fms)",
                    result.model,
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    duration_ms,
                    extra={
                        "llm_data": {
                            "event": "chat_completed",
                            "provider": "openai",
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
        collector = get_metrics_collector()
        model = kw.get("model", self.settings.default_model)
        start = time.monotonic()

        collector.increment(
            "llm_stream_requests_total",
            labels={"provider": "openai", "model": model},
        )

        async def _open_stream() -> Any:
            try:
                return await self._client.chat.completions.create(  # type: ignore[call-overload]
                    model=kw.pop("model", self.settings.default_model),
                    messages=[self._to_openai_message(m) for m in messages],
                    temperature=kw.pop("temperature", self.settings.temperature),
                    max_tokens=kw.pop("max_tokens", self.settings.max_tokens),
                    stream=True,
                    stream_options={"include_usage": True},
                    **kw,
                )
            except openai.OpenAIError as exc:
                raise _map_openai_error(exc) from exc

        # F-005: Wrap stream connection in retry + circuit breaker.
        response = await self._circuit_breaker.call(
            with_retry,
            _open_stream,
            retry_settings=self.settings.retry,
        )

        async def _iterate() -> AsyncIterator[StreamChunk]:
            collected = ""
            try:
                async for chunk in response:
                    if not chunk.choices:
                        # Final chunk with usage only.
                        usage: TokenUsage | None = None
                        if chunk.usage:
                            usage = TokenUsage(
                                prompt_tokens=chunk.usage.prompt_tokens,
                                completion_tokens=chunk.usage.completion_tokens,
                                total_tokens=chunk.usage.total_tokens,
                            )
                        yield StreamChunk(
                            delta="",
                            finish_reason=FinishReason.stop,
                            usage=usage,
                        )
                        continue

                    delta = chunk.choices[0].delta
                    delta_text = delta.content or ""
                    collected += delta_text
                    finish = (
                        _map_finish_reason(chunk.choices[0].finish_reason) if chunk.choices[0].finish_reason else None
                    )

                    yield StreamChunk(
                        delta=delta_text,
                        finish_reason=finish,
                    )
            except openai.OpenAIError as exc:
                collector.increment(
                    "llm_stream_errors_total",
                    labels={"provider": "openai", "model": model},
                )
                raise LLMStreamError(str(exc), partial_content=collected) from exc

        # F-006: Apply bounded backpressure buffering.
        async for chunk in buffered_stream(_iterate()):
            yield chunk

        duration_ms = (time.monotonic() - start) * 1000
        collector.observe(
            "llm_stream_duration_seconds", duration_ms / 1000, labels={"provider": "openai", "model": model}
        )
        logger.info(
            "OpenAI stream completed: model=%s (%.1fms)",
            model,
            duration_ms,
            extra={
                "llm_data": {
                    "event": "stream_completed",
                    "provider": "openai",
                    "model": model,
                    "duration_ms": round(duration_ms, 1),
                }
            },
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        model_name = model or self.settings.default_model
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    async def health_check(self) -> HealthStatus:
        collector = get_metrics_collector()
        start = time.monotonic()
        try:
            await self._client.models.list()
            elapsed = (time.monotonic() - start) * 1000
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "openai", "status": "healthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "openai"})
            return HealthStatus(status="healthy", latency_ms=elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.monotonic() - start) * 1000
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "openai", "status": "unhealthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "openai"})
            return HealthStatus(
                status="unhealthy",
                message=str(exc),
                latency_ms=elapsed,
            )

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_message(msg: Message) -> dict[str, Any]:
        m: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            m["content"] = msg.content
        if msg.tool_calls:
            m["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id is not None:
            m["tool_call_id"] = msg.tool_call_id
        return m
