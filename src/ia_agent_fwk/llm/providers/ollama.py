"""Ollama provider implementation using raw ``httpx``.

Communicates with the Ollama REST API (``/api/chat``, ``/api/generate``,
``/api/tags``).  No Ollama-specific SDK is required.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ia_agent_fwk.config.settings import LLMProviderSettings
from ia_agent_fwk.llm.base import LLMProvider
from ia_agent_fwk.llm.circuit_breaker import CircuitBreaker
from ia_agent_fwk.llm.exceptions import (
    LLMProviderError,
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


def _map_finish_reason(*, done: bool) -> FinishReason:
    return FinishReason.stop if done else FinishReason.length


class OllamaProvider(LLMProvider):
    """Ollama LLM provider (llama3, mistral, any GGUF model, ...)."""

    def __init__(self, settings: LLMProviderSettings, provider_name: str, **_kwargs: Any) -> None:
        super().__init__(settings, provider_name)
        base_url = settings.base_url or "http://localhost:11434"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(float(settings.timeout)),
        )
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
            # Copy kwargs to avoid mutating the outer dict on retry (F-004).
            kw = dict(kwargs)
            model = kw.pop("model", self.settings.default_model)
            payload: dict[str, Any] = {
                "model": model,
                "messages": [self._to_ollama_message(m) for m in messages],
                "stream": False,
                "options": {
                    "temperature": kw.pop("temperature", self.settings.temperature),
                    "num_predict": kw.pop("max_tokens", self.settings.max_tokens),
                },
            }
            payload.update(kw)
            try:
                resp = await self._client.post("/api/chat", json=payload)
                resp.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(str(exc)) from exc
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(str(exc)) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError(str(exc)) from exc

            data: dict[str, Any] = resp.json()
            msg_data: dict[str, Any] = data.get("message", {})

            # Parse tool calls if present.
            tool_calls: list[ToolCall] | None = None
            raw_tool_calls: list[dict[str, Any]] | None = msg_data.get("tool_calls")
            if raw_tool_calls:
                tool_calls = [
                    ToolCall(
                        id=str(i),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=json.dumps(tc.get("function", {}).get("arguments", {})),
                    )
                    for i, tc in enumerate(raw_tool_calls)
                ]

            finish = FinishReason.tool_calls if tool_calls else _map_finish_reason(done=data.get("done", True))

            usage_data: dict[str, Any] = data.get("usage", {})
            prompt_tokens = data.get("prompt_eval_count", usage_data.get("prompt_tokens", 0)) or 0
            completion_tokens = data.get("eval_count", usage_data.get("completion_tokens", 0)) or 0

            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=msg_data.get("content", ""),
                    tool_calls=tool_calls,
                ),
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
                model=data.get("model", model),
                finish_reason=finish,
            )

        with _tracer.start_as_current_span(
            "llm.chat",
            attributes={"llm.provider": "ollama", "llm.model": model},
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
                    labels={"provider": "ollama", "model": model, "status": "error", "error_type": type(exc).__name__},
                )
                collector.observe(
                    "llm_request_duration_seconds", duration_ms / 1000, labels={"provider": "ollama", "model": model}
                )
                logger.exception(
                    "Ollama chat failed: model=%s (%.1fms)",
                    model,
                    duration_ms,
                    extra={
                        "llm_data": {
                            "event": "chat_error",
                            "provider": "ollama",
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
                    labels={"provider": "ollama", "model": result.model, "status": "success", "error_type": ""},
                )
                collector.observe(
                    "llm_request_duration_seconds",
                    duration_ms / 1000,
                    labels={"provider": "ollama", "model": result.model},
                )
                collector.observe(
                    "llm_prompt_tokens",
                    result.usage.prompt_tokens,
                    labels={"provider": "ollama", "model": result.model},
                )
                collector.observe(
                    "llm_completion_tokens",
                    result.usage.completion_tokens,
                    labels={"provider": "ollama", "model": result.model},
                )
                logger.info(
                    "Ollama chat completed: model=%s, tokens=%d/%d (%.1fms)",
                    result.model,
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    duration_ms,
                    extra={
                        "llm_data": {
                            "event": "chat_completed",
                            "provider": "ollama",
                            "model": result.model,
                            "prompt_tokens": result.usage.prompt_tokens,
                            "completion_tokens": result.usage.completion_tokens,
                            "total_tokens": result.usage.total_tokens,
                            "finish_reason": result.finish_reason.value,
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
                return result  # type: ignore[no-any-return]

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        collector = get_metrics_collector()
        model = kwargs.get("model", self.settings.default_model)
        start = time.monotonic()

        async def _do_complete() -> CompletionResponse:
            # Copy kwargs to avoid mutating the outer dict on retry (F-004).
            kw = dict(kwargs)
            model = kw.pop("model", self.settings.default_model)
            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kw.pop("temperature", self.settings.temperature),
                    "num_predict": kw.pop("max_tokens", self.settings.max_tokens),
                },
            }
            payload.update(kw)
            try:
                resp = await self._client.post("/api/generate", json=payload)
                resp.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(str(exc)) from exc
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(str(exc)) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError(str(exc)) from exc

            data: dict[str, Any] = resp.json()

            prompt_tokens = data.get("prompt_eval_count", 0) or 0
            completion_tokens = data.get("eval_count", 0) or 0

            return CompletionResponse(
                text=data.get("response", ""),
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
                model=data.get("model", model),
                finish_reason=_map_finish_reason(done=data.get("done", True)),
            )

        with _tracer.start_as_current_span(
            "llm.complete",
            attributes={"llm.provider": "ollama", "llm.model": model},
        ) as span:
            try:
                result = await self._circuit_breaker.call(
                    with_retry,
                    _do_complete,
                    retry_settings=self.settings.retry,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.record_exception(exc)
                collector.increment(
                    "llm_requests_total",
                    labels={"provider": "ollama", "model": model, "status": "error", "error_type": type(exc).__name__},
                )
                collector.observe(
                    "llm_request_duration_seconds", duration_ms / 1000, labels={"provider": "ollama", "model": model}
                )
                raise
            else:
                duration_ms = (time.monotonic() - start) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                span.set_attribute("llm.prompt_tokens", result.usage.prompt_tokens)
                span.set_attribute("llm.completion_tokens", result.usage.completion_tokens)
                collector.increment(
                    "llm_requests_total",
                    labels={"provider": "ollama", "model": result.model, "status": "success", "error_type": ""},
                )
                collector.observe(
                    "llm_request_duration_seconds",
                    duration_ms / 1000,
                    labels={"provider": "ollama", "model": result.model},
                )
                collector.observe(
                    "llm_prompt_tokens",
                    result.usage.prompt_tokens,
                    labels={"provider": "ollama", "model": result.model},
                )
                collector.observe(
                    "llm_completion_tokens",
                    result.usage.completion_tokens,
                    labels={"provider": "ollama", "model": result.model},
                )
                return result  # type: ignore[no-any-return]

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        # Copy kwargs to avoid mutation.
        kw = dict(kwargs)
        model = kw.pop("model", self.settings.default_model)
        collector = get_metrics_collector()
        start = time.monotonic()

        collector.increment(
            "llm_stream_requests_total",
            labels={"provider": "ollama", "model": model},
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._to_ollama_message(m) for m in messages],
            "stream": True,
            "options": {
                "temperature": kw.pop("temperature", self.settings.temperature),
                "num_predict": kw.pop("max_tokens", self.settings.max_tokens),
            },
        }
        payload.update(kw)

        # F-005: Wrap stream connection in retry + circuit breaker.
        async def _open_stream() -> httpx.Response:
            try:
                resp = await self._client.send(
                    self._client.build_request("POST", "/api/chat", json=payload),
                    stream=True,
                )
                resp.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(str(exc)) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError(str(exc)) from exc
            return resp

        resp = await self._circuit_breaker.call(
            with_retry,
            _open_stream,
            retry_settings=self.settings.retry,
        )

        async def _iterate() -> AsyncIterator[StreamChunk]:
            collected = ""
            try:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data: dict[str, Any] = json.loads(line)
                    msg_data: dict[str, Any] = data.get("message", {})
                    delta_text: str = msg_data.get("content", "")
                    collected += delta_text
                    done: bool = data.get("done", False)

                    usage: TokenUsage | None = None
                    if done:
                        prompt_tokens = data.get("prompt_eval_count", 0) or 0
                        completion_tokens = data.get("eval_count", 0) or 0
                        usage = TokenUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )

                    yield StreamChunk(
                        delta=delta_text,
                        finish_reason=FinishReason.stop if done else None,
                        usage=usage,
                    )
            except httpx.HTTPError as exc:
                collector.increment(
                    "llm_stream_errors_total",
                    labels={"provider": "ollama", "model": model},
                )
                raise LLMStreamError(str(exc), partial_content=collected) from exc
            finally:
                await resp.aclose()

        # F-006: Apply bounded backpressure buffering.
        async for chunk in buffered_stream(_iterate()):
            yield chunk

        duration_ms = (time.monotonic() - start) * 1000
        collector.observe(
            "llm_stream_duration_seconds", duration_ms / 1000, labels={"provider": "ollama", "model": model}
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:  # noqa: ARG002
        # Ollama does not have a synchronous tokenize; use a heuristic.
        return max(1, len(text) // 4)

    async def health_check(self) -> HealthStatus:
        collector = get_metrics_collector()
        start = time.monotonic()
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            elapsed = (time.monotonic() - start) * 1000

            data: dict[str, Any] = resp.json()
            models: list[str] = [m.get("name", "") for m in data.get("models", [])]
            model_available = any(self.settings.default_model in m for m in models)
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "ollama", "status": "healthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "ollama"})
            if not model_available and models:
                return HealthStatus(
                    status="healthy",
                    message=(
                        f"Connected but model '{self.settings.default_model}' not found. Available: {', '.join(models)}"
                    ),
                    latency_ms=elapsed,
                )

            return HealthStatus(status="healthy", latency_ms=elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.monotonic() - start) * 1000
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "ollama", "status": "unhealthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "ollama"})
            return HealthStatus(
                status="unhealthy",
                message=str(exc),
                latency_ms=elapsed,
            )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ollama_message(msg: Message) -> dict[str, Any]:
        m: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
        # F-016: Forward tool_calls for assistant messages.
        if msg.tool_calls:
            m["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": json.loads(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        # Forward tool_call_id for tool-result messages.
        if msg.tool_call_id is not None:
            m["tool_call_id"] = msg.tool_call_id
        return m
