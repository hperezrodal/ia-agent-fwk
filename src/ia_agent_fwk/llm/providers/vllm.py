"""vLLM provider implementation using raw ``httpx``.

Communicates with the vLLM OpenAI-compatible API (``/v1/completions``,
``/v1/chat/completions``, ``/v1/models``).  No vLLM-specific SDK is required.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:  # pragma: no cover
    _HAS_TIKTOKEN = False

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


def _map_finish_reason(reason: str | None) -> FinishReason:
    """Map an OpenAI-compatible finish_reason string to ``FinishReason``."""
    mapping: dict[str, FinishReason] = {
        "stop": FinishReason.stop,
        "length": FinishReason.length,
        "tool_calls": FinishReason.tool_calls,
        "content_filter": FinishReason.content_filter,
    }
    return mapping.get(reason or "", FinishReason.stop)


class VLLMProvider(LLMProvider):
    """vLLM LLM provider (OpenAI-compatible API)."""

    def __init__(self, settings: LLMProviderSettings, provider_name: str, **_kwargs: Any) -> None:
        super().__init__(settings, provider_name)
        base_url = settings.base_url or "http://localhost:8000/v1"
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
            # Copy kwargs to avoid mutating the outer dict on retry.
            kw = dict(kwargs)
            model = kw.pop("model", self.settings.default_model)
            payload: dict[str, Any] = {
                "model": model,
                "messages": [self._to_openai_message(m) for m in messages],
                "temperature": kw.pop("temperature", self.settings.temperature),
                "max_tokens": kw.pop("max_tokens", self.settings.max_tokens),
            }
            payload.update(kw)
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(str(exc)) from exc
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(str(exc)) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError(str(exc)) from exc

            data: dict[str, Any] = resp.json()
            choice: dict[str, Any] = data.get("choices", [{}])[0]
            msg_data: dict[str, Any] = choice.get("message", {})

            # Parse tool calls if present.
            tool_calls: list[ToolCall] | None = None
            raw_tool_calls: list[dict[str, Any]] | None = msg_data.get("tool_calls")
            if raw_tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", str(i)),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=json.dumps(tc.get("function", {}).get("arguments", {}))
                        if isinstance(tc.get("function", {}).get("arguments"), dict)
                        else tc.get("function", {}).get("arguments", "{}"),
                    )
                    for i, tc in enumerate(raw_tool_calls)
                ]

            finish_reason_str: str | None = choice.get("finish_reason")
            finish = FinishReason.tool_calls if tool_calls else _map_finish_reason(finish_reason_str)

            usage_data: dict[str, Any] = data.get("usage", {})
            prompt_tokens = usage_data.get("prompt_tokens", 0) or 0
            completion_tokens = usage_data.get("completion_tokens", 0) or 0

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
            attributes={"llm.provider": "vllm", "llm.model": model},
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
                    labels={"provider": "vllm", "model": model, "status": "error", "error_type": type(exc).__name__},
                )
                collector.observe(
                    "llm_request_duration_seconds", duration_ms / 1000, labels={"provider": "vllm", "model": model}
                )
                raise
            else:
                duration_ms = (time.monotonic() - start) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                span.set_attribute("llm.prompt_tokens", result.usage.prompt_tokens)
                span.set_attribute("llm.completion_tokens", result.usage.completion_tokens)
                collector.increment(
                    "llm_requests_total",
                    labels={"provider": "vllm", "model": result.model, "status": "success", "error_type": ""},
                )
                collector.observe(
                    "llm_request_duration_seconds",
                    duration_ms / 1000,
                    labels={"provider": "vllm", "model": result.model},
                )
                collector.observe(
                    "llm_prompt_tokens", result.usage.prompt_tokens, labels={"provider": "vllm", "model": result.model}
                )
                collector.observe(
                    "llm_completion_tokens",
                    result.usage.completion_tokens,
                    labels={"provider": "vllm", "model": result.model},
                )
                return result  # type: ignore[no-any-return]

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        collector = get_metrics_collector()
        model = kwargs.get("model", self.settings.default_model)
        start = time.monotonic()

        async def _do_complete() -> CompletionResponse:
            # Copy kwargs to avoid mutating the outer dict on retry.
            kw = dict(kwargs)
            model = kw.pop("model", self.settings.default_model)
            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "temperature": kw.pop("temperature", self.settings.temperature),
                "max_tokens": kw.pop("max_tokens", self.settings.max_tokens),
            }
            payload.update(kw)
            try:
                resp = await self._client.post("/completions", json=payload)
                resp.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(str(exc)) from exc
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(str(exc)) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError(str(exc)) from exc

            data: dict[str, Any] = resp.json()
            choice: dict[str, Any] = data.get("choices", [{}])[0]

            usage_data: dict[str, Any] = data.get("usage", {})
            prompt_tokens = usage_data.get("prompt_tokens", 0) or 0
            completion_tokens = usage_data.get("completion_tokens", 0) or 0

            return CompletionResponse(
                text=choice.get("text", ""),
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
                model=data.get("model", model),
                finish_reason=_map_finish_reason(choice.get("finish_reason")),
            )

        with _tracer.start_as_current_span(
            "llm.complete",
            attributes={"llm.provider": "vllm", "llm.model": model},
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
                    labels={"provider": "vllm", "model": model, "status": "error", "error_type": type(exc).__name__},
                )
                collector.observe(
                    "llm_request_duration_seconds", duration_ms / 1000, labels={"provider": "vllm", "model": model}
                )
                raise
            else:
                duration_ms = (time.monotonic() - start) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                span.set_attribute("llm.prompt_tokens", result.usage.prompt_tokens)
                span.set_attribute("llm.completion_tokens", result.usage.completion_tokens)
                collector.increment(
                    "llm_requests_total",
                    labels={"provider": "vllm", "model": result.model, "status": "success", "error_type": ""},
                )
                collector.observe(
                    "llm_request_duration_seconds",
                    duration_ms / 1000,
                    labels={"provider": "vllm", "model": result.model},
                )
                collector.observe(
                    "llm_prompt_tokens", result.usage.prompt_tokens, labels={"provider": "vllm", "model": result.model}
                )
                collector.observe(
                    "llm_completion_tokens",
                    result.usage.completion_tokens,
                    labels={"provider": "vllm", "model": result.model},
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
            labels={"provider": "vllm", "model": model},
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._to_openai_message(m) for m in messages],
            "temperature": kw.pop("temperature", self.settings.temperature),
            "max_tokens": kw.pop("max_tokens", self.settings.max_tokens),
            "stream": True,
        }
        payload.update(kw)

        async def _open_stream() -> httpx.Response:
            try:
                resp = await self._client.send(
                    self._client.build_request("POST", "/chat/completions", json=payload),
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
                    chunk = self._parse_sse_line(line)
                    if chunk is None:
                        continue
                    collected += chunk.delta
                    yield chunk
            except httpx.HTTPError as exc:
                collector.increment(
                    "llm_stream_errors_total",
                    labels={"provider": "vllm", "model": model},
                )
                raise LLMStreamError(str(exc), partial_content=collected) from exc
            finally:
                await resp.aclose()

        async for chunk in buffered_stream(_iterate()):
            yield chunk

        duration_ms = (time.monotonic() - start) * 1000
        collector.observe(
            "llm_stream_duration_seconds", duration_ms / 1000, labels={"provider": "vllm", "model": model}
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Approximate token count using tiktoken (falls back to heuristic)."""
        if _HAS_TIKTOKEN:
            try:
                enc = tiktoken.encoding_for_model(model or self.settings.default_model)
                return len(enc.encode(text))
            except KeyError:
                # Model not recognized by tiktoken; fall back to cl100k_base.
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
        # Heuristic: ~4 chars per token.
        return max(1, len(text) // 4)

    async def health_check(self) -> HealthStatus:
        collector = get_metrics_collector()
        start = time.monotonic()
        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
            elapsed = (time.monotonic() - start) * 1000

            data: dict[str, Any] = resp.json()
            models: list[str] = [m.get("id", "") for m in data.get("data", [])]
            model_available = any(self.settings.default_model in m for m in models)
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "vllm", "status": "healthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "vllm"})
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
                labels={"provider": "vllm", "status": "unhealthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "vllm"})
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
    def _parse_sse_line(line: str) -> StreamChunk | None:
        """Parse a single SSE line into a ``StreamChunk`` or ``None``."""
        line = line.strip()
        if not line or line == "data: [DONE]":
            return None
        line = line.removeprefix("data: ")
        data: dict[str, Any] = json.loads(line)
        choice: dict[str, Any] = data.get("choices", [{}])[0]
        delta: dict[str, Any] = choice.get("delta", {})
        delta_text: str = delta.get("content", "") or ""
        finish_reason_str: str | None = choice.get("finish_reason")

        usage: TokenUsage | None = None
        usage_data: dict[str, Any] | None = data.get("usage")
        if usage_data:
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0) or 0,
                completion_tokens=usage_data.get("completion_tokens", 0) or 0,
            )

        return StreamChunk(
            delta=delta_text,
            finish_reason=_map_finish_reason(finish_reason_str) if finish_reason_str else None,
            usage=usage,
        )

    @staticmethod
    def _to_openai_message(msg: Message) -> dict[str, Any]:
        m: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
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
