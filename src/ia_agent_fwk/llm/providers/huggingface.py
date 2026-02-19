"""HuggingFace local provider implementation using the ``transformers`` library.

Runs models locally on CPU or GPU via the HuggingFace ``transformers`` pipeline.
The ``transformers``, ``torch``, and ``accelerate`` packages are optional
dependencies and are imported lazily with a try/except guard.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

try:
    import torch  # noqa: F401
    import transformers

    _HAS_TRANSFORMERS = True
except ImportError:
    transformers = None
    _HAS_TRANSFORMERS = False

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ia_agent_fwk.config.settings import LLMProviderSettings
from ia_agent_fwk.llm.base import LLMProvider
from ia_agent_fwk.llm.exceptions import (
    LLMConfigError,
    LLMProviderError,
)
from ia_agent_fwk.llm.models import (
    ChatResponse,
    CompletionResponse,
    FinishReason,
    HealthStatus,
    Message,
    StreamChunk,
    TokenUsage,
)
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

_MISSING_DEPS_MSG = (
    "HuggingFace provider requires 'transformers' and 'torch'. Install them with: pip install ia_agent_fwk[huggingface]"
)

_STREAM_NOT_SUPPORTED_MSG = "HuggingFaceProvider does not support streaming. Use chat() or complete() instead."


class HuggingFaceProvider(LLMProvider):
    """HuggingFace local LLM provider (transformers pipeline)."""

    def __init__(self, settings: LLMProviderSettings, provider_name: str, **_kwargs: Any) -> None:
        super().__init__(settings, provider_name)
        if not _HAS_TRANSFORMERS:
            raise LLMConfigError(_MISSING_DEPS_MSG)

        self._model_name = settings.default_model
        self._device = settings.base_url or "cpu"  # Reuse base_url field for device config.
        self._max_tokens = settings.max_tokens
        self._temperature = settings.temperature

        # Lazy-load model and tokenizer.
        self._pipeline: Any | None = None
        self._tokenizer: Any | None = None

    # ------------------------------------------------------------------
    # Internal: lazy pipeline initialization
    # ------------------------------------------------------------------

    def _ensure_pipeline(self) -> Any:
        """Lazily initialize the transformers pipeline."""
        if self._pipeline is None:
            try:
                self._pipeline = transformers.pipeline(
                    "text-generation",
                    model=self._model_name,
                    device_map=self._device if self._device != "cpu" else None,
                    torch_dtype="auto",
                )
                self._tokenizer = self._pipeline.tokenizer
            except Exception as exc:
                msg = f"Failed to load model '{self._model_name}': {exc}"
                raise LLMProviderError(msg) from exc
        return self._pipeline

    def _ensure_tokenizer(self) -> Any:
        """Lazily initialize the tokenizer (without loading the full model)."""
        if self._tokenizer is None:
            try:
                self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self._model_name,
                )
            except Exception as exc:
                msg = f"Failed to load tokenizer for '{self._model_name}': {exc}"
                raise LLMProviderError(msg) from exc
        return self._tokenizer

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    async def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        collector = get_metrics_collector()
        model = self._model_name
        start = time.monotonic()

        kw = dict(kwargs)
        max_tokens = kw.pop("max_tokens", self._max_tokens)
        temperature = kw.pop("temperature", self._temperature)

        pipe = self._ensure_pipeline()
        chat_messages = [{"role": m.role, "content": m.content or ""} for m in messages]

        with _tracer.start_as_current_span(
            "llm.chat",
            attributes={"llm.provider": "huggingface", "llm.model": model},
        ) as span:
            try:
                outputs = pipe(
                    chat_messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    return_full_text=False,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.record_exception(exc)
                collector.increment(
                    "llm_requests_total",
                    labels={
                        "provider": "huggingface",
                        "model": model,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    },
                )
                collector.observe(
                    "llm_request_duration_seconds",
                    duration_ms / 1000,
                    labels={"provider": "huggingface", "model": model},
                )
                msg = f"HuggingFace chat error: {exc}"
                raise LLMProviderError(msg) from exc

            generated_text: str = outputs[0]["generated_text"]

            # Count tokens for usage reporting.
            prompt_text = " ".join(m.content or "" for m in messages)
            prompt_tokens = self.count_tokens(prompt_text)
            completion_tokens = self.count_tokens(generated_text)

            duration_ms = (time.monotonic() - start) * 1000
            span.set_attribute("llm.duration_ms", duration_ms)
            span.set_attribute("llm.prompt_tokens", prompt_tokens)
            span.set_attribute("llm.completion_tokens", completion_tokens)
            collector.increment(
                "llm_requests_total",
                labels={"provider": "huggingface", "model": model, "status": "success", "error_type": ""},
            )
            collector.observe(
                "llm_request_duration_seconds", duration_ms / 1000, labels={"provider": "huggingface", "model": model}
            )
            collector.observe("llm_prompt_tokens", prompt_tokens, labels={"provider": "huggingface", "model": model})
            collector.observe(
                "llm_completion_tokens", completion_tokens, labels={"provider": "huggingface", "model": model}
            )

            return ChatResponse(
                message=Message(role="assistant", content=generated_text),
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
                model=self._model_name,
                finish_reason=FinishReason.stop,
            )

    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        collector = get_metrics_collector()
        model = self._model_name
        start = time.monotonic()

        kw = dict(kwargs)
        max_tokens = kw.pop("max_tokens", self._max_tokens)
        temperature = kw.pop("temperature", self._temperature)

        pipe = self._ensure_pipeline()

        with _tracer.start_as_current_span(
            "llm.complete",
            attributes={"llm.provider": "huggingface", "llm.model": model},
        ) as span:
            try:
                outputs = pipe(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    return_full_text=False,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.record_exception(exc)
                collector.increment(
                    "llm_requests_total",
                    labels={
                        "provider": "huggingface",
                        "model": model,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    },
                )
                collector.observe(
                    "llm_request_duration_seconds",
                    duration_ms / 1000,
                    labels={"provider": "huggingface", "model": model},
                )
                msg = f"HuggingFace completion error: {exc}"
                raise LLMProviderError(msg) from exc

            generated_text: str = outputs[0]["generated_text"]

            prompt_tokens = self.count_tokens(prompt)
            completion_tokens = self.count_tokens(generated_text)

            duration_ms = (time.monotonic() - start) * 1000
            span.set_attribute("llm.duration_ms", duration_ms)
            span.set_attribute("llm.prompt_tokens", prompt_tokens)
            span.set_attribute("llm.completion_tokens", completion_tokens)
            collector.increment(
                "llm_requests_total",
                labels={"provider": "huggingface", "model": model, "status": "success", "error_type": ""},
            )
            collector.observe(
                "llm_request_duration_seconds", duration_ms / 1000, labels={"provider": "huggingface", "model": model}
            )
            collector.observe("llm_prompt_tokens", prompt_tokens, labels={"provider": "huggingface", "model": model})
            collector.observe(
                "llm_completion_tokens", completion_tokens, labels={"provider": "huggingface", "model": model}
            )

            return CompletionResponse(
                text=generated_text,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
                model=self._model_name,
                finish_reason=FinishReason.stop,
            )

    async def stream(
        self,
        messages: list[Message],  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError(_STREAM_NOT_SUPPORTED_MSG)
        yield  # pragma: no cover

    def count_tokens(self, text: str, model: str | None = None) -> int:  # noqa: ARG002
        """Count tokens using the model's tokenizer."""
        tokenizer = self._ensure_tokenizer()
        tokens: list[int] = tokenizer.encode(text)
        return len(tokens)

    async def health_check(self) -> HealthStatus:
        collector = get_metrics_collector()
        start = time.monotonic()
        try:
            self._ensure_pipeline()
            elapsed = (time.monotonic() - start) * 1000
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "huggingface", "status": "healthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "huggingface"})
            return HealthStatus(status="healthy", latency_ms=elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.monotonic() - start) * 1000
            collector.increment(
                "llm_health_checks_total",
                labels={"provider": "huggingface", "status": "unhealthy"},
            )
            collector.observe("llm_health_check_duration_seconds", elapsed / 1000, labels={"provider": "huggingface"})
            return HealthStatus(
                status="unhealthy",
                message=str(exc),
                latency_ms=elapsed,
            )

    async def close(self) -> None:
        """Release model resources."""
        self._pipeline = None
        self._tokenizer = None
