"""Perceive-reason-act-observe reasoning loop.

The ``ReasoningLoop`` class executes the core agent cycle. It receives
context, provider, tool executor, and config as parameters -- it does
not own the context (the Agent base class creates and owns it).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from ia_agent_fwk.agents.exceptions import AgentMaxIterationsError
from ia_agent_fwk.llm.models import FinishReason, Message, TokenUsage
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

if TYPE_CHECKING:
    import asyncio

    from ia_agent_fwk.agents.config import AgentConfig
    from ia_agent_fwk.agents.context import AgentContext
    from ia_agent_fwk.agents.protocols import ToolExecutor
    from ia_agent_fwk.llm.base import LLMProvider
    from ia_agent_fwk.observability.prompt_log import PromptLogger

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class ReasoningLoop:
    """Execute the perceive-reason-act-observe reasoning cycle.

    Instance attributes ``partial_usage`` and ``partial_iterations`` are
    updated after every LLM call so callers can retrieve partial progress
    even when the loop is interrupted by timeout or other exceptions.
    """

    def __init__(self) -> None:
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._iteration_count = 0

    # ------------------------------------------------------------------
    # Partial-progress accessors
    # ------------------------------------------------------------------

    @property
    def partial_usage(self) -> TokenUsage:
        """Return token usage accumulated so far (may be partial)."""
        return TokenUsage(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens,
        )

    @property
    def partial_iterations(self) -> int:
        """Return the number of iterations completed so far."""
        return self._iteration_count

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_loop(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        context: AgentContext,
        provider: LLMProvider,
        tool_executor: ToolExecutor,
        config: AgentConfig,
        resume_event: asyncio.Event,
        tool_schemas: list[dict[str, Any]] | None = None,
        prompt_logger: PromptLogger | None = None,
    ) -> tuple[str, int, TokenUsage]:
        """Run the reasoning loop until completion.

        Returns
        -------
        tuple[str, int, TokenUsage]
            A tuple of (final_output, iteration_count, aggregated_usage).

        Raises
        ------
        AgentMaxIterationsError
            When the iteration limit is reached.

        """
        # Reset per-run counters
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._iteration_count = 0
        final_output = ""
        collector = get_metrics_collector()
        loop_outcome = "max_iterations"  # default if loop exhausts

        for _iteration in range(1, config.max_iterations + 1):
            self._iteration_count = _iteration
            iteration_start = time.monotonic()

            # Check resume event at top of each iteration (pause support)
            await resume_event.wait()

            with _tracer.start_as_current_span(
                f"reasoning.iteration/{_iteration}",
                attributes={
                    "reasoning.iteration": _iteration,
                    "reasoning.agent": config.agent_type,
                    "reasoning.max_iterations": config.max_iterations,
                },
            ) as iter_span:
                # --- Perceive ---
                messages = context.get_messages()
                context_msg_count = len(messages)
                context_tokens = sum(len(m.content or "") for m in messages) // 4  # rough estimate for span attribute
                iter_span.set_attribute("reasoning.perceive.message_count", context_msg_count)
                iter_span.set_attribute("reasoning.perceive.approx_tokens", context_tokens)

                # --- Reason ---
                chat_kwargs: dict[str, Any] = {
                    "max_tokens": config.max_tokens_per_response,
                }
                if config.model:
                    chat_kwargs["model"] = config.model
                if tool_schemas:
                    chat_kwargs["tools"] = provider.format_tools(tool_schemas)

                llm_start = time.monotonic()
                with _tracer.start_as_current_span(
                    "llm.chat",
                    attributes={
                        "llm.provider": provider.provider_name,
                        "llm.model": config.model or "",
                        "llm.agent": config.agent_type,
                        "llm.iteration": _iteration,
                    },
                ) as llm_span:
                    response = await provider.chat(messages, **chat_kwargs)
                    llm_duration_ms = (time.monotonic() - llm_start) * 1000
                    llm_span.set_attribute("llm.duration_ms", llm_duration_ms)
                    llm_span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens)
                    llm_span.set_attribute("llm.completion_tokens", response.usage.completion_tokens)
                    llm_span.set_attribute("llm.finish_reason", response.finish_reason.value)
                    if response.message.tool_calls:
                        llm_span.set_attribute("llm.tool_calls", len(response.message.tool_calls))
                        llm_span.set_attribute(
                            "llm.tool_names",
                            ",".join(tc.name for tc in response.message.tool_calls),
                        )

                # Aggregate token usage
                self._total_prompt_tokens += response.usage.prompt_tokens
                self._total_completion_tokens += response.usage.completion_tokens

                # Record LLM metrics
                collector.increment(
                    "llm_calls_total",
                    labels={"provider": provider.provider_name, "agent": config.agent_type},
                )
                collector.observe("llm_call_duration_seconds", llm_duration_ms / 1000)
                collector.increment(
                    "llm_prompt_tokens_total",
                    value=response.usage.prompt_tokens,
                    labels={"provider": provider.provider_name},
                )
                collector.increment(
                    "llm_completion_tokens_total",
                    value=response.usage.completion_tokens,
                    labels={"provider": provider.provider_name},
                )
                collector.increment(
                    "llm_finish_reason_total",
                    labels={"reason": response.finish_reason.value, "agent": config.agent_type},
                )
                if response.message.tool_calls:
                    collector.increment(
                        "llm_tool_calls_total",
                        value=len(response.message.tool_calls),
                        labels={"agent": config.agent_type},
                    )

                # Prompt logging
                if prompt_logger and prompt_logger.enabled:
                    tool_call_data = None
                    if response.message.tool_calls:
                        tool_call_data = [
                            {"name": tc.name, "arguments": tc.arguments} for tc in response.message.tool_calls
                        ]
                    prompt_logger.log_prompt(
                        provider=provider.provider_name,
                        model=response.model or config.model or "",
                        messages=[{"role": m.role, "content": m.content or ""} for m in messages],
                        response=response.message.content or "",
                        duration_ms=llm_duration_ms,
                        usage={
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        },
                        agent=config.agent_type,
                        iteration=_iteration,
                        tool_calls=tool_call_data,
                    )

                # --- Act: execute tool calls if present ---
                tool_count = 0
                tool_errors = 0
                if response.message.tool_calls:
                    # Append assistant message (with tool_calls) before tool results
                    context.add_message(response.message)
                    tool_count = len(response.message.tool_calls)

                    for tool_call in response.message.tool_calls:
                        with _tracer.start_as_current_span(
                            f"tool.execute/{tool_call.name}",
                            attributes={"tool.name": tool_call.name},
                        ) as tool_span:
                            result = await tool_executor.execute(tool_call)
                            tool_span.set_attribute("tool.status", "error" if result.error else "success")
                            if result.error:
                                tool_span.set_status(trace.StatusCode.ERROR, result.error[:200])
                                tool_errors += 1

                        # Store intermediate result
                        context.intermediate_results[result.tool_call_id] = result.output or result.error or ""

                        # Add tool result message to context
                        tool_content = result.output if not result.error else result.error
                        context.add_message(
                            Message(
                                role="tool",
                                content=tool_content,
                                tool_call_id=result.tool_call_id,
                            )
                        )
                else:
                    # --- Observe: append assistant message (no tools) ---
                    context.add_message(response.message)

                # Determine iteration outcome
                iteration_outcome = "continue"
                if response.finish_reason == FinishReason.length:
                    iteration_outcome = "truncated"
                elif response.finish_reason == FinishReason.stop and not response.message.tool_calls:
                    iteration_outcome = "stop"
                elif response.message.tool_calls:
                    iteration_outcome = "tool_call"

                # Iteration metrics & span attributes
                iteration_duration_ms = (time.monotonic() - iteration_start) * 1000
                iter_span.set_attribute("reasoning.act.tool_calls", tool_count)
                iter_span.set_attribute("reasoning.act.tool_errors", tool_errors)
                iter_span.set_attribute("reasoning.outcome", iteration_outcome)
                iter_span.set_attribute("reasoning.duration_ms", iteration_duration_ms)
                iter_span.set_attribute("reasoning.llm_duration_ms", llm_duration_ms)
                iter_span.set_attribute("reasoning.finish_reason", response.finish_reason.value)

                collector.increment(
                    "reasoning_iterations_total",
                    labels={"agent": config.agent_type, "outcome": iteration_outcome},
                )

                # Structured log per iteration
                logger.info(
                    "Iteration %d/%d: outcome=%s tools=%d duration=%.0fms tokens=%d",
                    _iteration,
                    config.max_iterations,
                    iteration_outcome,
                    tool_count,
                    iteration_duration_ms,
                    response.usage.prompt_tokens + response.usage.completion_tokens,
                    extra={
                        "reasoning_data": {
                            "event": "reasoning_iteration",
                            "agent": config.agent_type,
                            "iteration": _iteration,
                            "max_iterations": config.max_iterations,
                            "outcome": iteration_outcome,
                            "finish_reason": response.finish_reason.value,
                            "tool_calls": tool_count,
                            "tool_errors": tool_errors,
                            "context_messages": context_msg_count,
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "llm_duration_ms": round(llm_duration_ms, 1),
                            "iteration_duration_ms": round(iteration_duration_ms, 1),
                        }
                    },
                )

            # --- Evaluate stopping conditions ---
            if response.finish_reason == FinishReason.length:
                logger.warning(
                    "LLM response truncated (finish_reason=length). "
                    "Consider increasing max_tokens_per_response (current: %d).",
                    config.max_tokens_per_response,
                )
                loop_outcome = "truncated"
                final_output = response.message.content or ""
                break

            if response.finish_reason == FinishReason.stop and not response.message.tool_calls:
                loop_outcome = "stop"
                final_output = response.message.content or ""
                break

            # If tool calls were present, continue to next iteration
            # (finish_reason may be tool_calls or stop with tool_calls)

        else:
            # Loop exhausted without breaking -- max iterations reached
            collector.increment(
                "reasoning_loop_outcome_total",
                labels={"agent": config.agent_type, "outcome": "max_iterations"},
            )
            msg = f"Reasoning loop exceeded max_iterations ({config.max_iterations})"
            raise AgentMaxIterationsError(msg)

        # Record loop completion metrics
        collector.increment(
            "reasoning_loop_outcome_total",
            labels={"agent": config.agent_type, "outcome": loop_outcome},
        )

        aggregated_usage = TokenUsage(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens,
        )
        return final_output, self._iteration_count, aggregated_usage
