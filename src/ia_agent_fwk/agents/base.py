"""Agent abstract base class with lifecycle management.

The ``Agent`` ABC provides a concrete ``run()`` method that orchestrates
state transitions, lifecycle hooks, execution timeout, and reasoning loop
delegation. Subclasses override hooks, not ``run()`` itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from opentelemetry import context, trace

from ia_agent_fwk.agents.config import AgentResult
from ia_agent_fwk.agents.context import AgentContext
from ia_agent_fwk.agents.exceptions import (
    AgentError,
    AgentMaxIterationsError,
    AgentTimeoutError,
    InvalidStateTransitionError,
)
from ia_agent_fwk.agents.protocols import NoOpToolExecutor
from ia_agent_fwk.agents.reasoning import ReasoningLoop
from ia_agent_fwk.agents.state import AgentState, validate_transition
from ia_agent_fwk.llm.models import Message
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

if TYPE_CHECKING:
    from ia_agent_fwk.agents.config import AgentConfig
    from ia_agent_fwk.agents.protocols import ToolExecutor
    from ia_agent_fwk.llm.base import LLMProvider
    from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
    from ia_agent_fwk.memory.base import MemoryBackend
    from ia_agent_fwk.memory.models import MemoryResult

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

# Default context window when none is configured
_DEFAULT_CONTEXT_WINDOW = 8192


class Agent(ABC):
    """Abstract base class for all agents.

    Subclasses must implement ``agent_type`` to be concrete.
    Override lifecycle hooks (``on_start``, ``on_complete``, ``on_error``,
    ``on_timeout``) for custom behavior.

    Parameters
    ----------
    config:
        Agent configuration (Pydantic v2 model).
    provider:
        LLM provider instance (from Epic 2).
    tool_executor:
        Tool executor instance. If ``None``, ``NoOpToolExecutor`` is used.

    """

    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        tool_executor: ToolExecutor | None = None,
        memory_backend: MemoryBackend | None = None,
        conversation_backend: ConversationMemoryBackend | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._tool_executor: ToolExecutor = tool_executor or NoOpToolExecutor()
        self._memory_backend = memory_backend
        self._conversation_backend = conversation_backend
        self._state = AgentState.IDLE
        self._resume_event = asyncio.Event()
        self._resume_event.set()  # Start in resumed state
        self._reasoning_loop = ReasoningLoop()
        self._context: AgentContext | None = None
        self._current_task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------
    # Abstract property (ensures Agent cannot be instantiated directly)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        ...

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """Return the current agent state."""
        return self._state

    @property
    def config(self) -> AgentConfig:
        """Return the agent configuration."""
        return self._config

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: AgentState) -> None:
        """Validate and execute a state transition (atomic)."""
        old_state = self._state
        validate_transition(old_state, new_state)
        self._state = new_state
        collector = get_metrics_collector()
        collector.increment(
            "agent_state_transitions_total",
            labels={
                "agent_type": self.agent_type,
                "from_state": old_state.value,
                "to_state": new_state.value,
            },
        )
        logger.info(
            "Agent '%s' state transition: %s -> %s",
            self._config.name,
            old_state.value,
            new_state.value,
            extra={
                "agent_data": {
                    "event": "agent_state_transition",
                    "agent": self._config.name,
                    "agent_type": self.agent_type,
                    "from_state": old_state.value,
                    "to_state": new_state.value,
                }
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks (overridable by subclasses)
    # ------------------------------------------------------------------

    async def on_start(self) -> None:  # noqa: B027
        """Run custom initialization when the agent starts."""

    async def on_complete(self, result: AgentResult) -> None:  # noqa: B027
        """Handle successful completion of the agent."""

    async def on_error(self, error: Exception) -> None:  # noqa: B027
        """Handle an error during agent execution."""

    async def on_timeout(self) -> None:  # noqa: B027
        """Handle execution timeout."""

    # ------------------------------------------------------------------
    # Tool schema extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_prompt_logger() -> Any:
        """Create a PromptLogger from global config, if available."""
        try:
            from ia_agent_fwk.config.loader import load_config  # noqa: PLC0415
            from ia_agent_fwk.observability.prompt_log import PromptLogger  # noqa: PLC0415

            settings = load_config()
            return PromptLogger(settings.observability.prompt_logging)
        except Exception:  # noqa: BLE001
            return None

    def _get_tool_schemas(self) -> list[dict[str, Any]] | None:
        """Extract tool schemas from the executor's registry, if available."""
        from ia_agent_fwk.tools.executor import DefaultToolExecutor  # noqa: PLC0415

        if isinstance(self._tool_executor, DefaultToolExecutor):
            schemas = self._tool_executor.registry.openai_schemas(
                agent_id=self._config.name,
                permission_manager=self._tool_executor._permission_manager,  # noqa: SLF001
            )
            return schemas or None
        return None

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _load_conversation_history(self, conversation_id: str) -> list[Message]:
        """Load prior messages from conversation backend."""
        assert self._conversation_backend is not None  # noqa: S101
        collector = get_metrics_collector()
        with _tracer.start_as_current_span(
            "memory.load_conversation",
            attributes={"memory.conversation_id": conversation_id},
        ) as span:
            try:
                start = time.monotonic()
                msgs = await self._conversation_backend.get_messages(conversation_id)
                duration_ms = (time.monotonic() - start) * 1000
                messages = [Message(role=m.role, content=m.content) for m in msgs]
                span.set_attribute("memory.messages_loaded", len(messages))
                span.set_attribute("memory.duration_ms", duration_ms)
                collector.increment(
                    "memory_operations_total",
                    labels={"operation": "load_conversation", "status": "success"},
                )
                collector.observe("memory_operation_duration_seconds", duration_ms / 1000)
                logger.info(
                    "Loaded %d messages for conversation %s (%.1fms)",
                    len(messages),
                    conversation_id,
                    duration_ms,
                    extra={
                        "memory_data": {
                            "event": "memory_load_conversation",
                            "conversation_id": conversation_id,
                            "messages_loaded": len(messages),
                            "duration_ms": round(duration_ms, 1),
                        }
                    },
                )
                return messages  # noqa: TRY300
            except Exception:  # noqa: BLE001
                span.set_status(trace.StatusCode.ERROR, "Failed to load conversation")
                collector.increment(
                    "memory_operations_total",
                    labels={"operation": "load_conversation", "status": "error"},
                )
                logger.warning(
                    "Failed to load conversation %s, continuing without history",
                    conversation_id,
                    exc_info=True,
                )
                return []

    async def _search_semantic_memory(self, query: str) -> list[MemoryResult]:
        """Search vector memory for relevant past context."""
        assert self._memory_backend is not None  # noqa: S101
        mem_cfg = self._config.memory
        collector = get_metrics_collector()
        with _tracer.start_as_current_span(
            "memory.semantic_search",
            attributes={
                "memory.backend": self._memory_backend.backend_type,
                "memory.top_k": mem_cfg.semantic_search_top_k,
                "memory.query_length": len(query),
            },
        ) as span:
            try:
                start = time.monotonic()
                results = await self._memory_backend.search(
                    query,
                    top_k=mem_cfg.semantic_search_top_k,
                )
                duration_ms = (time.monotonic() - start) * 1000
                # Filter by score threshold
                filtered = [r for r in results if r.score >= mem_cfg.semantic_search_score_threshold]
                span.set_attribute("memory.results_total", len(results))
                span.set_attribute("memory.results_filtered", len(filtered))
                span.set_attribute("memory.duration_ms", duration_ms)
                collector.increment(
                    "memory_operations_total",
                    labels={"operation": "semantic_search", "status": "success"},
                )
                collector.observe("memory_operation_duration_seconds", duration_ms / 1000)
                collector.increment(
                    "memory_semantic_results_total",
                    value=len(filtered),
                )
                logger.info(
                    "Semantic search returned %d/%d results (%.1fms, threshold=%.2f)",
                    len(filtered),
                    len(results),
                    duration_ms,
                    mem_cfg.semantic_search_score_threshold,
                    extra={
                        "memory_data": {
                            "event": "memory_semantic_search",
                            "backend": self._memory_backend.backend_type,
                            "query_length": len(query),
                            "results_total": len(results),
                            "results_filtered": len(filtered),
                            "duration_ms": round(duration_ms, 1),
                            "score_threshold": mem_cfg.semantic_search_score_threshold,
                        }
                    },
                )
                return filtered  # noqa: TRY300
            except Exception:  # noqa: BLE001
                span.set_status(trace.StatusCode.ERROR, "Semantic search failed")
                collector.increment(
                    "memory_operations_total",
                    labels={"operation": "semantic_search", "status": "error"},
                )
                logger.warning(
                    "Semantic memory search failed, continuing without memory context",
                    exc_info=True,
                )
                return []

    async def _persist_turn(
        self,
        conversation_id: str,
        user_input: str,
        assistant_output: str,
    ) -> None:
        """Store the turn in both conversation and vector memory backends."""
        collector = get_metrics_collector()

        # 1. Persist to conversation backend
        if self._conversation_backend and self._config.memory.conversation_persistence_enabled:
            with _tracer.start_as_current_span(
                "memory.persist_conversation",
                attributes={"memory.conversation_id": conversation_id},
            ) as span:
                try:
                    start = time.monotonic()
                    await self._conversation_backend.add_message(
                        conversation_id=conversation_id,
                        role="user",
                        content=user_input,
                    )
                    await self._conversation_backend.add_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=assistant_output,
                    )
                    duration_ms = (time.monotonic() - start) * 1000
                    span.set_attribute("memory.duration_ms", duration_ms)
                    collector.increment(
                        "memory_operations_total",
                        labels={"operation": "persist_conversation", "status": "success"},
                    )
                    collector.observe("memory_operation_duration_seconds", duration_ms / 1000)
                    logger.info(
                        "Persisted turn to conversation %s (%.1fms)",
                        conversation_id,
                        duration_ms,
                        extra={
                            "memory_data": {
                                "event": "memory_persist_conversation",
                                "conversation_id": conversation_id,
                                "duration_ms": round(duration_ms, 1),
                            }
                        },
                    )
                except Exception:  # noqa: BLE001
                    span.set_status(trace.StatusCode.ERROR, "Failed to persist conversation")
                    collector.increment(
                        "memory_operations_total",
                        labels={"operation": "persist_conversation", "status": "error"},
                    )
                    logger.warning(
                        "Failed to persist conversation turn for %s",
                        conversation_id,
                        exc_info=True,
                    )

        # 2. Store in vector memory for future semantic retrieval
        if self._memory_backend and self._config.memory.semantic_search_enabled:
            with _tracer.start_as_current_span(
                "memory.store_vector",
                attributes={"memory.backend": self._memory_backend.backend_type},
            ) as span:
                try:
                    import uuid  # noqa: PLC0415
                    from datetime import datetime, timezone  # noqa: PLC0415

                    start = time.monotonic()
                    turn_key = f"{self._config.name}:{conversation_id}:{uuid.uuid4().hex[:12]}"
                    turn_value = f"User: {user_input}\nAssistant: {assistant_output}"
                    await self._memory_backend.store(
                        key=turn_key,
                        value=turn_value,
                        metadata={
                            "agent_name": self._config.name,
                            "agent_type": self.agent_type,
                            "conversation_id": conversation_id,
                            "role": "turn",
                            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                        },
                    )
                    duration_ms = (time.monotonic() - start) * 1000
                    span.set_attribute("memory.key", turn_key)
                    span.set_attribute("memory.duration_ms", duration_ms)
                    collector.increment(
                        "memory_operations_total",
                        labels={"operation": "store_vector", "status": "success"},
                    )
                    collector.observe("memory_operation_duration_seconds", duration_ms / 1000)
                    logger.info(
                        "Stored turn in vector memory key=%s (%.1fms)",
                        turn_key,
                        duration_ms,
                        extra={
                            "memory_data": {
                                "event": "memory_store_vector",
                                "backend": self._memory_backend.backend_type,
                                "key": turn_key,
                                "conversation_id": conversation_id,
                                "duration_ms": round(duration_ms, 1),
                            }
                        },
                    )
                except Exception:  # noqa: BLE001
                    span.set_status(trace.StatusCode.ERROR, "Failed to store vector memory")
                    collector.increment(
                        "memory_operations_total",
                        labels={"operation": "store_vector", "status": "error"},
                    )
                    logger.warning(
                        "Failed to store turn in vector memory",
                        exc_info=True,
                    )

    @staticmethod
    def _format_memories_as_context(memories: list[MemoryResult]) -> str:
        """Format retrieved memories into a text block for LLM context injection."""
        lines = ["[Relevant context from previous interactions]:"]
        for i, mem in enumerate(memories, 1):
            lines.append(f"{i}. (relevance: {mem.score:.2f}) {mem.value}")
        lines.append("---")
        lines.append("Use this context if relevant to the current conversation.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def run(  # noqa: C901, PLR0912, PLR0915
        self,
        input_text: str,
        conversation_history: list[Message] | None = None,
        conversation_id: str | None = None,
    ) -> AgentResult:
        """Execute the agent reasoning loop.

        Parameters
        ----------
        input_text:
            The user input to process.
        conversation_history:
            Optional list of prior conversation messages to pre-seed the
            agent context.  Messages are added *before* the current user
            message so the LLM can see the full conversation.
        conversation_id:
            Optional conversation identifier for memory persistence.
            When set (and memory backends are configured), the agent will
            auto-load history from the conversation backend and persist
            the turn after completion.

        Returns
        -------
        AgentResult
            Structured result containing output, state, usage, etc.

        """
        start_time = time.monotonic()
        collector = get_metrics_collector()
        memories_retrieved = 0

        # Validate and transition to RUNNING
        self._transition_to(AgentState.RUNNING)

        # Call on_start hook
        await self.on_start()

        # Auto-generate conversation_id if memory backends are present
        if conversation_id is None and (self._conversation_backend or self._memory_backend):
            import uuid  # noqa: PLC0415

            conversation_id = uuid.uuid4().hex

        # Create context
        context_window = self._config.context_window or _DEFAULT_CONTEXT_WINDOW
        self._context = AgentContext(
            system_prompt=self._config.system_prompt,
            token_budget=context_window,
            token_counter=self._provider.count_tokens,
        )
        self._context.current_task = input_text

        # --- Memory: load conversation history from backend ---
        if (
            self._conversation_backend
            and conversation_id
            and self._config.memory.enabled
            and self._config.memory.conversation_persistence_enabled
            and not conversation_history  # explicit history takes precedence
        ):
            conversation_history = await self._load_conversation_history(conversation_id)

        # Pre-seed with conversation history (if provided or loaded)
        if conversation_history:
            for msg in conversation_history:
                self._context.add_message(msg)

        # --- Memory: semantic search for relevant past context ---
        if self._memory_backend and self._config.memory.enabled and self._config.memory.semantic_search_enabled:
            memories = await self._search_semantic_memory(input_text)
            if memories:
                memories_retrieved = len(memories)
                memory_text = self._format_memories_as_context(memories)
                self._context.inject_memory_context(memory_text)

        self._context.add_message(Message(role="user", content=input_text))

        # Get tool schemas from the executor's registry (if available)
        tool_schemas = self._get_tool_schemas()

        # Create prompt logger if config is available
        prompt_logger = self._get_prompt_logger()

        # Wrap entire agent execution in a tracing span
        span = _tracer.start_span(
            f"agent.run/{self.agent_type}",
            attributes={
                "agent.type": self.agent_type,
                "agent.name": self._config.name,
                "agent.input_length": len(input_text),
            },
        )
        ctx = trace.set_span_in_context(span)
        token = context.attach(ctx)

        try:
            self._current_task = asyncio.current_task()
            output, iterations, usage = await asyncio.wait_for(
                self._reasoning_loop.run_loop(
                    self._context,
                    self._provider,
                    self._tool_executor,
                    self._config,
                    self._resume_event,
                    tool_schemas=tool_schemas,
                    prompt_logger=prompt_logger,
                ),
                timeout=float(self._config.execution_timeout),
            )

        except asyncio.CancelledError:
            # Task was cancelled (e.g. by stop()). State is already FAILED.
            duration_ms = (time.monotonic() - start_time) * 1000
            partial_usage = self._reasoning_loop.partial_usage
            collector.increment(
                "agent_executions_total",
                labels={"agent_type": self.agent_type, "outcome": "cancelled"},
            )
            collector.increment(
                "agent_safeguard_triggers_total",
                labels={"agent_type": self.agent_type, "safeguard": "cancelled"},
            )
            collector.observe("agent_execution_duration_seconds", duration_ms / 1000)
            span.set_status(trace.StatusCode.ERROR, "cancelled")
            logger.warning(
                "Agent '%s' cancelled after %.0fms (%d iterations)",
                self._config.name,
                duration_ms,
                self._reasoning_loop.partial_iterations,
                extra={
                    "agent_data": {
                        "event": "agent_cancelled",
                        "agent": self._config.name,
                        "agent_type": self.agent_type,
                        "duration_ms": round(duration_ms, 1),
                        "iterations": self._reasoning_loop.partial_iterations,
                    }
                },
            )
            return AgentResult(
                output="",
                state=self._state,
                usage=partial_usage,
                iterations=self._reasoning_loop.partial_iterations,
                duration_ms=duration_ms,
                error="Agent execution was cancelled",
            )

        except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
            duration_ms = (time.monotonic() - start_time) * 1000
            partial_usage = self._reasoning_loop.partial_usage
            self._transition_to(AgentState.FAILED)
            timeout_err = AgentTimeoutError(f"Agent execution timed out after {self._config.execution_timeout}s")
            collector.increment(
                "agent_executions_total",
                labels={"agent_type": self.agent_type, "outcome": "timeout"},
            )
            collector.increment(
                "agent_safeguard_triggers_total",
                labels={"agent_type": self.agent_type, "safeguard": "timeout"},
            )
            collector.observe("agent_execution_duration_seconds", duration_ms / 1000)
            span.set_status(trace.StatusCode.ERROR, "timeout")
            logger.error(  # noqa: TRY400
                "Agent '%s' timed out after %.0fms (limit: %ds, iterations: %d)",
                self._config.name,
                duration_ms,
                self._config.execution_timeout,
                self._reasoning_loop.partial_iterations,
                extra={
                    "agent_data": {
                        "event": "agent_timeout",
                        "agent": self._config.name,
                        "agent_type": self.agent_type,
                        "duration_ms": round(duration_ms, 1),
                        "timeout_seconds": self._config.execution_timeout,
                        "iterations": self._reasoning_loop.partial_iterations,
                    }
                },
            )
            await self.on_timeout()
            await self.on_error(timeout_err)
            return AgentResult(
                output="",
                state=AgentState.FAILED,
                usage=partial_usage,
                iterations=self._reasoning_loop.partial_iterations,
                duration_ms=duration_ms,
                error=str(timeout_err),
            )

        except AgentMaxIterationsError as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            partial_usage = self._reasoning_loop.partial_usage
            self._transition_to(AgentState.FAILED)
            collector.increment(
                "agent_executions_total",
                labels={"agent_type": self.agent_type, "outcome": "max_iterations"},
            )
            collector.increment(
                "agent_safeguard_triggers_total",
                labels={"agent_type": self.agent_type, "safeguard": "max_iterations"},
            )
            collector.observe("agent_execution_duration_seconds", duration_ms / 1000)
            span.set_status(trace.StatusCode.ERROR, "max_iterations")
            logger.error(  # noqa: TRY400
                "Agent '%s' hit max iterations (%d) after %.0fms",
                self._config.name,
                self._config.max_iterations,
                duration_ms,
                extra={
                    "agent_data": {
                        "event": "agent_max_iterations",
                        "agent": self._config.name,
                        "agent_type": self.agent_type,
                        "max_iterations": self._config.max_iterations,
                        "duration_ms": round(duration_ms, 1),
                    }
                },
            )
            await self.on_error(exc)
            return AgentResult(
                output="",
                state=AgentState.FAILED,
                usage=partial_usage,
                iterations=self._config.max_iterations,
                duration_ms=duration_ms,
                error=str(exc),
            )

        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.monotonic() - start_time) * 1000
            partial_usage = self._reasoning_loop.partial_usage
            self._transition_to(AgentState.FAILED)
            collector.increment(
                "agent_executions_total",
                labels={"agent_type": self.agent_type, "outcome": "error"},
            )
            collector.observe("agent_execution_duration_seconds", duration_ms / 1000)
            span.set_status(trace.StatusCode.ERROR, str(exc)[:200])
            logger.error(  # noqa: TRY400
                "Agent '%s' failed with %s after %.0fms: %s",
                self._config.name,
                type(exc).__name__,
                duration_ms,
                exc,
                extra={
                    "agent_data": {
                        "event": "agent_error",
                        "agent": self._config.name,
                        "agent_type": self.agent_type,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "duration_ms": round(duration_ms, 1),
                        "iterations": self._reasoning_loop.partial_iterations,
                    }
                },
            )
            await self.on_error(exc)
            return AgentResult(
                output="",
                state=AgentState.FAILED,
                usage=partial_usage,
                iterations=self._reasoning_loop.partial_iterations,
                duration_ms=duration_ms,
                error=str(exc),
            )

        else:
            # Success path
            duration_ms = (time.monotonic() - start_time) * 1000
            self._transition_to(AgentState.COMPLETED)

            collector.increment(
                "agent_executions_total",
                labels={"agent_type": self.agent_type, "outcome": "completed"},
            )
            collector.observe("agent_execution_duration_seconds", duration_ms / 1000)

            span.set_attribute("agent.iterations", iterations)
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_attribute("agent.tokens.total", usage.total_tokens)
            span.set_attribute("agent.memories_retrieved", memories_retrieved)
            span.set_status(trace.StatusCode.OK)

            # Log context stats for observability
            if self._context:
                ctx_stats = self._context.get_stats()
                span.set_attribute("agent.context.messages", ctx_stats["message_count"])
                span.set_attribute("agent.context.token_utilization", ctx_stats["token_utilization"])
                span.set_attribute("agent.context.evictions", ctx_stats["eviction_count"])
                logger.info(
                    "Agent '%s' completed: %d iterations, %.0fms, %d tokens, "
                    "context: %d msgs (%.0f%% budget, %d evictions)",
                    self._config.name,
                    iterations,
                    duration_ms,
                    usage.total_tokens,
                    ctx_stats["message_count"],
                    ctx_stats["token_utilization"] * 100,
                    ctx_stats["eviction_count"],
                    extra={
                        "agent_data": {
                            "event": "agent_completed",
                            "agent": self._config.name,
                            "agent_type": self.agent_type,
                            "iterations": iterations,
                            "duration_ms": round(duration_ms, 1),
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                            "memories_retrieved": memories_retrieved,
                            "context": ctx_stats,
                        }
                    },
                )

            # --- Memory: persist the turn ---
            if conversation_id and self._config.memory.enabled:
                await self._persist_turn(conversation_id, input_text, output)

            result = AgentResult(
                output=output,
                state=AgentState.COMPLETED,
                usage=usage,
                iterations=iterations,
                duration_ms=duration_ms,
                metadata={
                    "conversation_id": conversation_id,
                    "memories_retrieved": memories_retrieved,
                },
            )
            await self.on_complete(result)
            return result

        finally:
            span.end()
            context.detach(token)
            self._current_task = None

    # ------------------------------------------------------------------
    # Control operations
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Stop a running agent, cancel the task, and transition to FAILED.

        If the agent is already in a terminal state the call is a no-op.
        """
        error = AgentError("Agent stopped by user")
        try:
            self._transition_to(AgentState.FAILED)
        except InvalidStateTransitionError:
            # Already in a terminal state
            return

        # Cancel the running reasoning-loop task if one exists
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

        # Unblock a possibly paused loop so cancellation can propagate
        self._resume_event.set()

        await self.on_error(error)

    def pause(self) -> None:
        """Pause a running agent and transition to WAITING_FOR_INPUT.

        The resume event is cleared *before* the state transition to
        prevent a race where the reasoning loop observes
        WAITING_FOR_INPUT but the event is still set.

        Note: ``pause()`` and ``resume()`` must be called from the same
        asyncio event-loop thread as the running agent.
        """
        # Clear the event first so the loop blocks before observing state
        self._resume_event.clear()
        self._transition_to(AgentState.WAITING_FOR_INPUT)

    def resume(self, input_text: str) -> None:
        """Resume a paused agent with new input.

        The new input message is added to context *before* the event is
        set, ensuring the loop sees the message when it unblocks.

        Parameters
        ----------
        input_text:
            Additional input to add to the context.

        """
        if self._context is not None:
            self._context.add_message(Message(role="user", content=input_text))
        self._transition_to(AgentState.RUNNING)
        self._resume_event.set()
