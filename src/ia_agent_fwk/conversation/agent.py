"""ConversationalRAGAgent — orchestrates the full conversation pipeline.

Classify → Search/Cache → Build prompt → LLM → Output guard → Persist.
100% generic. All behavior from config. All data from Qdrant per tenant.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ia_agent_fwk.conversation.classifier import ClassifyResult, MessageClassifier
from ia_agent_fwk.conversation.context import clean_context, inject_context
from ia_agent_fwk.conversation.session import SessionManager
from ia_agent_fwk.ingestion.config_provider import RagConfigProvider
from ia_agent_fwk.ingestion.context_assembly import assemble_context
from ia_agent_fwk.ingestion.query_pipeline import QueryPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResponse:
    """Response from the conversational agent."""

    session_id: str
    response: str
    sources: list[dict[str, Any]]
    duration_ms: float
    mode: str  # "search" or "chat"


class ConversationalRAGAgent:
    """Generic conversational RAG agent.

    Parameters
    ----------
    config:
        RagConfigProvider for this tenant. All prompts, parameters, etc.
    query_pipeline:
        RAG search pipeline (embedding + hybrid search + reranking).
    session_manager:
        Session history + PostgreSQL persistence.
    llm_chat:
        Async callable(messages, temperature, max_tokens) -> str
    llm_stream:
        Async callable(messages, temperature, max_tokens) -> AsyncIterator[(token, done)]
    llm_quick:
        Async callable(prompt, max_tokens) -> str (for classify/rewrite)
    input_guard:
        Optional. Object with check(message) -> result with .ok and .detail
    output_guard:
        Optional. Object with check_and_clean(text) -> str

    """

    def __init__(
        self,
        config: RagConfigProvider,
        query_pipeline: QueryPipeline,
        session_manager: SessionManager,
        llm_chat: Callable[..., Awaitable[str]],
        llm_stream: Callable[..., Any],
        llm_quick: Callable[..., Awaitable[str]],
        *,
        input_guard: Any = None,
        output_guard: Any = None,
    ) -> None:
        self._config = config
        self._pipeline = query_pipeline
        self._session = session_manager
        self._llm_chat = llm_chat
        self._llm_stream = llm_stream
        self._classifier = MessageClassifier(
            llm_quick=llm_quick,
            classify_prompt=config.require("prompts.classify_rewrite"),
            max_tokens=int(config.get("llm.classify_max_tokens", 80)),
            history_window=int(config.get("chat.classify_history_window", 4)),
        )
        self._input_guard = input_guard
        self._output_guard = output_guard

    async def reload_config(self) -> None:
        """Reload config from DB. Updates classifier prompt, etc."""
        await self._config.load()
        self._classifier = MessageClassifier(
            llm_quick=self._classifier._llm_quick,
            classify_prompt=self._config.require("prompts.classify_rewrite"),
            max_tokens=int(self._config.get("llm.classify_max_tokens", 80)),
            history_window=int(self._config.get("chat.classify_history_window", 4)),
        )
        logger.info("Agent config reloaded")

    # ------------------------------------------------------------------
    # Sync (full response)
    # ------------------------------------------------------------------

    async def handle(
        self,
        message: str,
        *,
        session_id: str | None = None,
        agent: str = "sales-agent",
    ) -> AgentResponse:
        """Process a message end-to-end. Returns full response."""
        # Input guard
        if self._input_guard:
            check = self._input_guard.check(message)
            if not check.ok:
                return AgentResponse(
                    session_id=session_id or "",
                    response=check.detail,
                    sources=[],
                    duration_ms=0,
                    mode="rejected",
                )

        t0 = time.monotonic()
        sid = session_id or str(uuid.uuid4())[:8]
        history = self._session.get_history(sid)

        if self._session.is_new_session(sid):
            await self._session.ensure_conversation(sid, agent)

        # 1. Classify + search/cache
        context, results, mode = await self._resolve_context(message, history, sid)

        # 2. Build messages
        messages = self._build_messages(context, history, message, agent)

        # 3. LLM
        temp = float(self._config.get("llm.temperature", 0.3))
        max_tok = int(self._config.get("llm.max_tokens", 1024))
        response_text = await self._llm_chat(messages, temp, max_tok)

        # 4. Output guard
        if self._output_guard:
            response_text = self._output_guard.check_and_clean(response_text)

        # 5. Store
        self._session.append(sid, "user", message)
        self._session.append(sid, "assistant", response_text)

        sources = self._extract_sources(results)
        duration_ms = (time.monotonic() - t0) * 1000

        # 6. Persist (fire-and-forget)
        trace_id = str(uuid.uuid4())[:16]
        await self._session.persist_message(
            sid,
            "user",
            message,
            trace_id=trace_id,
            mode=mode,
        )
        await self._session.persist_message(
            sid,
            "assistant",
            response_text,
            trace_id=trace_id,
            sources=sources,
            duration_ms=round(duration_ms),
        )

        return AgentResponse(
            session_id=sid,
            response=response_text,
            sources=sources,
            duration_ms=round(duration_ms),
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Streaming (SSE)
    # ------------------------------------------------------------------

    async def handle_stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        agent: str = "sales-agent",
    ) -> AsyncIterator[dict[str, Any]]:
        """Process a message with streaming. Yields SSE event dicts."""
        # Input guard
        if self._input_guard:
            check = self._input_guard.check(message)
            if not check.ok:
                yield {"error": check.detail}
                return

        t0 = time.monotonic()
        sid = session_id or str(uuid.uuid4())[:8]
        history = self._session.get_history(sid)

        if self._session.is_new_session(sid):
            await self._session.ensure_conversation(sid, agent)

        # 1. Classify + search/cache
        context, results, mode = await self._resolve_context(message, history, sid)

        if mode == "search":
            yield {"status": "searching", "message": "Buscando información..."}

        # 2. Build messages
        messages = self._build_messages(context, history, message, agent)

        yield {"status": "thinking", "message": "Generando respuesta..."}

        # 3. Stream LLM
        temp = float(self._config.get("llm.temperature", 0.3))
        max_tok = int(self._config.get("llm.max_tokens", 1024))
        full_response: list[str] = []

        async for token, _done in self._llm_stream(messages, temp, max_tok):
            if token:
                full_response.append(token)
                yield {"token": token}

        response_text = "".join(full_response)

        # 4. Output guard (audit only — stream already sent)
        if self._output_guard:
            self._output_guard.check_and_clean(response_text)

        # 5. Store
        self._session.append(sid, "user", message)
        self._session.append(sid, "assistant", response_text)

        sources = self._extract_sources(results)
        duration_ms = (time.monotonic() - t0) * 1000

        # 6. Persist
        trace_id = str(uuid.uuid4())[:16]
        await self._session.persist_message(
            sid,
            "user",
            message,
            trace_id=trace_id,
            mode=mode,
        )
        await self._session.persist_message(
            sid,
            "assistant",
            response_text,
            trace_id=trace_id,
            sources=sources,
            duration_ms=round(duration_ms),
        )

        yield {
            "done": True,
            "session_id": sid,
            "sources": sources,
            "duration_ms": round(duration_ms),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _resolve_context(
        self,
        message: str,
        history: list[dict[str, str]],
        session_id: str,
    ) -> tuple[str, list[Any], str]:
        """Classify message and resolve context (search or cache)."""
        cached = self._session.get_cached_context(session_id)
        result: ClassifyResult = await self._classifier.classify(
            message,
            history,
            cached_context=cached,
        )

        if result.mode == "chat":
            return result.cached_context or "", [], "chat"

        # SEARCH
        top_k = int(self._config.get("retrieval.top_k", 5))
        search_results = await self._pipeline.search(
            query=result.search_query,
            top_k=top_k,
            filters=result.filters,
        )

        max_chars = int(self._config.get("retrieval.context_max_chars", 6000))
        fmt = str(self._config.get("retrieval.context_format", "plain"))
        raw_context = assemble_context(search_results, format=fmt, max_chars=max_chars)

        prefixes = self._config.get("context.cleanup_prefixes", [])
        context = clean_context(raw_context, prefixes)
        self._session.set_cached_context(session_id, context)

        return context, search_results, "search"

    def _build_messages(
        self,
        context: str,
        history: list[dict[str, str]],
        user_message: str,
        agent: str,
    ) -> list[dict[str, str]]:
        """Build the LLM message list: system + history + user."""
        system_prompt = str(self._config.require(f"prompts.{agent}"))
        if context:
            template = str(self._config.require("prompts.context_injection"))
            system_prompt = inject_context(system_prompt, context, template)

        window = int(self._config.get("chat.history_window", 10))
        return [
            {"role": "system", "content": system_prompt},
            *[{"role": m["role"], "content": m["content"]} for m in history[-window:]],
            {"role": "user", "content": user_message},
        ]

    @staticmethod
    def _extract_sources(results: list[Any]) -> list[dict[str, Any]]:
        """Extract source metadata from search results."""
        return [
            {
                "document": r.metadata.get("document_id", "?"),
                "section": r.metadata.get("section", "-"),
                "score": round(r.score, 3),
            }
            for r in results
        ]
