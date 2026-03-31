"""Session manager — in-memory history + PostgreSQL persistence."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage conversation sessions with in-memory history and optional persistence.

    Parameters
    ----------
    database_url:
        PostgreSQL connection URL. Empty = in-memory only.
    tenant_id:
        Tenant ID for multi-tenant isolation.

    """

    def __init__(self, database_url: str = "", tenant_id: str = "default") -> None:
        self._database_url = database_url
        self._tenant_id = tenant_id
        self._histories: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._cached_contexts: dict[str, str] = {}
        self._store: _PgStore | None = None
        self._store_lock = asyncio.Lock()

    async def _get_store(self) -> _PgStore | None:
        if not self._database_url:
            return None
        async with self._store_lock:
            if self._store is None:
                self._store = _PgStore(self._database_url, self._tenant_id)
                await self._store.ensure_tables()
        return self._store

    # ------------------------------------------------------------------
    # In-memory history
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return self._histories[session_id]

    def append(self, session_id: str, role: str, content: str) -> None:
        self._histories[session_id].append({"role": role, "content": content})

    def get_cached_context(self, session_id: str) -> str | None:
        return self._cached_contexts.get(session_id)

    def set_cached_context(self, session_id: str, context: str) -> None:
        self._cached_contexts[session_id] = context

    def is_new_session(self, session_id: str) -> bool:
        return len(self._histories[session_id]) == 0

    def clear_session(self, session_id: str) -> None:
        self._histories.pop(session_id, None)
        self._cached_contexts.pop(session_id, None)

    # ------------------------------------------------------------------
    # PostgreSQL persistence
    # ------------------------------------------------------------------

    async def ensure_conversation(self, session_id: str, agent: str = "default") -> None:
        store = await self._get_store()
        if store:
            await store.ensure_conversation(session_id, agent)

    async def persist_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        trace_id: str = "",
        mode: str = "",
        sources: list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        store = await self._get_store()
        if store:
            try:
                await store.save_message(
                    session_id,
                    role,
                    content,
                    trace_id=trace_id,
                    mode=mode,
                    sources=sources,
                    duration_ms=duration_ms,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Failed to persist message", exc_info=True)

    async def persist_llm_call(
        self,
        session_id: str,
        *,
        trace_id: str = "",
        purpose: str,
        provider: str,
        model: str,
        input_text: str = "",
        output_text: str = "",
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        duration_ms: float | None = None,
    ) -> None:
        store = await self._get_store()
        if store:
            try:
                await store.save_llm_call(
                    session_id,
                    trace_id=trace_id,
                    purpose=purpose,
                    provider=provider,
                    model=model,
                    input_text=input_text[:2000],
                    output_text=output_text[:2000],
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    duration_ms=duration_ms,
                )
            except Exception:  # noqa: BLE001
                logger.debug("Failed to persist LLM call", exc_info=True)

    # ------------------------------------------------------------------
    # Debug / admin
    # ------------------------------------------------------------------

    async def list_conversations(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        store = await self._get_store()
        if not store:
            return []
        return await store.list_conversations(limit, offset)

    async def get_conversation_debug(self, session_id: str) -> dict[str, Any]:
        store = await self._get_store()
        if not store:
            history = self._histories.get(session_id, [])
            return {"session_id": session_id, "messages": history, "llm_calls": []}
        return await store.get_conversation_debug(session_id)

    async def close(self) -> None:
        if self._store:
            await self._store.close()


# ═══════════════════════════════════════════════════════════════════════════
# PostgreSQL store (internal)
# ═══════════════════════════════════════════════════════════════════════════


class _PgStore:
    """PostgreSQL conversation store with multi-tenant support."""

    def __init__(self, database_url: str, tenant_id: str) -> None:
        self._database_url = database_url
        self._tenant_id = tenant_id
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()
        self._ready = False

    async def _ensure_pool(self) -> Any:
        async with self._pool_lock:
            if self._pool is None:
                import asyncpg  # noqa: PLC0415

                self._pool = await asyncpg.create_pool(
                    self._database_url,
                    min_size=1,
                    max_size=5,
                )
        return self._pool

    async def ensure_tables(self) -> None:
        if self._ready:
            return
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_conversations (
                    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                    session_id VARCHAR(64) NOT NULL,
                    agent VARCHAR(64) NOT NULL DEFAULT 'default',
                    message_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (tenant_id, session_id)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                    session_id VARCHAR(64) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    trace_id VARCHAR(64),
                    sources JSONB,
                    mode VARCHAR(20),
                    duration_ms REAL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_llm_calls (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                    session_id VARCHAR(64) NOT NULL,
                    trace_id VARCHAR(64),
                    purpose VARCHAR(32) NOT NULL,
                    provider VARCHAR(32) NOT NULL,
                    model VARCHAR(64) NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    duration_ms REAL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_msgs_tenant_session
                ON chat_messages(tenant_id, session_id, created_at)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_llm_tenant_session
                ON chat_llm_calls(tenant_id, session_id, created_at)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_msgs_trace
                ON chat_messages(trace_id)
            """)
        self._ready = True

    async def ensure_conversation(self, session_id: str, agent: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_conversations (tenant_id, session_id, agent)
                VALUES ($1, $2, $3)
                ON CONFLICT (tenant_id, session_id) DO NOTHING
                """,
                self._tenant_id,
                session_id,
                agent,
            )

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        trace_id: str = "",
        mode: str = "",
        sources: list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                    INSERT INTO chat_messages
                        (tenant_id, session_id, role, content, trace_id, sources, mode, duration_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                self._tenant_id,
                session_id,
                role,
                content,
                trace_id,
                json.dumps(sources) if sources else None,
                mode,
                duration_ms,
            )
            await conn.execute(
                """
                    UPDATE chat_conversations
                    SET message_count = message_count + 1, updated_at = NOW()
                    WHERE tenant_id = $1 AND session_id = $2
                    """,
                self._tenant_id,
                session_id,
            )

    async def save_llm_call(
        self,
        session_id: str,
        *,
        trace_id: str = "",
        purpose: str,
        provider: str,
        model: str,
        input_text: str = "",
        output_text: str = "",
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        duration_ms: float | None = None,
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_llm_calls
                    (tenant_id, session_id, trace_id, purpose, provider, model,
                     input_text, output_text, tokens_in, tokens_out, duration_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                self._tenant_id,
                session_id,
                trace_id,
                purpose,
                provider,
                model,
                input_text,
                output_text,
                tokens_in,
                tokens_out,
                duration_ms,
            )

    async def list_conversations(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, agent, message_count, created_at, updated_at
                FROM chat_conversations
                WHERE tenant_id = $1
                ORDER BY updated_at DESC
                LIMIT $2 OFFSET $3
                """,
                self._tenant_id,
                limit,
                offset,
            )
        return [dict(r) for r in rows]

    async def get_conversation_debug(self, session_id: str) -> dict[str, Any]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            msgs = await conn.fetch(
                """
                SELECT role, content, trace_id, sources, mode, duration_ms, created_at
                FROM chat_messages
                WHERE tenant_id = $1 AND session_id = $2
                ORDER BY created_at ASC
                """,
                self._tenant_id,
                session_id,
            )
            calls = await conn.fetch(
                """
                SELECT purpose, provider, model, input_text, output_text,
                       tokens_in, tokens_out, duration_ms, trace_id, created_at
                FROM chat_llm_calls
                WHERE tenant_id = $1 AND session_id = $2
                ORDER BY created_at ASC
                """,
                self._tenant_id,
                session_id,
            )
        return {
            "session_id": session_id,
            "tenant_id": self._tenant_id,
            "messages": [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "trace_id": r["trace_id"],
                    "mode": r["mode"],
                    "sources": json.loads(r["sources"]) if r["sources"] else None,
                    "duration_ms": r["duration_ms"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in msgs
            ],
            "llm_calls": [dict(r) for r in calls],
        }

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
