"""PostgreSQL key-value structured memory backend.

Stores key-value pairs with optional TTL in a PostgreSQL table.
Uses ``asyncpg`` for async database access.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from ia_agent_fwk.memory.backends._validation import validate_sql_identifier
from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.exceptions import MemoryConfigError, MemoryRetrieveError, MemoryStoreError
from ia_agent_fwk.memory.models import MemoryResult

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore[assignment,unused-ignore]

logger = logging.getLogger(__name__)


class StructuredMemoryBackend(MemoryBackend):
    """PostgreSQL key-value structured memory backend.

    Parameters
    ----------
    database_url:
        PostgreSQL connection string.
    table_name:
        Table name for KV storage.
    agent_namespace:
        Namespace for multi-agent isolation.
    default_ttl_seconds:
        Default TTL in seconds. ``0`` means no expiration.
    pool_min_size:
        Minimum connection pool size.
    pool_max_size:
        Maximum connection pool size.

    """

    def __init__(  # noqa: PLR0913
        self,
        database_url: str,
        table_name: str = "memory_kv",
        agent_namespace: str = "default",
        default_ttl_seconds: int = 0,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        validate_sql_identifier(table_name, label="table name")

        self._database_url = database_url
        self._table_name = table_name
        self._agent_namespace = agent_namespace
        self._default_ttl_seconds = default_ttl_seconds
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None
        self._table_ready = False

    @property
    def backend_type(self) -> str:
        """Return ``'structured'``."""
        return "structured"

    async def _ensure_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        """Create the connection pool on first use."""
        if self._pool is None:
            if asyncpg is None:  # pragma: no cover
                msg = "asyncpg is required for StructuredMemoryBackend"
                raise MemoryConfigError(msg)

            try:
                self._pool = await asyncpg.create_pool(
                    self._database_url,
                    min_size=self._pool_min_size,
                    max_size=self._pool_max_size,
                )
            except Exception as exc:
                msg = f"Failed to create structured memory connection pool: {exc}"
                raise MemoryStoreError(msg) from exc
        return self._pool

    async def _ensure_table(self) -> None:
        """Create the KV table and indexes if they do not exist."""
        if self._table_ready:
            return

        pool = await self._ensure_pool()
        table = self._table_name

        create_table = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key             VARCHAR(512) NOT NULL,
                value           JSONB NOT NULL,
                metadata        JSONB DEFAULT '{{}}'::jsonb,
                agent_namespace VARCHAR(128) NOT NULL DEFAULT 'default',
                ttl_at          TIMESTAMPTZ,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """

        create_unique_idx = f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_key_ns
                ON {table} (key, agent_namespace)
        """

        create_ns_idx = f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_namespace
                ON {table} (agent_namespace)
        """

        create_ttl_idx = f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_ttl
                ON {table} (ttl_at)
                WHERE ttl_at IS NOT NULL
        """

        try:
            async with pool.acquire() as conn:
                await conn.execute(create_table)
                await conn.execute(create_unique_idx)
                await conn.execute(create_ns_idx)
                await conn.execute(create_ttl_idx)
        except Exception as exc:
            msg = f"Failed to create structured memory table: {exc}"
            raise MemoryStoreError(msg) from exc

        self._table_ready = True

    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Store or upsert a key-value pair.

        If ``metadata`` contains a ``"ttl_seconds"`` key, that value is
        used as the TTL for this entry. Otherwise, the default TTL is used.
        """
        await self._ensure_table()
        pool = await self._ensure_pool()

        meta = dict(metadata) if metadata else {}
        ttl_seconds = meta.pop("ttl_seconds", self._default_ttl_seconds)

        value_json = json.dumps(value)
        meta_json = json.dumps(meta)

        if ttl_seconds and ttl_seconds > 0:
            ttl_interval = timedelta(seconds=int(ttl_seconds))
            sql = f"""
                INSERT INTO {self._table_name} (key, value, metadata, agent_namespace, ttl_at)
                VALUES ($1, $2::jsonb, $3::jsonb, $4, NOW() + $5::interval)
                ON CONFLICT (key, agent_namespace)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    metadata = EXCLUDED.metadata,
                    ttl_at = EXCLUDED.ttl_at,
                    updated_at = NOW()
            """  # noqa: S608
            try:
                async with pool.acquire() as conn:
                    await conn.execute(sql, key, value_json, meta_json, self._agent_namespace, ttl_interval)
            except Exception as exc:
                msg = f"Failed to store key {key!r}: {exc}"
                raise MemoryStoreError(msg) from exc
        else:
            sql = f"""
                INSERT INTO {self._table_name} (key, value, metadata, agent_namespace, ttl_at)
                VALUES ($1, $2::jsonb, $3::jsonb, $4, NULL)
                ON CONFLICT (key, agent_namespace)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    metadata = EXCLUDED.metadata,
                    ttl_at = NULL,
                    updated_at = NOW()
            """  # noqa: S608
            try:
                async with pool.acquire() as conn:
                    await conn.execute(sql, key, value_json, meta_json, self._agent_namespace)
            except Exception as exc:
                msg = f"Failed to store key {key!r}: {exc}"
                raise MemoryStoreError(msg) from exc

    async def retrieve(self, key: str) -> Any | None:
        """Retrieve a value by key, excluding expired entries."""
        await self._ensure_table()
        pool = await self._ensure_pool()

        sql = f"""
            SELECT value FROM {self._table_name}
            WHERE key = $1
              AND agent_namespace = $2
              AND (ttl_at IS NULL OR ttl_at > NOW())
        """  # noqa: S608

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(sql, key, self._agent_namespace)
        except Exception as exc:
            msg = f"Failed to retrieve key {key!r}: {exc}"
            raise MemoryRetrieveError(msg) from exc

        if row is None:
            return None

        raw_value = row["value"]
        if isinstance(raw_value, str):
            return json.loads(raw_value)
        return raw_value

    async def search(self, query: str, top_k: int = 5) -> list[MemoryResult]:
        """Case-insensitive substring search on keys and values.

        Scoring:
        - ``1.0`` for an exact key match.
        - ``0.5`` for a substring match on key or value text.
        """
        await self._ensure_table()
        pool = await self._ensure_pool()

        sql = f"""
            SELECT key, value, metadata,
                   CASE
                       WHEN LOWER(key) = LOWER($1) THEN 1.0
                       ELSE 0.5
                   END AS score
            FROM {self._table_name}
            WHERE agent_namespace = $2
              AND (ttl_at IS NULL OR ttl_at > NOW())
              AND (
                  key ILIKE '%' || $1 || '%'
                  OR value::text ILIKE '%' || $1 || '%'
              )
            ORDER BY score DESC
            LIMIT $3
        """  # noqa: S608

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, query, self._agent_namespace, top_k)
        except Exception as exc:
            msg = f"Failed to search: {exc}"
            raise MemoryRetrieveError(msg) from exc

        results: list[MemoryResult] = []
        for row in rows:
            raw_value = row["value"]
            value = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
            meta_raw = row["metadata"]
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            results.append(
                MemoryResult(
                    key=row["key"],
                    value=value,
                    score=float(row["score"]),
                    metadata=meta,
                )
            )
        return results

    async def delete(self, key: str) -> bool:
        """Delete an entry by key."""
        await self._ensure_table()
        pool = await self._ensure_pool()

        sql = f"""
            DELETE FROM {self._table_name}
            WHERE key = $1 AND agent_namespace = $2
        """  # noqa: S608

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(sql, key, self._agent_namespace)
        except Exception as exc:
            msg = f"Failed to delete key {key!r}: {exc}"
            raise MemoryStoreError(msg) from exc

        return str(result) != "DELETE 0"

    async def clear(self) -> None:
        """Remove all entries for the configured namespace."""
        await self._ensure_table()
        pool = await self._ensure_pool()

        sql = f"""
            DELETE FROM {self._table_name}
            WHERE agent_namespace = $1
        """  # noqa: S608

        try:
            async with pool.acquire() as conn:
                await conn.execute(sql, self._agent_namespace)
        except Exception as exc:
            msg = f"Failed to clear table: {exc}"
            raise MemoryStoreError(msg) from exc

    async def health_check(self) -> bool:
        """Test database connectivity."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
        except Exception:
            logger.exception("Structured memory health check failed")
            return False
        return True

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._table_ready = False
