"""PostgreSQL + pgvector memory backend.

Stores text content with vector embeddings for semantic similarity search.
Uses ``asyncpg`` for async PostgreSQL access and pgvector's ``<=>``
(cosine distance) operator for vector search.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ia_agent_fwk.memory.backends._validation import validate_sql_identifier
from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.exceptions import MemoryConfigError, MemoryRetrieveError, MemoryStoreError
from ia_agent_fwk.memory.models import MemoryResult

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore[assignment,unused-ignore]

if TYPE_CHECKING:
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class PgVectorMemoryBackend(MemoryBackend):
    """PostgreSQL + pgvector memory backend.

    Parameters
    ----------
    database_url:
        PostgreSQL connection string (asyncpg format).
    embedding_provider:
        Provider for generating embeddings.
    collection_name:
        Table name for vector storage.
    embedding_dimensions:
        Vector dimension. Must match ``embedding_provider.dimension()``.
    agent_namespace:
        Namespace for multi-agent isolation.
    pool_min_size:
        Minimum connection pool size.
    pool_max_size:
        Maximum connection pool size.

    """

    def __init__(  # noqa: PLR0913
        self,
        database_url: str,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "memory_vectors",
        embedding_dimensions: int = 1536,
        agent_namespace: str = "default",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        if embedding_provider.dimension() != embedding_dimensions:
            msg = (
                f"Embedding provider dimension ({embedding_provider.dimension()}) "
                f"does not match configured embedding_dimensions ({embedding_dimensions})"
            )
            raise MemoryConfigError(msg)

        validate_sql_identifier(collection_name, label="collection name")

        self._database_url = database_url
        self._embedding_provider = embedding_provider
        self._collection_name = collection_name
        self._embedding_dimensions = embedding_dimensions
        self._agent_namespace = agent_namespace
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None
        self._table_ready = False

    @property
    def backend_type(self) -> str:
        """Return ``'pgvector'``."""
        return "pgvector"

    async def _ensure_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        """Create the connection pool on first use."""
        if self._pool is None:
            if asyncpg is None:  # pragma: no cover
                msg = "asyncpg is required for PgVectorMemoryBackend"
                raise MemoryConfigError(msg)

            try:
                self._pool = await asyncpg.create_pool(
                    self._database_url,
                    min_size=self._pool_min_size,
                    max_size=self._pool_max_size,
                )
            except Exception as exc:
                msg = f"Failed to create pgvector connection pool: {exc}"
                raise MemoryStoreError(msg) from exc
        return self._pool

    async def _ensure_table(self) -> None:
        """Create the vector table and indexes if they do not exist."""
        if self._table_ready:
            return

        pool = await self._ensure_pool()
        table = self._collection_name
        dims = self._embedding_dimensions

        create_table = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key             VARCHAR(512) NOT NULL,
                value           TEXT,
                embedding       VECTOR({dims}),
                metadata        JSONB DEFAULT '{{}}'::jsonb,
                agent_namespace VARCHAR(128) NOT NULL DEFAULT 'default',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """

        create_unique_idx = f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_key_ns
                ON {table} (key, agent_namespace)
        """

        create_hnsw_idx = f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_embedding
                ON {table}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 128)
        """

        create_ns_idx = f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_namespace
                ON {table} (agent_namespace)
        """

        try:
            async with pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(create_table)
                await conn.execute(create_unique_idx)
                await conn.execute(create_hnsw_idx)
                await conn.execute(create_ns_idx)
        except Exception as exc:
            msg = f"Failed to create pgvector table: {exc}"
            raise MemoryStoreError(msg) from exc

        self._table_ready = True

    @staticmethod
    def _vector_to_sql(vector: list[float]) -> str:
        """Convert a list of floats to pgvector text format."""
        return "[" + ",".join(str(v) for v in vector) + "]"

    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Store a value with auto-generated embedding.

        If ``metadata`` contains an ``"embedding"`` key, the pre-computed
        vector is used instead of calling the embedding provider.
        """
        await self._ensure_table()
        pool = await self._ensure_pool()

        meta = dict(metadata) if metadata else {}

        # Allow pre-computed embeddings via metadata
        embedding: list[float] | None = meta.pop("embedding", None)
        if embedding is None:
            text_to_embed = value if isinstance(value, str) else json.dumps(value)
            vectors = await self._embedding_provider.embed([text_to_embed])
            embedding = vectors[0]

        embedding_sql = self._vector_to_sql(embedding)
        value_str = value if isinstance(value, str) else json.dumps(value)
        meta_json = json.dumps(meta)

        sql = f"""
            INSERT INTO {self._collection_name} (key, value, embedding, metadata, agent_namespace)
            VALUES ($1, $2, $3::vector, $4::jsonb, $5)
            ON CONFLICT (key, agent_namespace)
            DO UPDATE SET
                value = EXCLUDED.value,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """  # noqa: S608

        try:
            async with pool.acquire() as conn:
                await conn.execute(sql, key, value_str, embedding_sql, meta_json, self._agent_namespace)
        except Exception as exc:
            msg = f"Failed to store key {key!r}: {exc}"
            raise MemoryStoreError(msg) from exc

    async def retrieve(self, key: str) -> Any | None:
        """Retrieve a value by exact key lookup."""
        await self._ensure_table()
        pool = await self._ensure_pool()

        sql = f"""
            SELECT value FROM {self._collection_name}
            WHERE key = $1 AND agent_namespace = $2
        """  # noqa: S608

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(sql, key, self._agent_namespace)
        except Exception as exc:
            msg = f"Failed to retrieve key {key!r}: {exc}"
            raise MemoryRetrieveError(msg) from exc

        if row is None:
            return None
        return row["value"]

    async def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryResult]:
        """Perform cosine similarity search.

        Parameters
        ----------
        query:
            Text to search for (will be embedded).
        top_k:
            Maximum number of results.
        **kwargs:
            Optional ``score_threshold`` (float) and ``metadata_filter`` (dict).

        """
        await self._ensure_table()
        pool = await self._ensure_pool()

        score_threshold: float = kwargs.get("score_threshold", 0.0)
        metadata_filter: dict[str, Any] | None = kwargs.get("metadata_filter")

        vectors = await self._embedding_provider.embed([query])
        query_embedding = self._vector_to_sql(vectors[0])

        # Cosine distance: 1 - distance = similarity score
        where_clauses = ["agent_namespace = $2"]
        params: list[Any] = [query_embedding, self._agent_namespace]

        if score_threshold > 0:
            params.append(score_threshold)
            where_clauses.append(f"(1 - (embedding <=> $1::vector)) >= ${len(params)}")

        if metadata_filter:
            for filter_key, filter_value in metadata_filter.items():
                validate_sql_identifier(filter_key, label="metadata filter key")
                params.append(json.dumps(filter_value))
                where_clauses.append(f"metadata->>'{filter_key}' = ${len(params)}")

        params.append(top_k)
        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT key, value, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM {self._collection_name}
            WHERE {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT ${len(params)}
        """  # noqa: S608

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except Exception as exc:
            msg = f"Failed to search: {exc}"
            raise MemoryRetrieveError(msg) from exc

        results: list[MemoryResult] = []
        for row in rows:
            meta_raw = row["metadata"]
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            results.append(
                MemoryResult(
                    key=row["key"],
                    value=row["value"],
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
            DELETE FROM {self._collection_name}
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
            DELETE FROM {self._collection_name}
            WHERE agent_namespace = $1
        """  # noqa: S608

        try:
            async with pool.acquire() as conn:
                await conn.execute(sql, self._agent_namespace)
        except Exception as exc:
            msg = f"Failed to clear collection: {exc}"
            raise MemoryStoreError(msg) from exc

    async def health_check(self) -> bool:
        """Test database connectivity."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
        except Exception:
            logger.exception("pgvector health check failed")
            return False
        return True

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._table_ready = False
