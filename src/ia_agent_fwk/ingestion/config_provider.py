"""RAG config provider — loads pipeline configuration from PostgreSQL.

Separates behavior (code) from configuration (data). Values come from
a `rag_config` table, scoped by tenant_id. Each tenant gets:
  - Its own config (synonyms, prompts, models, thresholds)
  - Its own Qdrant collection ("{tenant_id}_rag")
  - Its own scope rules and agent permissions

Can be changed from a backoffice without redeploy.

Usage:
    provider = RagConfigProvider(
        database_url="postgresql://user:pass@localhost/db",
        tenant_id="webdelseguro",
    )
    await provider.load()

    synonyms = provider.get("query_expansion.synonyms", default={})
    collection = provider.collection_name  # "webdelseguro_rag"
    scopes = provider.resolve_scopes("automotor/manuals/Allianz/AUTOS.pdf")
    filters = provider.get_agent_filter("sales-agent")

Table DDL:
    CREATE TABLE rag_config (
        tenant_id    TEXT NOT NULL,
        config_key   TEXT NOT NULL,
        config_value JSONB NOT NULL,
        updated_at   TIMESTAMPTZ DEFAULT NOW(),
        updated_by   TEXT,
        PRIMARY KEY (tenant_id, config_key)
    );
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RagConfigProvider:
    """Load and cache RAG configuration from PostgreSQL, scoped by tenant.

    Parameters
    ----------
    database_url:
        PostgreSQL connection string.
    tenant_id:
        Tenant identifier. Determines which config rows to load
        and which Qdrant collection to use.
    cache_ttl_seconds:
        How long to cache config in memory before re-reading from DB.

    """

    def __init__(
        self,
        database_url: str,
        tenant_id: str = "default",
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._database_url = database_url
        self._tenant_id = tenant_id
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, Any] = {}
        self._cache_loaded_at: float = 0

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def collection_name(self) -> str:
        """Qdrant collection name for this tenant."""
        return f"{self._tenant_id}_rag"

    # ═══════════════════════════════════════════════════════════════════════
    # Load / cache
    # ═══════════════════════════════════════════════════════════════════════

    async def load(self) -> None:
        """Load config for this tenant from the database."""
        try:
            import asyncpg  # noqa: PLC0415

            conn = await asyncpg.connect(self._database_url)
            try:
                rows = await conn.fetch(
                    "SELECT config_key, config_value FROM rag_config WHERE tenant_id = $1",
                    self._tenant_id,
                )
                self._cache = {row["config_key"]: json.loads(row["config_value"]) for row in rows}
                self._cache_loaded_at = time.monotonic()
                logger.info(
                    "Loaded %d config keys for tenant '%s'",
                    len(self._cache),
                    self._tenant_id,
                )
            finally:
                await conn.close()
        except ImportError:
            logger.warning("asyncpg not installed — using defaults only")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load config for tenant '%s': %s", self._tenant_id, exc)

    async def _ensure_fresh(self) -> None:
        """Reload cache if stale."""
        if self._cache_ttl > 0 and (time.monotonic() - self._cache_loaded_at) > self._cache_ttl:
            await self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value. Returns *default* if key doesn't exist."""
        return self._cache.get(key, default)

    def get_all(self) -> dict[str, Any]:
        """Get all cached config as a dict."""
        return dict(self._cache)

    # ═══════════════════════════════════════════════════════════════════════
    # Document scoping
    # ═══════════════════════════════════════════════════════════════════════

    def resolve_scopes(self, file_path: str | Path) -> list[str]:
        """Determine which scopes a document belongs to.

        Matches the file path against scope_rules from the DB.
        Config key: "scope_rules"
        Format: {"path_prefix": ["scope1", "scope2"], ...}

        Returns ["default"] if no rules match.
        """
        rules: dict[str, list[str]] = self.get("scope_rules", {})
        path_str = str(file_path)

        matched_scopes: set[str] = set()
        for prefix, scopes in rules.items():
            if prefix in path_str:
                matched_scopes.update(scopes)

        return sorted(matched_scopes) if matched_scopes else ["default"]

    def get_agent_scopes(self, agent_name: str) -> list[str]:
        """Get the scopes an agent is allowed to access.

        Config key: "agent_scopes"
        Format: {"agent_name": ["scope1", "scope2"], ...}

        Returns ["default"] if the agent is not configured.
        """
        agent_scopes: dict[str, list[str]] = self.get("agent_scopes", {})
        return agent_scopes.get(agent_name, ["default"])

    def get_agent_filter(self, agent_name: str) -> dict[str, Any]:
        """Build a Qdrant metadata filter for an agent's allowed scopes.

        For single scope: {"scope": "manuales"}
        For multiple: {"scope__in": ["manuales", "legal"]}
        """
        scopes = self.get_agent_scopes(agent_name)
        if len(scopes) == 1:
            return {"scope": scopes[0]}
        return {"scope__in": scopes}

    # ═══════════════════════════════════════════════════════════════════════
    # Write / setup
    # ═══════════════════════════════════════════════════════════════════════

    async def ensure_table(self) -> None:
        """Create the rag_config table if it doesn't exist."""
        try:
            import asyncpg  # noqa: PLC0415

            conn = await asyncpg.connect(self._database_url)
            try:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS rag_config (
                        tenant_id    TEXT NOT NULL,
                        config_key   TEXT NOT NULL,
                        config_value JSONB NOT NULL,
                        updated_at   TIMESTAMPTZ DEFAULT NOW(),
                        updated_by   TEXT,
                        PRIMARY KEY (tenant_id, config_key)
                    )
                """)
                logger.info("rag_config table ensured")
            finally:
                await conn.close()
        except ImportError:
            logger.warning("asyncpg not installed — cannot create table")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create rag_config table: %s", exc)

    async def set(self, key: str, value: Any, updated_by: str = "") -> None:
        """Set a config value for this tenant."""
        try:
            import asyncpg  # noqa: PLC0415

            conn = await asyncpg.connect(self._database_url)
            try:
                await conn.execute(
                    """
                    INSERT INTO rag_config (tenant_id, config_key, config_value, updated_at, updated_by)
                    VALUES ($1, $2, $3::jsonb, NOW(), $4)
                    ON CONFLICT (tenant_id, config_key)
                    DO UPDATE SET config_value = $3::jsonb, updated_at = NOW(), updated_by = $4
                    """,
                    self._tenant_id,
                    key,
                    json.dumps(value),
                    updated_by,
                )
                self._cache[key] = value
                logger.info("Set config '%s' for tenant '%s'", key, self._tenant_id)
            finally:
                await conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to set config %s: %s", key, exc)

    async def seed_defaults(self, defaults: dict[str, Any]) -> None:
        """Insert default values for keys that don't exist yet.

        Safe to call on every startup — won't overwrite existing values.
        """
        try:
            import asyncpg  # noqa: PLC0415

            conn = await asyncpg.connect(self._database_url)
            try:
                for key, value in defaults.items():
                    await conn.execute(
                        """
                        INSERT INTO rag_config (tenant_id, config_key, config_value, updated_by)
                        VALUES ($1, $2, $3::jsonb, 'seed_defaults')
                        ON CONFLICT (tenant_id, config_key) DO NOTHING
                        """,
                        self._tenant_id,
                        key,
                        json.dumps(value),
                    )
                logger.info(
                    "Seeded %d default config keys for tenant '%s'",
                    len(defaults),
                    self._tenant_id,
                )
            finally:
                await conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to seed defaults: %s", exc)
