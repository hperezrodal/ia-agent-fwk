"""In-process dict-based memory backend with LRU eviction.

Uses ``collections.OrderedDict`` for efficient LRU ordering.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any

from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.models import MemoryEntry, MemoryResult
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class InMemoryBackend(MemoryBackend):
    """In-process dict-based memory backend with LRU eviction.

    Parameters
    ----------
    max_items:
        Maximum number of entries. When exceeded, the oldest (least
        recently used) entry is evicted.

    """

    def __init__(self, max_items: int = 1000) -> None:
        self._max_items = max_items
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def backend_type(self) -> str:
        """Return ``'in_memory'``."""
        return "in_memory"

    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Store a value under *key*, evicting the oldest entry if full."""
        collector = get_metrics_collector()
        entry = MemoryEntry(key=key, value=value, metadata=metadata or {})

        async with self._lock:
            if key in self._store:
                self._store[key] = entry
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self._max_items:
                    self._store.popitem(last=False)
                    collector.increment("memory_evictions_total", labels={"backend": "in_memory"})
                self._store[key] = entry

        collector.increment("memory_operations_total", labels={"backend": "in_memory", "operation": "store"})

    async def retrieve(self, key: str) -> Any | None:
        """Retrieve a value by *key*, refreshing its LRU position.

        Returns ``None`` if the key does not exist.
        """
        collector = get_metrics_collector()
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                collector.increment(
                    "memory_operations_total", labels={"backend": "in_memory", "operation": "retrieve_miss"}
                )
                return None
            self._store.move_to_end(key)
        collector.increment("memory_operations_total", labels={"backend": "in_memory", "operation": "retrieve_hit"})
        return entry.value

    async def search(self, query: str, top_k: int = 5) -> list[MemoryResult]:
        """Case-insensitive substring search on keys and string values.

        Scoring:
        - ``1.0`` for an exact key match.
        - ``0.5`` for a substring match on the key or a string value.
        """
        collector = get_metrics_collector()
        start = time.monotonic()
        results: list[MemoryResult] = []
        query_lower = query.lower()

        async with self._lock:
            items = list(self._store.items())

        for key, entry in items:
            key_lower = key.lower()

            if key_lower == query_lower:
                results.append(MemoryResult(key=key, value=entry.value, score=1.0, metadata=entry.metadata))
            elif query_lower in key_lower or (isinstance(entry.value, str) and query_lower in entry.value.lower()):
                results.append(MemoryResult(key=key, value=entry.value, score=0.5, metadata=entry.metadata))

        # Sort by score descending, then take top_k
        results.sort(key=lambda r: r.score, reverse=True)
        duration_ms = (time.monotonic() - start) * 1000
        collector.increment("memory_operations_total", labels={"backend": "in_memory", "operation": "search"})
        collector.observe("memory_search_duration_seconds", duration_ms / 1000, labels={"backend": "in_memory"})
        collector.observe("memory_search_results_count", len(results[:top_k]), labels={"backend": "in_memory"})
        return results[:top_k]

    async def delete(self, key: str) -> bool:
        """Delete an entry by *key*. Returns ``True`` if it existed."""
        collector = get_metrics_collector()
        async with self._lock:
            if key in self._store:
                del self._store[key]
                collector.increment("memory_operations_total", labels={"backend": "in_memory", "operation": "delete"})
                return True
        return False

    async def clear(self) -> None:
        """Remove all stored entries."""
        async with self._lock:
            self._store.clear()

    async def health_check(self) -> bool:
        """Return ``True`` (in-memory backend is always healthy)."""
        return True
