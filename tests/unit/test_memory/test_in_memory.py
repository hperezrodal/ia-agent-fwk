"""Tests for InMemoryBackend."""

from __future__ import annotations

import pytest

from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend


@pytest.mark.unit
class TestInMemoryBackend:
    async def test_store_and_retrieve(self, in_memory_backend: InMemoryBackend):
        await in_memory_backend.store("key1", "value1")
        result = await in_memory_backend.retrieve("key1")
        assert result == "value1"

    async def test_retrieve_nonexistent_key(self, in_memory_backend: InMemoryBackend):
        result = await in_memory_backend.retrieve("nonexistent")
        assert result is None

    async def test_store_overwrite(self, in_memory_backend: InMemoryBackend):
        await in_memory_backend.store("key1", "value1")
        await in_memory_backend.store("key1", "value2")
        result = await in_memory_backend.retrieve("key1")
        assert result == "value2"

    async def test_delete_existing_key(self, in_memory_backend: InMemoryBackend):
        await in_memory_backend.store("key1", "value1")
        deleted = await in_memory_backend.delete("key1")
        assert deleted is True
        result = await in_memory_backend.retrieve("key1")
        assert result is None

    async def test_delete_nonexistent_key(self, in_memory_backend: InMemoryBackend):
        deleted = await in_memory_backend.delete("nonexistent")
        assert deleted is False

    async def test_clear(self, in_memory_backend: InMemoryBackend):
        await in_memory_backend.store("key1", "value1")
        await in_memory_backend.store("key2", "value2")
        await in_memory_backend.clear()
        assert await in_memory_backend.retrieve("key1") is None
        assert await in_memory_backend.retrieve("key2") is None

    async def test_search_exact_key_match(self, in_memory_backend: InMemoryBackend):
        await in_memory_backend.store("hello", "world")
        results = await in_memory_backend.search("hello")
        assert len(results) == 1
        assert results[0].key == "hello"
        assert results[0].score == 1.0

    async def test_search_substring_match(self, in_memory_backend: InMemoryBackend):
        await in_memory_backend.store("key1", "hello world")
        results = await in_memory_backend.search("world")
        assert len(results) == 1
        assert results[0].key == "key1"
        assert results[0].score == 0.5

    async def test_search_no_match(self, in_memory_backend: InMemoryBackend):
        await in_memory_backend.store("key1", "value1")
        results = await in_memory_backend.search("nonexistent")
        assert results == []

    async def test_search_respects_top_k(self, in_memory_backend: InMemoryBackend):
        for i in range(5):
            await in_memory_backend.store(f"item_{i}", f"value_{i}")
        results = await in_memory_backend.search("item", top_k=3)
        assert len(results) == 3

    async def test_lru_eviction(self):
        backend = InMemoryBackend(max_items=3)
        await backend.store("a", 1)
        await backend.store("b", 2)
        await backend.store("c", 3)
        # Adding a fourth entry should evict "a" (oldest)
        await backend.store("d", 4)
        assert await backend.retrieve("a") is None
        assert await backend.retrieve("b") == 2
        assert await backend.retrieve("c") == 3
        assert await backend.retrieve("d") == 4

    async def test_lru_access_refreshes(self):
        backend = InMemoryBackend(max_items=3)
        await backend.store("a", 1)
        await backend.store("b", 2)
        await backend.store("c", 3)
        # Access "a" to refresh it, making "b" the oldest
        await backend.retrieve("a")
        # Adding "d" should evict "b" (now oldest)
        await backend.store("d", 4)
        assert await backend.retrieve("a") == 1
        assert await backend.retrieve("b") is None
        assert await backend.retrieve("c") == 3
        assert await backend.retrieve("d") == 4

    async def test_health_check(self, in_memory_backend: InMemoryBackend):
        result = await in_memory_backend.health_check()
        assert result is True

    async def test_backend_type(self, in_memory_backend: InMemoryBackend):
        assert in_memory_backend.backend_type == "in_memory"

    async def test_store_with_metadata(self, in_memory_backend: InMemoryBackend):
        await in_memory_backend.store("key1", "value1", metadata={"tag": "important"})
        # Verify metadata is preserved via search (which exposes it)
        results = await in_memory_backend.search("key1")
        assert len(results) == 1
        assert results[0].metadata == {"tag": "important"}
