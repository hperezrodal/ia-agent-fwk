"""Tests for MemoryBackend ABC and memory data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.models import MemoryEntry, MemoryResult


@pytest.mark.unit
class TestMemoryBackendABC:
    def test_memory_backend_is_abstract(self):
        with pytest.raises(TypeError):
            MemoryBackend()  # type: ignore[abstract]


@pytest.mark.unit
class TestMemoryResultModel:
    def test_fields_and_defaults(self):
        result = MemoryResult(key="k1", value="v1")
        assert result.key == "k1"
        assert result.value == "v1"
        assert result.score == 0.0
        assert result.metadata is None

    def test_with_score_and_metadata(self):
        result = MemoryResult(key="k1", value="v1", score=0.8, metadata={"source": "test"})
        assert result.score == 0.8
        assert result.metadata == {"source": "test"}

    def test_frozen(self):
        result = MemoryResult(key="k1", value="v1")
        with pytest.raises(Exception):  # noqa: B017, PT011
            result.key = "k2"  # type: ignore[misc]


@pytest.mark.unit
class TestMemoryEntryModel:
    def test_fields_and_defaults(self):
        entry = MemoryEntry(key="k1", value="v1")
        assert entry.key == "k1"
        assert entry.value == "v1"
        assert entry.metadata == {}
        assert isinstance(entry.created_at, datetime)

    def test_created_at_is_utc(self):
        entry = MemoryEntry(key="k1", value="v1")
        assert entry.created_at.tzinfo == timezone.utc  # noqa: UP017

    def test_with_metadata(self):
        entry = MemoryEntry(key="k1", value="v1", metadata={"tag": "test"})
        assert entry.metadata == {"tag": "test"}

    def test_frozen(self):
        entry = MemoryEntry(key="k1", value="v1")
        with pytest.raises(Exception):  # noqa: B017, PT011
            entry.key = "k2"  # type: ignore[misc]
