"""Tests for memory exception hierarchy."""

from __future__ import annotations

import pytest

from ia_agent_fwk.memory.exceptions import (
    MemoryBackendError,
    MemoryConfigError,
    MemoryRetrieveError,
    MemoryStoreError,
)


@pytest.mark.unit
class TestMemoryExceptionHierarchy:
    def test_memory_backend_error_is_base(self):
        assert issubclass(MemoryBackendError, Exception)

    def test_memory_backend_error_does_not_shadow_builtin(self):
        """Ensure MemoryBackendError does not shadow Python's built-in MemoryError."""
        assert MemoryBackendError is not MemoryError

    def test_memory_config_error_inherits(self):
        assert issubclass(MemoryConfigError, MemoryBackendError)

    def test_memory_store_error_inherits(self):
        assert issubclass(MemoryStoreError, MemoryBackendError)

    def test_memory_retrieve_error_inherits(self):
        assert issubclass(MemoryRetrieveError, MemoryBackendError)

    def test_all_catchable_by_memory_backend_error(self):
        for exc_cls in (MemoryConfigError, MemoryStoreError, MemoryRetrieveError):
            with pytest.raises(MemoryBackendError):
                raise exc_cls("test")


@pytest.mark.unit
class TestExceptionMessages:
    def test_memory_backend_error_message(self):
        err = MemoryBackendError("something went wrong")
        assert str(err) == "something went wrong"

    def test_memory_config_error_message(self):
        err = MemoryConfigError("bad config")
        assert str(err) == "bad config"

    def test_memory_store_error_message(self):
        err = MemoryStoreError("store failed")
        assert str(err) == "store failed"

    def test_memory_retrieve_error_message(self):
        err = MemoryRetrieveError("retrieve failed")
        assert str(err) == "retrieve failed"
