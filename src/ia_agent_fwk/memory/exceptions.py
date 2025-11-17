"""Memory system exception hierarchy.

All memory-specific exceptions inherit from ``MemoryBackendError`` which itself
inherits from the built-in ``Exception``.
"""

from __future__ import annotations


class MemoryBackendError(Exception):
    """Base exception for all memory errors."""


class MemoryConfigError(MemoryBackendError):
    """Raised for memory configuration errors."""


class MemoryStoreError(MemoryBackendError):
    """Raised when storing a memory entry fails."""


class MemoryRetrieveError(MemoryBackendError):
    """Raised when retrieving a memory entry fails."""
