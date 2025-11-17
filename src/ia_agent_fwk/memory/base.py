"""Abstract base class for memory backends.

All concrete backends (in-memory, conversation, vector, ...) extend
``MemoryBackend`` and implement every abstract method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ia_agent_fwk.memory.models import MemoryResult


class MemoryBackend(ABC):
    """Abstract base class for all memory backends.

    Subclasses must implement the five core operations (``store``,
    ``retrieve``, ``search``, ``delete``, ``clear``) and the
    ``backend_type`` property.
    """

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type identifier (e.g. ``'in_memory'``, ``'conversation'``)."""
        ...

    @abstractmethod
    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Store a value under the given key."""
        ...

    @abstractmethod
    async def retrieve(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns ``None`` if not found."""
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[MemoryResult]:
        """Search for entries matching *query*. Returns up to *top_k* results."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an entry by key. Returns ``True`` if it existed, ``False`` otherwise."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all stored entries."""
        ...

    async def health_check(self) -> bool:
        """Check if the backend is operational. Returns ``True`` if healthy."""
        return True

    async def close(self) -> None:  # noqa: B027
        """Release any resources held by the backend. Default: no-op."""
