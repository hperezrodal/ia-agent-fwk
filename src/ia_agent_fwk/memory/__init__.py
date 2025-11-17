"""Memory subsystem: backends, models, and factory.

Public API
----------
.. autoclass:: MemoryBackend
.. autoclass:: MemoryEntry
.. autoclass:: MemoryResult
.. autoclass:: MemoryFactory
.. autoclass:: InMemoryBackend
.. autoclass:: ConversationMemoryBackend
.. autoclass:: ConversationInfo
.. autoclass:: ConversationMessage
.. autoclass:: EmbeddingProvider
.. autoclass:: OpenAIEmbeddingProvider
.. autoclass:: EmbeddingFactory
.. autoclass:: PgVectorMemoryBackend
.. autoclass:: QdrantMemoryBackend
.. autoclass:: StructuredMemoryBackend
.. autoclass:: WeaviateMemoryBackend
"""

from __future__ import annotations

from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend
from ia_agent_fwk.memory.backends.pgvector import PgVectorMemoryBackend
from ia_agent_fwk.memory.backends.qdrant import QdrantMemoryBackend
from ia_agent_fwk.memory.backends.structured import StructuredMemoryBackend
from ia_agent_fwk.memory.backends.weaviate_backend import WeaviateMemoryBackend
from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.embeddings.factory import EmbeddingFactory
from ia_agent_fwk.memory.embeddings.openai import OpenAIEmbeddingProvider
from ia_agent_fwk.memory.exceptions import (
    MemoryBackendError,
    MemoryConfigError,
    MemoryRetrieveError,
    MemoryStoreError,
)
from ia_agent_fwk.memory.factory import MemoryFactory
from ia_agent_fwk.memory.models import (
    ConversationInfo,
    ConversationMessage,
    MemoryEntry,
    MemoryResult,
)

__all__ = [
    "ConversationInfo",
    "ConversationMemoryBackend",
    "ConversationMessage",
    "EmbeddingFactory",
    "EmbeddingProvider",
    "InMemoryBackend",
    "MemoryBackend",
    "MemoryBackendError",
    "MemoryConfigError",
    "MemoryEntry",
    "MemoryFactory",
    "MemoryResult",
    "MemoryRetrieveError",
    "MemoryStoreError",
    "OpenAIEmbeddingProvider",
    "PgVectorMemoryBackend",
    "QdrantMemoryBackend",
    "StructuredMemoryBackend",
    "WeaviateMemoryBackend",
]
