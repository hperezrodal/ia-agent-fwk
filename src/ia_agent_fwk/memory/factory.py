"""Memory backend factory with lazy registration.

Built-in backends are registered as dotted module paths and imported only
when ``create()`` is called, avoiding eager imports.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, ClassVar

from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.exceptions import MemoryConfigError

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import EmbeddingSettings, MemorySettings
    from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Type alias for registry values: either a concrete class or a lazy string path.
_RegistryEntry = type[MemoryBackend] | str


class MemoryFactory:
    """Configuration-driven memory backend factory.

    Follows the ``LLMProviderFactory`` pattern with a ``ClassVar`` registry
    and lazy colon-delimited dotted-path imports.

    Six built-in backends are pre-registered:

    * ``in_memory``    -> ``InMemoryBackend``
    * ``conversation`` -> ``ConversationMemoryBackend``
    * ``pgvector``     -> ``PgVectorMemoryBackend``
    * ``qdrant``       -> ``QdrantMemoryBackend``
    * ``structured``   -> ``StructuredMemoryBackend``
    * ``weaviate``     -> ``WeaviateMemoryBackend``
    """

    _registry: ClassVar[dict[str, _RegistryEntry]] = {
        "in_memory": "ia_agent_fwk.memory.backends.in_memory:InMemoryBackend",
        "conversation": "ia_agent_fwk.memory.backends.conversation:ConversationMemoryBackend",
        "pgvector": "ia_agent_fwk.memory.backends.pgvector:PgVectorMemoryBackend",
        "qdrant": "ia_agent_fwk.memory.backends.qdrant:QdrantMemoryBackend",
        "structured": "ia_agent_fwk.memory.backends.structured:StructuredMemoryBackend",
        "weaviate": "ia_agent_fwk.memory.backends.weaviate_backend:WeaviateMemoryBackend",
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        backend_cls: type[MemoryBackend] | str,
        *,
        replace: bool = False,
    ) -> None:
        """Register a backend class (or lazy dotted path) under *name*.

        Parameters
        ----------
        name:
            Logical name for the backend (e.g. ``"in_memory"``).
        backend_cls:
            Concrete class or lazy ``"module.path:ClassName"`` string.
        replace:
            If ``True``, silently overwrite an existing registration.
            If ``False`` (default), raise ``MemoryConfigError`` on duplicates.

        """
        if name in cls._registry and not replace:
            msg = f"Backend '{name}' is already registered. Use replace=True to overwrite."
            raise MemoryConfigError(msg)
        cls._registry[name] = backend_cls

    @classmethod
    def create(cls, settings: MemorySettings) -> MemoryBackend:  # noqa: PLR0911
        """Instantiate and return the default memory backend.

        Parameters
        ----------
        settings:
            The ``memory`` section of the application settings.

        """
        name = settings.default_backend
        entry = cls._registry.get(name)

        if entry is None:
            valid = ", ".join(sorted(cls._registry))
            msg = f"Unknown memory backend '{name}'. Valid backends: {valid}"
            raise MemoryConfigError(msg)

        backend_cls = cls._resolve(entry, name=name)

        # Pass backend-specific settings.
        # The registry guarantees the correct class for each name, so
        # keyword arguments are safe despite the base-class signature.
        if name == "in_memory":
            return backend_cls(max_items=settings.backends.in_memory.max_items)  # type: ignore[call-arg]
        if name == "conversation":
            return backend_cls(max_history=settings.backends.conversation.max_history)  # type: ignore[call-arg]
        if name == "pgvector":
            embedding_provider = cls._create_embedding_provider(settings.embedding)
            return backend_cls(  # type: ignore[call-arg]
                database_url=settings.backends.pgvector.database_url,
                embedding_provider=embedding_provider,
                collection_name=settings.backends.pgvector.collection_name,
                embedding_dimensions=settings.backends.pgvector.embedding_dimensions,
            )
        if name == "qdrant":
            embedding_provider = cls._create_embedding_provider(settings.embedding)
            return backend_cls(  # type: ignore[call-arg]
                url=settings.backends.qdrant.url,
                embedding_provider=embedding_provider,
                collection_name=settings.backends.qdrant.collection_name,
                embedding_dimensions=settings.backends.qdrant.embedding_dimensions,
            )
        if name == "structured":
            return backend_cls(  # type: ignore[call-arg]
                database_url=settings.backends.structured.database_url,
                table_name=settings.backends.structured.table_name,
                default_ttl_seconds=settings.backends.structured.default_ttl_seconds,
            )
        if name == "weaviate":
            embedding_provider = cls._create_embedding_provider(settings.embedding)
            return backend_cls(  # type: ignore[call-arg]
                url=settings.backends.weaviate.url,
                embedding_provider=embedding_provider,
                collection_name=settings.backends.weaviate.collection_name,
                embedding_dimensions=settings.backends.weaviate.embedding_dimensions,
            )

        # Generic fallback for custom backends
        return backend_cls()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _create_embedding_provider(cls, settings: EmbeddingSettings) -> EmbeddingProvider:
        """Create an embedding provider from settings.

        Parameters
        ----------
        settings:
            The embedding configuration section.

        """
        from ia_agent_fwk.memory.embeddings.factory import EmbeddingFactory  # noqa: PLC0415

        return EmbeddingFactory.create(settings)

    @classmethod
    def _resolve(cls, entry: _RegistryEntry, name: str | None = None) -> type[MemoryBackend]:
        """Resolve a registry entry to a concrete class.

        When *name* is provided and the entry is a lazy string path, the
        resolved class is cached back into the registry so subsequent calls
        skip the import overhead.
        """
        if isinstance(entry, str):
            module_path, _, attr_name = entry.rpartition(":")
            module = importlib.import_module(module_path)
            klass: type[MemoryBackend] = getattr(module, attr_name)
            if name is not None:
                cls._registry[name] = klass
            return klass
        return entry
