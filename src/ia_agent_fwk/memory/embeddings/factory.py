"""Embedding provider factory with lazy registration."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, ClassVar

from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.exceptions import MemoryConfigError

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)

# Type alias for registry values: either a concrete class or a lazy string path.
_RegistryEntry = type[EmbeddingProvider] | str


class EmbeddingFactory:
    """Configuration-driven embedding provider factory.

    Follows the ``MemoryFactory`` pattern with a ``ClassVar`` registry
    and lazy colon-delimited dotted-path imports.
    """

    _registry: ClassVar[dict[str, _RegistryEntry]] = {
        "openai": "ia_agent_fwk.memory.embeddings.openai:OpenAIEmbeddingProvider",
        "ollama": "ia_agent_fwk.memory.embeddings.ollama:OllamaEmbeddingProvider",
    }

    @classmethod
    def register(
        cls,
        name: str,
        provider_cls: type[EmbeddingProvider] | str,
        *,
        replace: bool = False,
    ) -> None:
        """Register a provider class (or lazy dotted path) under *name*.

        Parameters
        ----------
        name:
            Logical name for the provider (e.g. ``"openai"``).
        provider_cls:
            Concrete class or lazy ``"module.path:ClassName"`` string.
        replace:
            If ``True``, silently overwrite an existing registration.

        """
        if name in cls._registry and not replace:
            msg = f"Embedding provider '{name}' is already registered. Use replace=True to overwrite."
            raise MemoryConfigError(msg)
        cls._registry[name] = provider_cls

    @classmethod
    def create(cls, settings: EmbeddingSettings) -> EmbeddingProvider:
        """Instantiate and return the configured embedding provider.

        Parameters
        ----------
        settings:
            The embedding configuration section.

        """
        name = settings.provider
        entry = cls._registry.get(name)

        if entry is None:
            valid = ", ".join(sorted(cls._registry))
            msg = f"Unknown embedding provider '{name}'. Valid providers: {valid}"
            raise MemoryConfigError(msg)

        provider_cls = cls._resolve(entry, name=name)

        if name == "openai":
            return provider_cls(api_key=settings.api_key, model=settings.model)  # type: ignore[call-arg]

        if name == "ollama":
            return provider_cls(  # type: ignore[call-arg]
                base_url=settings.base_url,
                model=settings.model,
            )

        # Generic fallback for custom providers
        return provider_cls()

    @classmethod
    def _resolve(cls, entry: _RegistryEntry, name: str | None = None) -> type[EmbeddingProvider]:
        """Resolve a registry entry to a concrete class."""
        if isinstance(entry, str):
            module_path, _, attr_name = entry.rpartition(":")
            module = importlib.import_module(module_path)
            klass: type[EmbeddingProvider] = getattr(module, attr_name)
            if name is not None:
                cls._registry[name] = klass
            return klass
        return entry
