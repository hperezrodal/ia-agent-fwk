"""LLM provider factory with lazy registration.

Built-in providers are registered as dotted module paths and imported only
when ``create()`` is called, avoiding eager SDK imports.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, ClassVar

from ia_agent_fwk.config.settings import LLMProviderSettings, LLMSettings
from ia_agent_fwk.llm.base import LLMProvider
from ia_agent_fwk.llm.exceptions import LLMConfigError
from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# Type alias for registry values: either a concrete class or a lazy string path.
_RegistryEntry = type[LLMProvider] | str


class LLMProviderFactory:
    """Configuration-driven provider factory.

    Five built-in providers are pre-registered using lazy module paths:

    * ``openai``  -> ``ia_agent_fwk.llm.providers.openai.OpenAIProvider``
    * ``anthropic`` -> ``ia_agent_fwk.llm.providers.anthropic.AnthropicProvider``
    * ``ollama``  -> ``ia_agent_fwk.llm.providers.ollama.OllamaProvider``
    * ``vllm``  -> ``ia_agent_fwk.llm.providers.vllm.VLLMProvider``
    * ``huggingface`` -> ``ia_agent_fwk.llm.providers.huggingface.HuggingFaceProvider``
    """

    _registry: ClassVar[dict[str, _RegistryEntry]] = {
        "openai": "ia_agent_fwk.llm.providers.openai:OpenAIProvider",
        "anthropic": "ia_agent_fwk.llm.providers.anthropic:AnthropicProvider",
        "ollama": "ia_agent_fwk.llm.providers.ollama:OllamaProvider",
        "vllm": "ia_agent_fwk.llm.providers.vllm:VLLMProvider",
        "huggingface": "ia_agent_fwk.llm.providers.huggingface:HuggingFaceProvider",
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, provider_cls: type[LLMProvider] | str, *, replace: bool = False) -> None:
        """Register a provider class (or lazy dotted path) under *name*.

        Parameters
        ----------
        name:
            Logical name for the provider (e.g. ``"openai"``).
        provider_cls:
            Concrete class or lazy dotted path.
        replace:
            If ``True``, silently overwrite an existing registration.
            If ``False`` (default), raise ``LLMConfigError`` on duplicates.

        """
        if name in cls._registry and not replace:
            msg = f"Provider '{name}' is already registered. Use replace=True to overwrite."
            raise LLMConfigError(msg)
        cls._registry[name] = provider_cls

    @classmethod
    def create(
        cls,
        settings: LLMSettings,
        provider_name: str | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """Instantiate and return the requested provider.

        Parameters
        ----------
        settings:
            The ``llm`` section of the application settings.
        provider_name:
            Explicit provider name.  Falls back to
            ``settings.default_provider``.
        **kwargs:
            Extra keyword arguments forwarded to the provider constructor.

        """
        name = provider_name or settings.default_provider
        entry = cls._registry.get(name)

        if entry is None:
            valid = ", ".join(sorted(cls._registry))
            msg = f"Unknown LLM provider '{name}'. Valid providers: {valid}"
            raise LLMConfigError(msg)

        provider_cls = cls._resolve(entry)

        # Look up provider-specific settings.
        provider_settings = settings.providers.get(name, LLMProviderSettings())

        collector = get_metrics_collector()
        collector.increment(
            "llm_provider_created_total",
            labels={"provider": name},
        )

        return provider_cls(settings=provider_settings, provider_name=name, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve(cls, entry: _RegistryEntry) -> type[LLMProvider]:
        """Resolve a registry entry to a concrete class."""
        if isinstance(entry, str):
            module_path, _, attr_name = entry.rpartition(":")
            module = importlib.import_module(module_path)
            klass: type[LLMProvider] = getattr(module, attr_name)
            return klass
        return entry
