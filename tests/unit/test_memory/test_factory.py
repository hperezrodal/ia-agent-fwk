"""Tests for MemoryFactory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from ia_agent_fwk.config.settings import EmbeddingSettings, MemorySettings
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend
from ia_agent_fwk.memory.backends.pgvector import PgVectorMemoryBackend
from ia_agent_fwk.memory.backends.qdrant import QdrantMemoryBackend
from ia_agent_fwk.memory.backends.structured import StructuredMemoryBackend
from ia_agent_fwk.memory.base import MemoryBackend
from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.exceptions import MemoryConfigError
from ia_agent_fwk.memory.factory import MemoryFactory

if TYPE_CHECKING:
    from ia_agent_fwk.memory.models import MemoryResult


class _MockEmbeddingProvider(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    def dimension(self) -> int:
        return 1536

    def max_tokens(self) -> int:
        return 8191


@pytest.mark.unit
class TestMemoryFactory:
    def test_create_in_memory_backend(self, sample_memory_settings: MemorySettings):
        backend = MemoryFactory.create(sample_memory_settings)
        assert isinstance(backend, InMemoryBackend)
        assert backend.backend_type == "in_memory"

    def test_create_conversation_backend(self):
        settings = MemorySettings(default_backend="conversation")
        backend = MemoryFactory.create(settings)
        assert isinstance(backend, ConversationMemoryBackend)
        assert backend.backend_type == "conversation"

    def test_create_unknown_backend(self):
        settings = MemorySettings(default_backend="nonexistent")
        with pytest.raises(MemoryConfigError, match="Unknown memory backend"):
            MemoryFactory.create(settings)

    def test_register_custom_backend(self, sample_memory_settings: MemorySettings):
        class CustomBackend(MemoryBackend):
            @property
            def backend_type(self) -> str:
                return "custom"

            async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
                pass

            async def retrieve(self, key: str) -> Any | None:
                return None

            async def search(self, query: str, top_k: int = 5) -> list[MemoryResult]:
                return []

            async def delete(self, key: str) -> bool:
                return False

            async def clear(self) -> None:
                pass

        try:
            MemoryFactory.register("custom", CustomBackend)
            settings = MemorySettings(default_backend="custom")
            backend = MemoryFactory.create(settings)
            assert isinstance(backend, CustomBackend)
            assert backend.backend_type == "custom"
        finally:
            # Clean up the registry to avoid test pollution
            MemoryFactory._registry.pop("custom", None)

    def test_register_duplicate_raises(self):
        with pytest.raises(MemoryConfigError, match="already registered"):
            MemoryFactory.register("in_memory", "some.module:SomeClass")

    def test_register_duplicate_with_replace(self):
        original = MemoryFactory._registry["in_memory"]
        try:
            MemoryFactory.register(
                "in_memory",
                "ia_agent_fwk.memory.backends.in_memory:InMemoryBackend",
                replace=True,
            )
            # Should succeed without error
            assert "in_memory" in MemoryFactory._registry
        finally:
            # Restore original
            MemoryFactory._registry["in_memory"] = original

    def test_factory_registry_has_all_backends(self):
        expected = {"in_memory", "conversation", "pgvector", "qdrant", "structured"}
        assert expected.issubset(set(MemoryFactory._registry.keys()))

    @patch("ia_agent_fwk.memory.embeddings.factory.EmbeddingFactory.create")
    def test_factory_create_pgvector(self, mock_embed_create):
        mock_embed_create.return_value = _MockEmbeddingProvider()
        settings = MemorySettings(
            default_backend="pgvector",
            embedding=EmbeddingSettings(provider="openai", api_key="test"),
        )
        settings.backends.pgvector.database_url = "postgresql://test:test@localhost:5432/test"
        backend = MemoryFactory.create(settings)
        assert isinstance(backend, PgVectorMemoryBackend)
        assert backend.backend_type == "pgvector"

    @patch("ia_agent_fwk.memory.embeddings.factory.EmbeddingFactory.create")
    def test_factory_create_qdrant(self, mock_embed_create):
        mock_embed_create.return_value = _MockEmbeddingProvider()
        settings = MemorySettings(
            default_backend="qdrant",
            embedding=EmbeddingSettings(provider="openai", api_key="test"),
        )
        backend = MemoryFactory.create(settings)
        assert isinstance(backend, QdrantMemoryBackend)
        assert backend.backend_type == "qdrant"

    def test_factory_create_structured(self):
        settings = MemorySettings(default_backend="structured")
        settings.backends.structured.database_url = "postgresql://test:test@localhost:5432/test"
        backend = MemoryFactory.create(settings)
        assert isinstance(backend, StructuredMemoryBackend)
        assert backend.backend_type == "structured"

    @patch("ia_agent_fwk.memory.embeddings.factory.EmbeddingFactory.create")
    def test_embedding_provider_creation(self, mock_embed_create):
        mock_provider = _MockEmbeddingProvider()
        mock_embed_create.return_value = mock_provider

        settings = EmbeddingSettings(provider="openai", api_key="test-key", model="text-embedding-3-small")
        result = MemoryFactory._create_embedding_provider(settings)
        assert result is mock_provider
        mock_embed_create.assert_called_once_with(settings)

    def test_embedding_provider_unknown(self):
        settings = EmbeddingSettings(provider="nonexistent")
        with pytest.raises(MemoryConfigError, match="Unknown embedding provider"):
            MemoryFactory._create_embedding_provider(settings)
