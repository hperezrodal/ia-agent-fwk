"""Tests for embedding providers and factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.embeddings.factory import EmbeddingFactory
from ia_agent_fwk.memory.embeddings.openai import OpenAIEmbeddingProvider
from ia_agent_fwk.memory.exceptions import MemoryConfigError, MemoryStoreError


@pytest.mark.unit
class TestEmbeddingProviderABC:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError, match="abstract"):
            EmbeddingProvider()  # type: ignore[abstract]


@pytest.mark.unit
class TestOpenAIEmbeddingProvider:
    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    def test_dimension_small_model(self, mock_openai_mod):
        mock_openai_mod.AsyncOpenAI.return_value = MagicMock()
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-3-small")
        assert provider.dimension() == 1536

    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    def test_dimension_large_model(self, mock_openai_mod):
        mock_openai_mod.AsyncOpenAI.return_value = MagicMock()
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-3-large")
        assert provider.dimension() == 3072

    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    def test_max_tokens(self, mock_openai_mod):
        mock_openai_mod.AsyncOpenAI.return_value = MagicMock()
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-3-small")
        assert provider.max_tokens() == 8191

    def test_unknown_model_raises(self):
        with pytest.raises(MemoryConfigError, match="Unknown OpenAI embedding model"):
            OpenAIEmbeddingProvider(api_key="test-key", model="nonexistent-model")

    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    async def test_embed_single(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        embedding_data = MagicMock()
        embedding_data.index = 0
        embedding_data.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [embedding_data]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        result = await provider.embed(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        mock_client.embeddings.create.assert_awaited_once_with(
            input=["hello world"],
            model="text-embedding-3-small",
        )

    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    async def test_embed_batch(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        data_items = []
        for i in range(3):
            item = MagicMock()
            item.index = i
            item.embedding = [float(i)] * 1536
            data_items.append(item)

        mock_response = MagicMock()
        mock_response.data = data_items
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        result = await provider.embed(["text1", "text2", "text3"])

        assert len(result) == 3
        for vec in result:
            assert len(vec) == 1536

    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    async def test_embed_empty_texts(self, mock_openai_mod):
        mock_openai_mod.AsyncOpenAI.return_value = AsyncMock()
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        result = await provider.embed([])
        assert result == []

    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    async def test_embed_api_error(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client
        mock_client.embeddings.create = AsyncMock(side_effect=RuntimeError("API error"))

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        with pytest.raises(MemoryStoreError, match="OpenAI embedding generation failed"):
            await provider.embed(["hello"])

    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    async def test_close(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        await provider.close()
        mock_client.close.assert_awaited_once()


@pytest.mark.unit
class TestEmbeddingFactory:
    def test_registry_has_openai(self):
        assert "openai" in EmbeddingFactory._registry

    @patch("ia_agent_fwk.memory.embeddings.openai.openai")
    def test_create_openai(self, mock_openai_mod):
        from ia_agent_fwk.config.settings import EmbeddingSettings

        mock_openai_mod.AsyncOpenAI.return_value = MagicMock()
        settings = EmbeddingSettings(provider="openai", api_key="test-key", model="text-embedding-3-small")
        provider = EmbeddingFactory.create(settings)
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_create_unknown_provider(self):
        from ia_agent_fwk.config.settings import EmbeddingSettings

        settings = EmbeddingSettings(provider="nonexistent")
        with pytest.raises(MemoryConfigError, match="Unknown embedding provider"):
            EmbeddingFactory.create(settings)

    def test_register_duplicate_raises(self):
        with pytest.raises(MemoryConfigError, match="already registered"):
            EmbeddingFactory.register("openai", "some.module:SomeClass")

    def test_register_custom_provider(self):
        class CustomProvider(EmbeddingProvider):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 10 for _ in texts]

            def dimension(self) -> int:
                return 10

            def max_tokens(self) -> int:
                return 512

        try:
            EmbeddingFactory.register("custom_test", CustomProvider)
            assert "custom_test" in EmbeddingFactory._registry
        finally:
            EmbeddingFactory._registry.pop("custom_test", None)
