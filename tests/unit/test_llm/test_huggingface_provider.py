"""Tests for the HuggingFace provider (mocked transformers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ia_agent_fwk.llm.exceptions import LLMConfigError, LLMProviderError
from ia_agent_fwk.llm.models import FinishReason, Message


@pytest.mark.unit
class TestHuggingFaceProvider:
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock transformers pipeline."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = mock_tokenizer
        mock_pipe.return_value = [{"generated_text": "Hello, world!"}]

        return mock_pipe, mock_tokenizer

    @pytest.fixture
    def provider(self, mock_huggingface_provider_settings, mock_pipeline):
        """Create a HuggingFaceProvider with mocked pipeline."""
        mock_pipe, mock_tokenizer = mock_pipeline

        with (
            patch("ia_agent_fwk.llm.providers.huggingface._HAS_TRANSFORMERS", True),
            patch("ia_agent_fwk.llm.providers.huggingface.transformers") as mock_transformers,
        ):
            mock_transformers.pipeline.return_value = mock_pipe
            mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            from ia_agent_fwk.llm.providers.huggingface import HuggingFaceProvider

            provider = HuggingFaceProvider(
                settings=mock_huggingface_provider_settings,
                provider_name="huggingface",
            )
            # Pre-set the pipeline and tokenizer to avoid lazy loading issues.
            provider._pipeline = mock_pipe
            provider._tokenizer = mock_tokenizer
            return provider

    async def test_chat_success(self, provider):
        resp = await provider.chat([Message(role="user", content="Hi")])
        assert resp.message.role == "assistant"
        assert resp.message.content == "Hello, world!"
        assert resp.finish_reason == FinishReason.stop
        assert resp.usage.prompt_tokens >= 1
        assert resp.usage.completion_tokens >= 1

    async def test_complete_success(self, provider):
        resp = await provider.complete("Once upon a time")
        assert resp.text == "Hello, world!"
        assert resp.usage.total_tokens > 0
        assert resp.finish_reason == FinishReason.stop

    async def test_chat_with_custom_params(self, provider):
        resp = await provider.chat(
            [Message(role="user", content="Hi")],
            max_tokens=100,
            temperature=0.5,
        )
        assert resp.message.role == "assistant"
        assert resp.message.content == "Hello, world!"

    async def test_complete_with_custom_params(self, provider):
        resp = await provider.complete(
            "Once upon a time",
            max_tokens=100,
            temperature=0.5,
        )
        assert resp.text == "Hello, world!"

    async def test_chat_zero_temperature(self, provider):
        """When temperature is 0, do_sample should be False."""
        resp = await provider.chat(
            [Message(role="user", content="Hi")],
            temperature=0,
        )
        assert resp.message.content == "Hello, world!"

    async def test_stream_not_implemented(self, provider):
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            async for _ in provider.stream([Message(role="user", content="hi")]):
                pass

    def test_count_tokens(self, provider):
        count = provider.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count == 5  # Mock returns [1, 2, 3, 4, 5]

    async def test_health_check_success(self, provider):
        status = await provider.health_check()
        assert status.status == "healthy"
        assert status.latency_ms is not None

    async def test_health_check_failure(self, mock_huggingface_provider_settings):
        with (
            patch("ia_agent_fwk.llm.providers.huggingface._HAS_TRANSFORMERS", True),
            patch("ia_agent_fwk.llm.providers.huggingface.transformers") as mock_transformers,
        ):
            mock_transformers.pipeline.side_effect = RuntimeError("Model not found")
            mock_transformers.AutoTokenizer.from_pretrained.side_effect = RuntimeError("Model not found")

            from ia_agent_fwk.llm.providers.huggingface import HuggingFaceProvider

            provider = HuggingFaceProvider(
                settings=mock_huggingface_provider_settings,
                provider_name="huggingface",
            )

            status = await provider.health_check()
            assert status.status == "unhealthy"
            assert "Model not found" in (status.message or "")

    async def test_close(self, provider):
        await provider.close()
        assert provider._pipeline is None
        assert provider._tokenizer is None

    async def test_chat_pipeline_error(self, provider):
        provider._pipeline.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(LLMProviderError, match="HuggingFace chat error"):
            await provider.chat([Message(role="user", content="Hi")])

    async def test_complete_pipeline_error(self, provider):
        provider._pipeline.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(LLMProviderError, match="HuggingFace completion error"):
            await provider.complete("Once upon a time")

    def test_missing_transformers_dependency(self, mock_huggingface_provider_settings):
        with patch("ia_agent_fwk.llm.providers.huggingface._HAS_TRANSFORMERS", False):
            from ia_agent_fwk.llm.providers.huggingface import HuggingFaceProvider

            with pytest.raises(LLMConfigError, match="transformers"):
                HuggingFaceProvider(
                    settings=mock_huggingface_provider_settings,
                    provider_name="huggingface",
                )

    async def test_ensure_pipeline_lazy_init(self, mock_huggingface_provider_settings):
        """Pipeline is initialized lazily on first use."""
        mock_pipe = MagicMock()
        mock_pipe.tokenizer = MagicMock()
        mock_pipe.tokenizer.encode.return_value = [1, 2, 3]
        mock_pipe.return_value = [{"generated_text": "Lazy init!"}]

        with (
            patch("ia_agent_fwk.llm.providers.huggingface._HAS_TRANSFORMERS", True),
            patch("ia_agent_fwk.llm.providers.huggingface.transformers") as mock_transformers,
        ):
            mock_transformers.pipeline.return_value = mock_pipe

            from ia_agent_fwk.llm.providers.huggingface import HuggingFaceProvider

            provider = HuggingFaceProvider(
                settings=mock_huggingface_provider_settings,
                provider_name="huggingface",
            )
            assert provider._pipeline is None

            resp = await provider.chat([Message(role="user", content="Hi")])
            assert resp.message.content == "Lazy init!"
            assert provider._pipeline is not None
            mock_transformers.pipeline.assert_called_once()

    def test_ensure_tokenizer_lazy_init(self, mock_huggingface_provider_settings):
        """Tokenizer is initialized lazily on first use without loading the full model."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]

        with (
            patch("ia_agent_fwk.llm.providers.huggingface._HAS_TRANSFORMERS", True),
            patch("ia_agent_fwk.llm.providers.huggingface.transformers") as mock_transformers,
        ):
            mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            from ia_agent_fwk.llm.providers.huggingface import HuggingFaceProvider

            provider = HuggingFaceProvider(
                settings=mock_huggingface_provider_settings,
                provider_name="huggingface",
            )
            assert provider._tokenizer is None

            count = provider.count_tokens("Hello world")
            assert count == 4
            assert provider._tokenizer is not None
            mock_transformers.AutoTokenizer.from_pretrained.assert_called_once()

    def test_ensure_tokenizer_error(self, mock_huggingface_provider_settings):
        """Tokenizer loading error is wrapped in LLMProviderError."""
        with (
            patch("ia_agent_fwk.llm.providers.huggingface._HAS_TRANSFORMERS", True),
            patch("ia_agent_fwk.llm.providers.huggingface.transformers") as mock_transformers,
        ):
            mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("Network error")

            from ia_agent_fwk.llm.providers.huggingface import HuggingFaceProvider

            provider = HuggingFaceProvider(
                settings=mock_huggingface_provider_settings,
                provider_name="huggingface",
            )

            with pytest.raises(LLMProviderError, match="Failed to load tokenizer"):
                provider.count_tokens("hello")

    async def test_chat_multiple_messages(self, provider):
        """Chat with multiple messages in the conversation."""
        resp = await provider.chat(
            [
                Message(role="system", content="You are helpful."),
                Message(role="user", content="What is 2+2?"),
            ]
        )
        assert resp.message.role == "assistant"
        assert resp.usage.prompt_tokens >= 1
