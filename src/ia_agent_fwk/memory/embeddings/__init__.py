"""Embedding provider interface and implementations.

Public API
----------
.. autoclass:: EmbeddingProvider
.. autoclass:: OpenAIEmbeddingProvider
.. autoclass:: EmbeddingFactory
"""

from __future__ import annotations

from ia_agent_fwk.memory.embeddings.base import EmbeddingProvider
from ia_agent_fwk.memory.embeddings.factory import EmbeddingFactory
from ia_agent_fwk.memory.embeddings.ollama import OllamaEmbeddingProvider
from ia_agent_fwk.memory.embeddings.openai import OpenAIEmbeddingProvider

__all__ = [
    "EmbeddingFactory",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
]
