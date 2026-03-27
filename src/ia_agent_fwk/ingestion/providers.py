"""Embedding provider factory — shared between ingest and query scripts.

Builds the right embedding provider based on config + provider name.
"""

from __future__ import annotations

import logging

from ia_agent_fwk.config import load_config
from ia_agent_fwk.ingestion.embedding_store import EmbeddingProvider, EmbeddingStore, StoreConfig
from ia_agent_fwk.memory.embeddings.ollama import OllamaEmbeddingProvider
from ia_agent_fwk.memory.embeddings.openai import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)

# Model → dimensions mapping
OLLAMA_MODELS = {
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "mxbai-embed-large": 1024,
    "multilingual-e5-large": 1024,
    "snowflake-arctic-embed": 1024,
    "all-minilm": 384,
}

EMBEDDING_DIMENSIONS = {
    "openai": 1536,
    "ollama": 768,  # default, overridden by model
}


def build_embedding_provider(
    provider: str,
    settings: object | None = None,
    model: str | None = None,
) -> OllamaEmbeddingProvider | OpenAIEmbeddingProvider:
    """Build embedding provider from config.

    Parameters
    ----------
    model:
        Override the model name. For Ollama: "bge-m3", "nomic-embed-text", etc.

    """
    if settings is None:
        settings = load_config()
    mem_cfg = settings.memory
    if provider == "ollama":
        ollama_model = model or "nomic-embed-text"
        return OllamaEmbeddingProvider(
            base_url=mem_cfg.embedding.base_url or "http://localhost:11434",
            model=ollama_model,
        )
    return OpenAIEmbeddingProvider(
        api_key=mem_cfg.embedding.api_key,
        model=model or mem_cfg.embedding.model,
    )


def get_embedding_dim(provider: str, model: str | None = None) -> int:
    """Get embedding dimensions for a provider+model combo."""
    if provider == "openai":
        return 1536
    if model and model in OLLAMA_MODELS:
        return OLLAMA_MODELS[model]
    return 768  # default for unknown Ollama models


def detect_collection_model(
    provider: str,  # noqa: ARG001
    settings: object | None = None,
) -> str | None:
    """Auto-detect the embedding model from an existing Qdrant collection.

    Reads the model name stored as metadata point (id=0) in the collection.
    Returns None if the collection doesn't exist or has no stored model.
    """
    if settings is None:
        settings = load_config()
    qdrant_cfg = settings.memory.backends.qdrant
    url = qdrant_cfg.url or "http://localhost:6333"

    try:
        from ia_agent_fwk.ingestion.hybrid_store import HybridStore  # noqa: PLC0415

        store = HybridStore(url=url, collection_name=qdrant_cfg.collection_name)
        model_name = store.get_stored_model()
        if model_name:
            logger.info("Auto-detected embedding model from collection: %s", model_name)
            return model_name
    except Exception:  # noqa: BLE001
        logger.debug("Failed to detect model from collection")
    return None


def build_embedding_store(
    provider: str,
    settings: object | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    model: str | None = None,
) -> EmbeddingStore:
    """Build EmbeddingStore from config.

    If *model* is not provided and a collection already exists, auto-detects
    the model from the collection's vector dimensions. This prevents the
    dimension mismatch error (e.g. ingested with bge-m3 1024d, queried
    with nomic-embed-text 768d).
    """
    if settings is None:
        settings = load_config()

    # Auto-detect model if not specified and collection exists
    if model is None and provider == "ollama":
        model = detect_collection_model(provider, settings)

    if embedding_provider is None:
        embedding_provider = build_embedding_provider(provider, settings, model=model)

    qdrant_cfg = settings.memory.backends.qdrant
    embed_dim = get_embedding_dim(provider, model)

    return EmbeddingStore(
        embedding_provider=embedding_provider,
        config=StoreConfig(
            qdrant_url=qdrant_cfg.url or "http://localhost:6333",
            collection_name=qdrant_cfg.collection_name,
            dense_dim=embed_dim,
            embedding_model=model or "",
        ),
    )
