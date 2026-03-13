"""Shared fixtures for RAG evaluation integration tests."""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

OLLAMA_URL = os.environ.get("RAG_EVAL_OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("RAG_EVAL_QDRANT_URL", "http://127.0.0.1:6333")


def _qdrant_available() -> bool:
    """Check if Qdrant is reachable."""
    try:
        resp = httpx.get(f"{QDRANT_URL}/healthz", timeout=5)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    else:
        return resp.status_code == 200


def _ollama_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    else:
        return resp.status_code == 200


skip_no_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason=f"Qdrant not available on {QDRANT_URL}",
)

skip_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason=f"Ollama not available on {OLLAMA_URL}",
)

skip_no_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture
def qdrant_collection_name() -> str:
    """Generate a unique Qdrant collection name for test isolation."""
    return f"rag_eval_test_{uuid.uuid4().hex[:8]}"
