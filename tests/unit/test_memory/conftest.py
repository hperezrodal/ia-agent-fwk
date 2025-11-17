"""Shared fixtures for memory unit tests."""

from __future__ import annotations

import pytest

from ia_agent_fwk.config.settings import MemorySettings
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend
from ia_agent_fwk.memory.backends.in_memory import InMemoryBackend


@pytest.fixture
def sample_memory_settings() -> MemorySettings:
    """Default test memory settings."""
    return MemorySettings()


@pytest.fixture
def in_memory_backend() -> InMemoryBackend:
    """Fresh InMemoryBackend with max_items=10 for easy eviction testing."""
    return InMemoryBackend(max_items=10)


@pytest.fixture
def conversation_backend() -> ConversationMemoryBackend:
    """Fresh ConversationMemoryBackend with max_history=5."""
    return ConversationMemoryBackend(max_history=5)
