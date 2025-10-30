"""Tests for AgentFactory."""

from __future__ import annotations

import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.exceptions import AgentConfigError
from ia_agent_fwk.agents.factory import AgentFactory
from ia_agent_fwk.agents.registry import AgentRegistry
from ia_agent_fwk.config.settings import LLMProviderSettings, LLMSettings


class _FactoryTestAgent(Agent):
    """Concrete agent for factory tests."""

    @property
    def agent_type(self) -> str:
        return "factory_test"


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state between tests."""
    saved = dict(AgentRegistry._registry)
    AgentRegistry._registry.clear()
    yield
    AgentRegistry._registry.clear()
    AgentRegistry._registry.update(saved)


@pytest.fixture
def llm_settings() -> LLMSettings:
    # Use ollama which doesn't require an API key
    return LLMSettings(
        default_provider="ollama",
        providers={
            "ollama": LLMProviderSettings(
                base_url="http://localhost:11434",
                default_model="llama3.1",
            ),
        },
    )


class TestFactoryCreate:
    def test_create_with_valid_config(self, llm_settings):
        """AC-08: create with valid config returns correct agent instance."""
        AgentRegistry.register("factory_test", _FactoryTestAgent)
        config = AgentConfig(name="test", agent_type="factory_test", provider_name="ollama")
        agent = AgentFactory.create(config, llm_settings)
        assert isinstance(agent, _FactoryTestAgent)
        assert isinstance(agent, Agent)

    def test_unknown_agent_type_raises(self, llm_settings):
        """AC-09: unknown agent_type raises AgentConfigError."""
        config = AgentConfig(name="test", agent_type="nonexistent", provider_name="ollama")
        with pytest.raises(AgentConfigError) as exc_info:
            AgentFactory.create(config, llm_settings)
        assert "nonexistent" in str(exc_info.value)

    def test_invalid_provider_raises(self):
        """Invalid provider_name raises AgentConfigError."""
        AgentRegistry.register("factory_test", _FactoryTestAgent)
        config = AgentConfig(
            name="test",
            agent_type="factory_test",
            provider_name="nonexistent_provider",
        )
        llm_settings = LLMSettings(
            default_provider="openai",
            providers={},
        )
        with pytest.raises(AgentConfigError) as exc_info:
            AgentFactory.create(config, llm_settings)
        assert "nonexistent_provider" in str(exc_info.value)
