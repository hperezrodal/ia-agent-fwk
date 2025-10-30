"""Tests for AgentRegistry."""

from __future__ import annotations

import pytest

from ia_agent_fwk.agents.base import Agent
from ia_agent_fwk.agents.exceptions import AgentConfigError
from ia_agent_fwk.agents.registry import AgentRegistry


class _TestAgent(Agent):
    """Minimal concrete agent for registry tests."""

    @property
    def agent_type(self) -> str:
        return "test"


class _AnotherTestAgent(Agent):
    """Another concrete agent for registry tests."""

    @property
    def agent_type(self) -> str:
        return "another"


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state between tests."""
    saved = dict(AgentRegistry._registry)
    AgentRegistry._registry.clear()
    yield
    AgentRegistry._registry.clear()
    AgentRegistry._registry.update(saved)


class TestRegister:
    def test_register_and_get(self):
        """AC-10: register and get succeeds."""
        AgentRegistry.register("support", _TestAgent)
        result = AgentRegistry.get("support")
        assert result is _TestAgent

    def test_duplicate_raises(self):
        AgentRegistry.register("test", _TestAgent)
        with pytest.raises(AgentConfigError):
            AgentRegistry.register("test", _AnotherTestAgent)

    def test_replace_true_allows_overwrite(self):
        AgentRegistry.register("test", _TestAgent)
        AgentRegistry.register("test", _AnotherTestAgent, replace=True)
        assert AgentRegistry.get("test") is _AnotherTestAgent


class TestGet:
    def test_unknown_name_raises(self):
        AgentRegistry.register("known", _TestAgent)
        with pytest.raises(AgentConfigError) as exc_info:
            AgentRegistry.get("unknown")
        assert "unknown" in str(exc_info.value)
        assert "known" in str(exc_info.value)

    def test_empty_registry_error_message(self):
        with pytest.raises(AgentConfigError) as exc_info:
            AgentRegistry.get("anything")
        assert "(none)" in str(exc_info.value)


class TestList:
    def test_list_returns_registered_names(self):
        """AC-11: list returns all registered names."""
        AgentRegistry.register("support", _TestAgent)
        AgentRegistry.register("research", _AnotherTestAgent)
        names = AgentRegistry.list()
        assert "support" in names
        assert "research" in names

    def test_list_sorted_alphabetically(self):
        AgentRegistry.register("zebra", _TestAgent)
        AgentRegistry.register("alpha", _AnotherTestAgent)
        names = AgentRegistry.list()
        assert names == ["alpha", "zebra"]

    def test_list_empty(self):
        assert AgentRegistry.list() == []
