"""Tests for AgentConfig and AgentResult models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ia_agent_fwk.agents.config import AgentConfig, AgentResult
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.models import TokenUsage


class TestAgentConfig:
    def test_defaults_match_spec(self):
        config = AgentConfig(name="test", agent_type="base")
        assert config.max_iterations == 10
        assert config.execution_timeout == 300
        assert config.max_tokens_per_response == 4096

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            AgentConfig()  # type: ignore[call-arg]

    def test_name_and_type_required(self):
        with pytest.raises(ValidationError):
            AgentConfig(name="test")  # type: ignore[call-arg]

    def test_all_defaults(self):
        config = AgentConfig(name="test", agent_type="base")
        assert config.system_prompt == ""
        assert config.provider_name == "openai"
        assert config.model is None
        assert config.tools == []
        assert config.memory.enabled is True
        assert config.context_window is None

    def test_frozen(self):
        config = AgentConfig(name="test", agent_type="base")
        with pytest.raises(ValidationError):
            config.name = "changed"  # type: ignore[misc]

    def test_negative_max_iterations_rejected(self):
        with pytest.raises(ValidationError):
            AgentConfig(name="test", agent_type="base", max_iterations=0)

    def test_negative_execution_timeout_rejected(self):
        with pytest.raises(ValidationError):
            AgentConfig(name="test", agent_type="base", execution_timeout=0)

    def test_custom_values(self):
        config = AgentConfig(
            name="my-agent",
            agent_type="custom",
            system_prompt="Be helpful.",
            provider_name="anthropic",
            model="claude-3-haiku",
            max_iterations=5,
            execution_timeout=60,
            max_tokens_per_response=2048,
            tools=["search", "calculate"],
            memory={"enabled": True, "semantic_search_top_k": 10},
            context_window=16384,
        )
        assert config.name == "my-agent"
        assert config.agent_type == "custom"
        assert config.tools == ["search", "calculate"]
        assert config.context_window == 16384


class TestAgentResult:
    def test_basic_creation(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
        result = AgentResult(
            output="Hello!",
            state=AgentState.COMPLETED,
            usage=usage,
            iterations=1,
            duration_ms=100.5,
        )
        assert result.output == "Hello!"
        assert result.state == AgentState.COMPLETED
        assert result.usage.total_tokens == 30
        assert result.iterations == 1
        assert result.duration_ms == 100.5
        assert result.error is None
        assert result.metadata is None

    def test_with_error(self):
        usage = TokenUsage()
        result = AgentResult(
            output="",
            state=AgentState.FAILED,
            usage=usage,
            iterations=0,
            duration_ms=50.0,
            error="Timeout exceeded",
        )
        assert result.error == "Timeout exceeded"

    def test_with_metadata(self):
        usage = TokenUsage()
        result = AgentResult(
            output="done",
            state=AgentState.COMPLETED,
            usage=usage,
            iterations=2,
            duration_ms=200.0,
            metadata={"agent_id": "123"},
        )
        assert result.metadata == {"agent_id": "123"}

    def test_frozen(self):
        usage = TokenUsage()
        result = AgentResult(
            output="done",
            state=AgentState.COMPLETED,
            usage=usage,
            iterations=1,
            duration_ms=100.0,
        )
        with pytest.raises(ValidationError):
            result.output = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        result = AgentResult(
            output="response text",
            state=AgentState.COMPLETED,
            usage=usage,
            iterations=3,
            duration_ms=1500.0,
            metadata={"key": "value"},
        )
        data = result.model_dump()
        restored = AgentResult(**data)
        assert restored.output == result.output
        assert restored.state == result.state
        assert restored.usage.total_tokens == result.usage.total_tokens
        assert restored.iterations == result.iterations
        assert restored.duration_ms == result.duration_ms
        assert restored.metadata == result.metadata

    def test_uses_token_usage_from_llm_models(self):
        usage = TokenUsage(prompt_tokens=5, completion_tokens=10)
        result = AgentResult(
            output="test",
            state=AgentState.COMPLETED,
            usage=usage,
            iterations=1,
            duration_ms=10.0,
        )
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.total_tokens == 15


class TestAgentConfigSettings:
    """F-007: AgentConfigSettings validation matches AgentConfig requirements."""

    def test_both_empty_allowed(self):
        """Empty defaults are valid (for default config)."""
        from ia_agent_fwk.config.settings import AgentConfigSettings

        s = AgentConfigSettings()
        assert s.name == ""
        assert s.agent_type == ""

    def test_both_set_allowed(self):
        from ia_agent_fwk.config.settings import AgentConfigSettings

        s = AgentConfigSettings(name="bot", agent_type="support")
        assert s.name == "bot"
        assert s.agent_type == "support"

    def test_name_without_type_rejected(self):
        from ia_agent_fwk.config.settings import AgentConfigSettings

        with pytest.raises(ValidationError, match=r"name.*agent_type"):
            AgentConfigSettings(name="bot")

    def test_type_without_name_rejected(self):
        from ia_agent_fwk.config.settings import AgentConfigSettings

        with pytest.raises(ValidationError, match=r"name.*agent_type"):
            AgentConfigSettings(agent_type="support")
