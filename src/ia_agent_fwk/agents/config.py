"""Agent configuration and result Pydantic v2 models.

``AgentConfig`` defines all agent parameters with spec-compliant defaults.
``AgentResult`` captures execution output, state, token usage, iteration
count, duration, optional error, and metadata.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.agents.state import AgentState  # noqa: TC001
from ia_agent_fwk.llm.models import TokenUsage  # noqa: TC001


class AgentMemoryConfig(BaseModel):
    """Memory integration configuration for an agent."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    semantic_search_enabled: bool = True
    semantic_search_top_k: int = Field(default=5, ge=1)
    semantic_search_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    conversation_persistence_enabled: bool = True


class AgentConfig(BaseModel):
    """Pydantic v2 configuration model for an agent instance."""

    model_config = ConfigDict(frozen=True)

    name: str
    agent_type: str
    system_prompt: str = ""
    provider_name: str = "openai"
    model: str | None = None
    max_iterations: int = Field(default=10, ge=1)
    execution_timeout: int = Field(default=300, ge=1)
    max_tokens_per_response: int = Field(default=4096, ge=1)
    tools: list[str] = Field(default_factory=list)
    memory: AgentMemoryConfig = Field(default_factory=AgentMemoryConfig)
    context_window: int | None = Field(default=None, ge=1)


class AgentResult(BaseModel):
    """Pydantic v2 result model returned from ``Agent.run()``."""

    model_config = ConfigDict(frozen=True)

    output: str
    state: AgentState
    usage: TokenUsage
    iterations: int
    duration_ms: float
    error: str | None = None
    metadata: dict[str, Any] | None = None
