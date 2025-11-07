"""Current time built-in tool.

Returns the current date and time in ISO 8601 format.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict

from ia_agent_fwk.tools.base import Tool, ToolContext


class CurrentTimeInput(BaseModel):
    """Input schema for the current time tool."""

    model_config = ConfigDict(frozen=True)

    timezone: str = "UTC"


class CurrentTimeOutput(BaseModel):
    """Output schema for the current time tool."""

    model_config = ConfigDict(frozen=True)

    current_time: str
    timezone: str


class CurrentTimeTool(Tool):
    """Return the current date and time."""

    @property
    def name(self) -> str:
        return "current_time"

    @property
    def description(self) -> str:
        return "Get the current date and time in ISO 8601 format."

    @property
    def input_schema(self) -> type[BaseModel]:
        return CurrentTimeInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return CurrentTimeOutput

    @property
    def tags(self) -> list[str]:
        return ["time", "utility", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Return the current time."""
        assert isinstance(validated_input, CurrentTimeInput)  # noqa: S101
        now = datetime.datetime.now(tz=datetime.UTC)
        return CurrentTimeOutput(
            current_time=now.isoformat(),
            timezone=validated_input.timezone,
        )
