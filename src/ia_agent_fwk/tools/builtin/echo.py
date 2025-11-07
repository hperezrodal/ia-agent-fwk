"""Echo built-in tool.

Returns the input message unchanged. Useful for testing the tool pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ia_agent_fwk.tools.base import Tool, ToolContext


class EchoInput(BaseModel):
    """Input schema for the echo tool."""

    model_config = ConfigDict(frozen=True)

    message: str


class EchoOutput(BaseModel):
    """Output schema for the echo tool."""

    model_config = ConfigDict(frozen=True)

    message: str


class EchoTool(Tool):
    """Echo tool that returns the input message unchanged.

    Useful for testing the tool execution pipeline.
    """

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo back the input message. Useful for testing."

    @property
    def input_schema(self) -> type[BaseModel]:
        return EchoInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return EchoOutput

    @property
    def tags(self) -> list[str]:
        return ["utility", "builtin", "testing"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Return the input message."""
        assert isinstance(validated_input, EchoInput)  # noqa: S101
        return EchoOutput(message=validated_input.message)
