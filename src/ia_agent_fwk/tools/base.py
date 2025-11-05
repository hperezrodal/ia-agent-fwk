"""Tool abstract base class and ToolContext dataclass.

The ``Tool`` ABC defines the interface all tools must implement.
``ToolContext`` provides execution context passed to tools at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


class Tool(ABC):
    """Abstract base class for all tools.

    Subclasses must implement ``name``, ``description``, ``input_schema``,
    ``output_schema``, and ``execute()``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the tool."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Return the Pydantic model class for tool input validation."""
        ...

    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Return the Pydantic model class for tool output validation."""
        ...

    @property
    def tags(self) -> list[str]:
        """Return tags for tool categorization (default: empty list)."""
        return []

    @abstractmethod
    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        """Execute the tool with validated input.

        Parameters
        ----------
        validated_input:
            The validated input model instance.
        context:
            Execution context with agent info and metadata.

        Returns
        -------
        BaseModel
            The output model instance.

        """
        ...


@dataclass
class ToolContext:
    """Execution context passed to tools at runtime.

    Attributes
    ----------
    execution_id:
        Unique identifier for this execution.
    agent_id:
        Identifier of the agent executing the tool.
    timeout:
        Maximum execution time in seconds.
    metadata:
        Additional context data.

    """

    execution_id: str
    agent_id: str = ""
    timeout: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)
