"""Default tool executor implementing the ToolExecutor Protocol.

``DefaultToolExecutor`` orchestrates the full tool execution pipeline:
parse arguments, look up tool, check permissions, validate input,
execute with timeout, validate output, and return a ``ToolResult``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ia_agent_fwk.agents.protocols import ToolResult
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
    ToolValidationError,
)

if TYPE_CHECKING:
    from ia_agent_fwk.llm.models import ToolCall
    from ia_agent_fwk.tools.permissions import ToolPermissionManager
    from ia_agent_fwk.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class DefaultToolExecutor:
    """Production tool executor implementing the ToolExecutor Protocol.

    Parameters
    ----------
    registry:
        The tool registry for looking up tools.
    permission_manager:
        The permission manager for access control.
    agent_id:
        The agent identifier (passed at construction time, one executor per agent).
    default_timeout:
        Default timeout in seconds for tool execution.

    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_manager: ToolPermissionManager,
        agent_id: str = "",
        default_timeout: float = 30.0,
    ) -> None:
        self.registry = registry
        self._permission_manager = permission_manager
        self._agent_id = agent_id
        self._default_timeout = default_timeout

    async def execute(self, tool_call: ToolCall) -> ToolResult:  # noqa: PLR0911
        """Execute a tool call through the full pipeline.

        The pipeline:
        1. Parse arguments from the ToolCall.
        2. Look up the tool by name in the registry.
        3. Check permissions for the agent.
        4. Validate input against the tool's input_schema.
        5. Execute the tool with asyncio.wait_for timeout.
        6. Validate output against the tool's output_schema.
        7. Return ToolResult with serialized output.

        All exceptions are caught and returned as ToolResult.error.

        Parameters
        ----------
        tool_call:
            The tool call from the LLM.

        Returns
        -------
        ToolResult
            The result of tool execution.

        """
        start_time = time.monotonic()
        tool_name = tool_call.name
        tool_call_id = tool_call.id

        try:
            return await self._execute_pipeline(tool_call, start_time)

        except ToolNotFoundError as exc:
            self._log_result(tool_name, start_time, error=str(exc))
            return ToolResult(output="", error=str(exc), tool_call_id=tool_call_id)

        except ToolPermissionError as exc:
            self._log_result(tool_name, start_time, error=str(exc))
            return ToolResult(output="", error=str(exc), tool_call_id=tool_call_id)

        except ToolValidationError as exc:
            error_msg = f"Validation error for tool '{tool_name}': {exc}"
            self._log_result(tool_name, start_time, error=error_msg)
            return ToolResult(output="", error=error_msg, tool_call_id=tool_call_id)

        except ToolTimeoutError as exc:
            self._log_result(tool_name, start_time, error=str(exc))
            return ToolResult(output="", error=str(exc), tool_call_id=tool_call_id)

        except ToolExecutionError as exc:
            self._log_result(tool_name, start_time, error=str(exc))
            return ToolResult(output="", error=str(exc), tool_call_id=tool_call_id)

        except Exception as exc:  # noqa: BLE001
            error_msg = f"Unexpected error executing tool '{tool_name}': {type(exc).__name__}: {exc}"
            self._log_result(tool_name, start_time, error=error_msg)
            return ToolResult(output="", error=error_msg, tool_call_id=tool_call_id)

    async def _execute_pipeline(self, tool_call: ToolCall, start_time: float) -> ToolResult:
        """Run the core execution pipeline (may raise)."""
        tool_name = tool_call.name
        tool_call_id = tool_call.id

        # Step 1: Parse arguments
        try:
            arguments = tool_call.parse_arguments()
        except Exception as exc:
            msg = f"Failed to parse arguments for tool '{tool_name}': {exc}"
            raise ToolValidationError(msg) from exc

        # Step 2: Look up tool
        tool = self.registry.get(tool_name)

        # Step 3: Check permissions
        self._permission_manager.check_permission(self._agent_id, tool_name)

        # Step 4: Validate input
        validated_input = self._validate_input(tool_name, tool.input_schema, arguments)

        # Step 5: Execute with timeout
        context = ToolContext(
            execution_id=uuid.uuid4().hex,
            agent_id=self._agent_id,
            timeout=self._default_timeout,
        )

        try:
            result = await asyncio.wait_for(
                tool.execute(validated_input, context),
                timeout=self._default_timeout,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:  # noqa: UP041
            msg = f"Tool '{tool_name}' timed out after {self._default_timeout}s"
            raise ToolTimeoutError(msg, timeout=self._default_timeout) from exc
        except (ToolExecutionError, ToolValidationError, ToolPermissionError):
            raise
        except Exception as exc:
            msg = f"Tool '{tool_name}' execution failed: {type(exc).__name__}: {exc}"
            raise ToolExecutionError(msg, tool_name=tool_name) from exc

        # Step 6: Validate output
        validated_output = self._validate_output(tool_name, tool.output_schema, result)

        # Step 7: Return ToolResult
        output_json = validated_output.model_dump_json()
        self._log_result(tool_name, start_time)
        return ToolResult(output=output_json, tool_call_id=tool_call_id)

    @staticmethod
    def _validate_input(
        tool_name: str,
        schema: type[BaseModel],
        arguments: dict[str, Any],
    ) -> BaseModel:
        """Validate input arguments against the tool's input schema."""
        try:
            return schema.model_validate(arguments)
        except ValidationError as exc:
            details = [
                {
                    "field": ".".join(str(loc) for loc in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"],
                }
                for err in exc.errors()
            ]
            msg = f"Input validation failed for tool '{tool_name}': {exc}"
            raise ToolValidationError(msg, details=details) from exc

    @staticmethod
    def _validate_output(
        tool_name: str,
        schema: type[BaseModel],
        result: BaseModel,
    ) -> BaseModel:
        """Validate output against the tool's output schema."""
        if isinstance(result, schema):
            return result

        # If the result is not the expected type, try to validate as dict
        try:
            data = result.model_dump() if isinstance(result, BaseModel) else result
            return schema.model_validate(data)
        except ValidationError as exc:
            details = [
                {
                    "field": ".".join(str(loc) for loc in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"],
                }
                for err in exc.errors()
            ]
            msg = f"Output validation failed for tool '{tool_name}': {exc}"
            raise ToolValidationError(msg, details=details) from exc
        except Exception as exc:
            msg = f"Output validation failed for tool '{tool_name}': {exc}"
            raise ToolValidationError(msg) from exc

    @staticmethod
    def _log_result(tool_name: str, start_time: float, error: str | None = None) -> None:
        """Log the result of a tool execution and record metrics."""
        elapsed_ms = (time.monotonic() - start_time) * 1000
        elapsed_s = elapsed_ms / 1000

        collector = get_metrics_collector()
        status = "error" if error else "success"

        collector.increment(
            "tool_executions_total",
            labels={"tool": tool_name, "status": status},
        )
        collector.observe("tool_execution_duration_seconds", elapsed_s, labels={"tool": tool_name})

        # Structured log data for Loki/JSON searchability
        log_data: dict[str, Any] = {
            "event": "tool_execution",
            "tool": tool_name,
            "status": status,
            "duration_ms": round(elapsed_ms, 1),
        }

        if error:
            # Extract error type from the error message
            error_type = "unknown"
            for etype in ("NotFound", "Permission", "Validation", "Timeout", "Execution"):
                if etype.lower() in error.lower():
                    error_type = etype.lower()
                    break
            collector.increment(
                "tool_errors_total",
                labels={"tool": tool_name, "error_type": error_type},
            )
            log_data["error_type"] = error_type
            log_data["error"] = error
            logger.warning(
                "Tool '%s' executed in %.1fms (error: %s)",
                tool_name,
                elapsed_ms,
                error,
                extra={"tool_data": log_data},
            )
        else:
            logger.info(
                "Tool '%s' executed in %.1fms (success)",
                tool_name,
                elapsed_ms,
                extra={"tool_data": log_data},
            )
