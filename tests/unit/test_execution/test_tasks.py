"""Tests for execute_agent_task Celery task."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ia_agent_fwk.agents.config import AgentResult
from ia_agent_fwk.agents.state import AgentState
from ia_agent_fwk.llm.models import TokenUsage


@pytest.mark.unit
class TestRunAgent:
    """Test the _run_agent async helper function."""

    async def test_run_agent_success(self):
        from ia_agent_fwk.execution.tasks import _run_agent

        mock_settings = MagicMock()
        mock_settings.llm.default_provider = "openai"

        mock_result = AgentResult(
            output="Hello!",
            state=AgentState.COMPLETED,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            iterations=1,
            duration_ms=42.0,
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_agent

        mock_registry = MagicMock()

        with (
            patch("ia_agent_fwk.config.loader.load_config", return_value=mock_settings),
            patch("ia_agent_fwk.agents.factory.AgentFactory.create", mock_factory.create),
            patch("ia_agent_fwk.agents.registry.AgentRegistry.get", mock_registry.get),
        ):
            result = await _run_agent("test", "Hello", None, None)

        assert result["output"] == "Hello!"
        assert result["state"] == "COMPLETED"
        assert result["iterations"] == 1
        mock_registry.get.assert_called_once_with("test")
        mock_factory.create.assert_called_once()

    async def test_run_agent_propagates_error(self):
        from ia_agent_fwk.execution.tasks import _run_agent

        mock_settings = MagicMock()
        mock_settings.llm.default_provider = "openai"

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM failed"))

        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_agent

        with (
            patch("ia_agent_fwk.config.loader.load_config", return_value=mock_settings),
            patch("ia_agent_fwk.agents.factory.AgentFactory.create", mock_factory.create),
            patch("ia_agent_fwk.agents.registry.AgentRegistry.get"),
            pytest.raises(RuntimeError, match="LLM failed"),
        ):
            await _run_agent("test", "Hello", None, None)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestExecuteAgentTask:
    """Test the execute_agent_task Celery task function.

    We patch ``update_state`` on the real task object and ``asyncio`` at
    the tasks module level.  Using ``__wrapped__`` with ``self`` being
    the real Celery Task instance is fine as long as ``update_state`` is
    mocked to prevent backend access.
    """

    def test_task_returns_result_on_success(self):
        from ia_agent_fwk.execution.tasks import execute_agent_task

        mock_result_dict = {
            "output": "Hello!",
            "state": "COMPLETED",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "iterations": 1,
            "duration_ms": 42.0,
            "error": None,
            "metadata": None,
        }

        with (
            patch.object(execute_agent_task, "update_state"),
            patch("ia_agent_fwk.execution.tasks.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = mock_result_dict
            # Provide a fake request id
            execute_agent_task.request.id = "job-123"

            result = execute_agent_task("test", "Hello")

        mock_asyncio.run.assert_called_once()
        assert result["output"] == "Hello!"
        assert result["state"] == "COMPLETED"

    def test_task_returns_error_dict_on_exception(self):
        from ia_agent_fwk.execution.tasks import execute_agent_task

        with (
            patch.object(execute_agent_task, "update_state"),
            patch("ia_agent_fwk.execution.tasks.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.side_effect = RuntimeError("Agent failed")
            execute_agent_task.request.id = "job-456"

            result = execute_agent_task("test", "Hello")

        assert result["state"] == "FAILED"
        assert result["error"] == "Agent failed"
        assert result["output"] == ""
        assert result["iterations"] == 0

    def test_task_calls_update_state(self):
        from ia_agent_fwk.execution.tasks import execute_agent_task

        with (
            patch.object(execute_agent_task, "update_state") as mock_update,
            patch("ia_agent_fwk.execution.tasks.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = {
                "output": "done",
                "state": "COMPLETED",
                "usage": {},
                "iterations": 1,
                "duration_ms": 10.0,
            }
            execute_agent_task.request.id = "job-789"

            execute_agent_task("conversational", "Hi")

        mock_update.assert_called_once_with(
            state="RUNNING",
            meta={"agent_type": "conversational"},
        )
