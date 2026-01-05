"""Tests for TriggerManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ia_agent_fwk.execution.models import EventTrigger
from ia_agent_fwk.execution.triggers import TriggerManager


@pytest.fixture
def mock_job_manager():
    manager = MagicMock()
    manager.submit.return_value = "job-abc-123"
    return manager


@pytest.fixture
def trigger_manager(mock_job_manager):
    return TriggerManager(job_manager=mock_job_manager)


@pytest.fixture
def sample_trigger():
    return EventTrigger(
        name="deploy-check",
        agent_type="monitor",
        prompt_template="Check deployment status for {service}",
        event_type="deploy",
    )


@pytest.mark.unit
class TestRegisterTrigger:
    def test_register_trigger(self, trigger_manager, sample_trigger):
        trigger_id = trigger_manager.register_trigger(sample_trigger)

        assert trigger_id is not None
        assert len(trigger_id) > 0

        result = trigger_manager.get_trigger(trigger_id)
        assert result is not None
        assert result.name == "deploy-check"
        assert result.event_type == "deploy"


@pytest.mark.unit
class TestUnregisterTrigger:
    def test_unregister_trigger(self, trigger_manager, sample_trigger):
        trigger_id = trigger_manager.register_trigger(sample_trigger)
        assert trigger_manager.unregister_trigger(trigger_id) is True
        assert trigger_manager.get_trigger(trigger_id) is None

    def test_unregister_nonexistent(self, trigger_manager):
        assert trigger_manager.unregister_trigger("nonexistent") is False


@pytest.mark.unit
class TestListTriggers:
    def test_list_triggers_empty(self, trigger_manager):
        assert trigger_manager.list_triggers() == []

    def test_list_triggers(self, trigger_manager, sample_trigger):
        tid1 = trigger_manager.register_trigger(sample_trigger)
        other = EventTrigger(
            name="alert",
            agent_type="alerter",
            prompt_template="Alert: {message}",
            event_type="alert",
        )
        tid2 = trigger_manager.register_trigger(other)

        triggers = trigger_manager.list_triggers()
        assert len(triggers) == 2
        ids = {t[0] for t in triggers}
        assert tid1 in ids
        assert tid2 in ids


@pytest.mark.unit
class TestFireTrigger:
    def test_fire_trigger_matches(self, trigger_manager, sample_trigger, mock_job_manager):
        trigger_id = trigger_manager.register_trigger(sample_trigger)

        result = trigger_manager.fire_trigger("deploy", {"service": "api-server"})

        assert result is not None
        fired_tid, job_id = result
        assert fired_tid == trigger_id
        assert job_id == "job-abc-123"

        mock_job_manager.submit.assert_called_once_with(
            agent_type="monitor",
            prompt="Check deployment status for api-server",
            config_overrides=None,
        )

    def test_fire_trigger_no_match(self, trigger_manager, sample_trigger):
        trigger_manager.register_trigger(sample_trigger)

        result = trigger_manager.fire_trigger("unknown-event", {"key": "value"})
        assert result is None

    def test_fire_trigger_submits_job(self, trigger_manager, mock_job_manager):
        trigger = EventTrigger(
            name="simple",
            agent_type="worker",
            prompt_template="Process event data",
            event_type="process",
            config_overrides={"timeout": 120},
        )
        trigger_manager.register_trigger(trigger)

        result = trigger_manager.fire_trigger("process", {})
        assert result is not None
        _, job_id = result
        assert job_id == "job-abc-123"

        mock_job_manager.submit.assert_called_once_with(
            agent_type="worker",
            prompt="Process event data",
            config_overrides={"timeout": 120},
        )

    def test_fire_trigger_template_fallback(self, trigger_manager, mock_job_manager):
        """When template has a missing key, fallback to appending JSON payload."""
        trigger = EventTrigger(
            name="templated",
            agent_type="worker",
            prompt_template="Process: {missing_key}",
            event_type="evt",
        )
        trigger_manager.register_trigger(trigger)

        result = trigger_manager.fire_trigger("evt", {"actual_key": "value"})
        assert result is not None
        # The prompt should contain the fallback JSON payload
        call_args = mock_job_manager.submit.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "actual_key" in prompt
