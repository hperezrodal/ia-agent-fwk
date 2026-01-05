"""Tests for ScheduleManager and cron validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ia_agent_fwk.execution.exceptions import (
    InvalidCronExpressionError,
    ScheduleNotFoundError,
)
from ia_agent_fwk.execution.models import ScheduleDefinition
from ia_agent_fwk.execution.scheduler import ScheduleManager, validate_cron_expression


@pytest.fixture
def schedule_manager():
    return ScheduleManager()


@pytest.fixture
def sample_definition():
    return ScheduleDefinition(
        name="daily-report",
        agent_type="report",
        prompt="Generate daily report",
        cron_expression="0 9 * * *",
    )


@pytest.mark.unit
class TestCronValidation:
    def test_valid_cron_expression(self):
        validate_cron_expression("0 9 * * *")
        validate_cron_expression("*/5 * * * *")
        validate_cron_expression("0 0 1 1 *")
        validate_cron_expression("0,15,30,45 * * * *")
        validate_cron_expression("0 0-6 * * 1-5")

    def test_invalid_cron_expression_too_few_fields(self):
        with pytest.raises(InvalidCronExpressionError, match="Invalid cron"):
            validate_cron_expression("0 9 *")

    def test_invalid_cron_expression_too_many_fields(self):
        with pytest.raises(InvalidCronExpressionError, match="Invalid cron"):
            validate_cron_expression("0 9 * * * *")

    def test_invalid_cron_expression_bad_chars(self):
        with pytest.raises(InvalidCronExpressionError, match="Invalid cron"):
            validate_cron_expression("abc def ghi jkl mno")

    def test_invalid_cron_expression_empty(self):
        with pytest.raises(InvalidCronExpressionError, match="Invalid cron"):
            validate_cron_expression("")


@pytest.mark.unit
class TestScheduleManagerAdd:
    def test_add_schedule(self, schedule_manager, sample_definition):
        schedule_id = schedule_manager.add_schedule(sample_definition)

        assert schedule_id is not None
        assert len(schedule_id) > 0
        retrieved = schedule_manager.get_schedule(schedule_id)
        assert retrieved is not None
        assert retrieved.name == "daily-report"
        assert retrieved.agent_type == "report"

    def test_add_schedule_invalid_cron(self, schedule_manager):
        definition = ScheduleDefinition(
            name="bad-schedule",
            agent_type="test",
            prompt="test prompt",
            cron_expression="invalid",
        )
        with pytest.raises(InvalidCronExpressionError):
            schedule_manager.add_schedule(definition)


@pytest.mark.unit
class TestScheduleManagerRemove:
    def test_remove_schedule(self, schedule_manager, sample_definition):
        schedule_id = schedule_manager.add_schedule(sample_definition)
        assert schedule_manager.remove_schedule(schedule_id) is True
        assert schedule_manager.get_schedule(schedule_id) is None

    def test_remove_nonexistent(self, schedule_manager):
        assert schedule_manager.remove_schedule("nonexistent") is False


@pytest.mark.unit
class TestScheduleManagerList:
    def test_list_schedules_empty(self, schedule_manager):
        assert schedule_manager.list_schedules() == []

    def test_list_schedules(self, schedule_manager, sample_definition):
        sid1 = schedule_manager.add_schedule(sample_definition)
        other = ScheduleDefinition(
            name="weekly-check",
            agent_type="monitor",
            prompt="Run weekly check",
            cron_expression="0 0 * * 0",
        )
        sid2 = schedule_manager.add_schedule(other)

        schedules = schedule_manager.list_schedules()
        assert len(schedules) == 2
        ids = {s[0] for s in schedules}
        assert sid1 in ids
        assert sid2 in ids


@pytest.mark.unit
class TestScheduleManagerGet:
    def test_get_schedule(self, schedule_manager, sample_definition):
        schedule_id = schedule_manager.add_schedule(sample_definition)
        result = schedule_manager.get_schedule(schedule_id)
        assert result is not None
        assert result.name == "daily-report"

    def test_get_schedule_nonexistent(self, schedule_manager):
        assert schedule_manager.get_schedule("nonexistent") is None


@pytest.mark.unit
class TestScheduleManagerUpdate:
    def test_update_schedule(self, schedule_manager, sample_definition):
        schedule_id = schedule_manager.add_schedule(sample_definition)

        updated = ScheduleDefinition(
            name="updated-report",
            agent_type="report",
            prompt="Updated report prompt",
            cron_expression="0 10 * * *",
        )
        schedule_manager.update_schedule(schedule_id, updated)

        result = schedule_manager.get_schedule(schedule_id)
        assert result is not None
        assert result.name == "updated-report"
        assert result.cron_expression == "0 10 * * *"

    def test_update_schedule_not_found(self, schedule_manager):
        definition = ScheduleDefinition(
            name="test",
            agent_type="test",
            prompt="test",
            cron_expression="0 0 * * *",
        )
        with pytest.raises(ScheduleNotFoundError):
            schedule_manager.update_schedule("nonexistent", definition)

    def test_update_schedule_invalid_cron(self, schedule_manager, sample_definition):
        schedule_id = schedule_manager.add_schedule(sample_definition)

        bad_update = ScheduleDefinition(
            name="bad",
            agent_type="test",
            prompt="test",
            cron_expression="invalid",
        )
        with pytest.raises(InvalidCronExpressionError):
            schedule_manager.update_schedule(schedule_id, bad_update)


@pytest.mark.unit
class TestGenerateBeatSchedule:
    def test_generate_beat_schedule(self, schedule_manager, sample_definition):
        schedule_id = schedule_manager.add_schedule(sample_definition)

        with patch("celery.schedules.crontab") as mock_crontab:
            mock_crontab.return_value = "mocked-crontab"
            beat = schedule_manager.generate_beat_schedule()

        key = f"schedule-{schedule_id}"
        assert key in beat
        assert beat[key]["task"] == "ia_agent_fwk.execute_agent"
        assert beat[key]["args"] == ["report", "Generate daily report"]
        assert beat[key]["schedule"] == "mocked-crontab"

    def test_generate_beat_schedule_disabled(self, schedule_manager):
        disabled_def = ScheduleDefinition(
            name="disabled",
            agent_type="test",
            prompt="test",
            cron_expression="0 0 * * *",
            enabled=False,
        )
        schedule_manager.add_schedule(disabled_def)

        with patch("celery.schedules.crontab"):
            beat = schedule_manager.generate_beat_schedule()

        assert len(beat) == 0

    def test_generate_beat_schedule_empty(self, schedule_manager):
        with patch("celery.schedules.crontab"):
            beat = schedule_manager.generate_beat_schedule()
        assert beat == {}
