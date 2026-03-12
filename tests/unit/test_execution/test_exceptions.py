"""Tests for execution layer exception hierarchy."""

from __future__ import annotations

import pytest

from ia_agent_fwk.execution.exceptions import (
    ExecutionError,
    InvalidCronExpressionError,
    JobCancellationError,
    JobNotFoundError,
    JobTimeoutError,
    ScheduleError,
    ScheduleNotFoundError,
    TriggerError,
    TriggerNotFoundError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    def test_execution_error_is_exception(self):
        assert issubclass(ExecutionError, Exception)

    def test_job_not_found_is_execution_error(self):
        assert issubclass(JobNotFoundError, ExecutionError)

    def test_job_cancellation_is_execution_error(self):
        assert issubclass(JobCancellationError, ExecutionError)

    def test_job_timeout_is_execution_error(self):
        assert issubclass(JobTimeoutError, ExecutionError)

    def test_execution_error_message(self):
        msg = "test error"
        exc = ExecutionError(msg)
        assert str(exc) == "test error"

    def test_job_not_found_message(self):
        msg = "job-123 not found"
        exc = JobNotFoundError(msg)
        assert str(exc) == "job-123 not found"

    def test_job_timeout_message(self):
        msg = "timed out after 300s"
        exc = JobTimeoutError(msg)
        assert str(exc) == "timed out after 300s"

    def test_job_cancellation_message(self):
        msg = "failed to cancel"
        exc = JobCancellationError(msg)
        assert str(exc) == "failed to cancel"

    def test_exception_can_be_caught_as_base(self):
        msg = "not found"
        with pytest.raises(ExecutionError):
            raise JobNotFoundError(msg)

    def test_exception_can_be_caught_as_specific(self):
        msg = "not found"
        with pytest.raises(JobNotFoundError):
            raise JobNotFoundError(msg)

    # --- ScheduleError hierarchy ---

    def test_schedule_error_is_execution_error(self):
        assert issubclass(ScheduleError, ExecutionError)

    def test_schedule_error_message(self):
        msg = "schedule failed"
        exc = ScheduleError(msg)
        assert str(exc) == "schedule failed"

    def test_schedule_not_found_is_schedule_error(self):
        assert issubclass(ScheduleNotFoundError, ScheduleError)

    def test_schedule_not_found_is_execution_error(self):
        assert issubclass(ScheduleNotFoundError, ExecutionError)

    def test_schedule_not_found_message(self):
        msg = "schedule-123 not found"
        exc = ScheduleNotFoundError(msg)
        assert str(exc) == "schedule-123 not found"

    def test_schedule_not_found_caught_as_schedule_error(self):
        msg = "not found"
        with pytest.raises(ScheduleError):
            raise ScheduleNotFoundError(msg)

    def test_invalid_cron_is_schedule_error(self):
        assert issubclass(InvalidCronExpressionError, ScheduleError)

    def test_invalid_cron_is_execution_error(self):
        assert issubclass(InvalidCronExpressionError, ExecutionError)

    def test_invalid_cron_message(self):
        msg = "bad cron: * * *"
        exc = InvalidCronExpressionError(msg)
        assert str(exc) == "bad cron: * * *"

    def test_invalid_cron_caught_as_schedule_error(self):
        msg = "bad cron"
        with pytest.raises(ScheduleError):
            raise InvalidCronExpressionError(msg)

    # --- TriggerError hierarchy ---

    def test_trigger_error_is_execution_error(self):
        assert issubclass(TriggerError, ExecutionError)

    def test_trigger_error_message(self):
        msg = "trigger failed"
        exc = TriggerError(msg)
        assert str(exc) == "trigger failed"

    def test_trigger_not_found_is_trigger_error(self):
        assert issubclass(TriggerNotFoundError, TriggerError)

    def test_trigger_not_found_is_execution_error(self):
        assert issubclass(TriggerNotFoundError, ExecutionError)

    def test_trigger_not_found_message(self):
        msg = "trigger-456 not found"
        exc = TriggerNotFoundError(msg)
        assert str(exc) == "trigger-456 not found"

    def test_trigger_not_found_caught_as_trigger_error(self):
        msg = "not found"
        with pytest.raises(TriggerError):
            raise TriggerNotFoundError(msg)

    def test_trigger_not_found_caught_as_execution_error(self):
        msg = "not found"
        with pytest.raises(ExecutionError):
            raise TriggerNotFoundError(msg)
