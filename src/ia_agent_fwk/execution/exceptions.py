"""Exception hierarchy for the execution layer."""

from __future__ import annotations


class ExecutionError(Exception):
    """Base exception for all execution-layer errors."""


class JobNotFoundError(ExecutionError):
    """Raised when a job ID is not found in the result backend."""


class JobCancellationError(ExecutionError):
    """Raised when a job cancellation fails."""


class JobTimeoutError(ExecutionError):
    """Raised when a job exceeds its time limit."""


class ScheduleError(ExecutionError):
    """Raised for schedule-related errors."""


class ScheduleNotFoundError(ScheduleError):
    """Raised when a schedule ID is not found."""


class InvalidCronExpressionError(ScheduleError):
    """Raised when a cron expression is invalid."""


class TriggerError(ExecutionError):
    """Raised for trigger-related errors."""


class TriggerNotFoundError(TriggerError):
    """Raised when a trigger ID is not found."""
