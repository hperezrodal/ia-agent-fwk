"""Shared fixtures for execution layer unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ia_agent_fwk.execution.manager import JobManager


@pytest.fixture
def mock_celery_app():
    """Create a mock Celery app for testing."""
    app = MagicMock()
    app.conf = MagicMock()
    return app


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    redis = MagicMock()
    redis.zadd = MagicMock(return_value=1)
    redis.zcard = MagicMock(return_value=0)
    redis.zrevrange = MagicMock(return_value=[])
    redis.hset = MagicMock(return_value=True)
    redis.hgetall = MagicMock(return_value={})
    return redis


@pytest.fixture
def job_manager(mock_celery_app, mock_redis):
    """Create a JobManager with mocked Celery app and Redis."""
    return JobManager(celery_app=mock_celery_app, redis_client=mock_redis)


@pytest.fixture
def job_manager_no_redis(mock_celery_app):
    """Create a JobManager with mocked Celery app and no Redis."""
    return JobManager(celery_app=mock_celery_app, redis_client=None)
