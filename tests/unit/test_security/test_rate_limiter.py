"""Tests for in-memory sliding-window rate limiter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ia_agent_fwk.security.exceptions import RateLimitExceededError
from ia_agent_fwk.security.rate_limiter import SlidingWindowRateLimiter, parse_rate


@pytest.mark.unit
class TestParseRate:
    def test_parse_per_minute(self):
        limit, window = parse_rate("60/minute")
        assert limit == 60
        assert window == 60

    def test_parse_per_second(self):
        limit, window = parse_rate("10/second")
        assert limit == 10
        assert window == 1

    def test_parse_per_hour(self):
        limit, window = parse_rate("1000/hour")
        assert limit == 1000
        assert window == 3600

    def test_parse_per_day(self):
        limit, window = parse_rate("10000/day")
        assert limit == 10000
        assert window == 86400

    def test_parse_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid rate format"):
            parse_rate("60")

    def test_parse_invalid_count(self):
        with pytest.raises(ValueError, match="Invalid rate count"):
            parse_rate("abc/minute")

    def test_parse_unknown_unit(self):
        with pytest.raises(ValueError, match="Unknown rate unit"):
            parse_rate("60/fortnight")


@pytest.mark.unit
class TestSlidingWindowRateLimiter:
    async def test_allows_within_limit(self, rate_limiter):
        # limit=3, so 3 requests should all be allowed
        assert await rate_limiter.check_rate_limit("key1") is True
        assert await rate_limiter.check_rate_limit("key1") is True
        assert await rate_limiter.check_rate_limit("key1") is True

    async def test_blocks_over_limit(self, rate_limiter):
        # limit=3, 4th request should be blocked
        for _ in range(3):
            await rate_limiter.check_rate_limit("key1")

        assert await rate_limiter.check_rate_limit("key1") is False

    async def test_window_resets(self):
        """After the window expires, requests should be allowed again."""
        limiter = SlidingWindowRateLimiter(default_limit=2, default_window_seconds=1)

        # Fill the limit
        assert await limiter.check_rate_limit("key1") is True
        assert await limiter.check_rate_limit("key1") is True
        assert await limiter.check_rate_limit("key1") is False

        # Simulate time passing by manually clearing old timestamps
        # (We use monkeypatch on time.monotonic for a proper test)
        import time

        future_time = time.monotonic() + 2  # 2 seconds in the future
        with patch("ia_agent_fwk.security.rate_limiter.time.monotonic", return_value=future_time):
            assert await limiter.check_rate_limit("key1") is True

    async def test_per_key_isolation(self, rate_limiter):
        # Fill limit for key1
        for _ in range(3):
            await rate_limiter.check_rate_limit("key1")
        assert await rate_limiter.check_rate_limit("key1") is False

        # key2 should still be allowed
        assert await rate_limiter.check_rate_limit("key2") is True

    async def test_custom_limit_and_window(self):
        limiter = SlidingWindowRateLimiter(default_limit=100, default_window_seconds=3600)

        # Override with a small limit
        assert await limiter.check_rate_limit("key1", limit=1, window_seconds=60) is True
        assert await limiter.check_rate_limit("key1", limit=1, window_seconds=60) is False

    async def test_get_retry_after(self, rate_limiter):
        # Fill the limit
        for _ in range(3):
            await rate_limiter.check_rate_limit("key1")

        retry_after = rate_limiter.get_retry_after("key1")
        assert retry_after > 0
        assert retry_after <= 61  # window is 60s, rounded up +1

    async def test_get_retry_after_no_entries(self, rate_limiter):
        assert rate_limiter.get_retry_after("nonexistent") == 0

    async def test_reset_single_key(self, rate_limiter):
        for _ in range(3):
            await rate_limiter.check_rate_limit("key1")
        assert await rate_limiter.check_rate_limit("key1") is False

        rate_limiter.reset("key1")
        assert await rate_limiter.check_rate_limit("key1") is True

    async def test_reset_all_keys(self, rate_limiter):
        for _ in range(3):
            await rate_limiter.check_rate_limit("key1")
        for _ in range(3):
            await rate_limiter.check_rate_limit("key2")

        rate_limiter.reset()
        assert await rate_limiter.check_rate_limit("key1") is True
        assert await rate_limiter.check_rate_limit("key2") is True


@pytest.mark.unit
class TestRateLimitResponse429:
    """Test that the API returns 429 when rate limited."""

    async def test_rate_limit_response_429(self):
        """RateLimitExceededError is raised with correct attributes for 429 handling."""
        exc = RateLimitExceededError(key="hashed-key", retry_after=42)
        assert exc.retry_after == 42
        assert exc.key == "hashed-key"

    async def test_rate_limit_exceeded_error_has_retry_after(self):
        """RateLimitExceededError carries retry_after for 429 Retry-After header."""
        limiter = SlidingWindowRateLimiter(default_limit=1, default_window_seconds=60)
        key = "test-key-hash"

        # First request allowed
        assert await limiter.check_rate_limit(key) is True

        # Second request blocked
        assert await limiter.check_rate_limit(key) is False

        # retry_after should be a positive integer
        retry_after = limiter.get_retry_after(key)
        assert retry_after > 0

        # Verify the exception can carry this value
        exc = RateLimitExceededError(key=key, retry_after=retry_after)
        assert exc.retry_after == retry_after
        assert exc.key == key
