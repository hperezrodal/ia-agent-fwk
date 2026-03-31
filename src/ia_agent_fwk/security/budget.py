"""LLM API budget tracker.

Tracks token usage and cost per provider, enforcing daily/monthly spending
limits. Prevents runaway costs from abuse or bugs.

Usage:
    budget = BudgetTracker(daily_limit_usd=10.0)
    budget.record_usage("openai", "gpt-4o-mini", tokens_in=2500, tokens_out=200)

    if not budget.check_budget():
        raise BudgetExceededError("Daily budget exceeded")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

# Cost per 1M tokens (input/output) — update as pricing changes
PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "text-embedding-3-small": (0.02, 0.0),
    },
    "anthropic": {
        "claude-3-5-haiku-latest": (0.80, 4.00),
        "claude-sonnet-4-20250514": (3.00, 15.00),
    },
}


@dataclass
class UsageRecord:
    """A single token usage record."""

    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    timestamp: float


class BudgetTracker:
    """Track LLM API spending and enforce budget limits.

    Parameters
    ----------
    daily_limit_usd:
        Maximum daily spend in USD. 0 = unlimited.
    monthly_limit_usd:
        Maximum monthly spend in USD. 0 = unlimited.

    """

    def __init__(
        self,
        daily_limit_usd: float = 0.0,
        monthly_limit_usd: float = 0.0,
    ) -> None:
        self._daily_limit = daily_limit_usd
        self._monthly_limit = monthly_limit_usd
        self._records: list[UsageRecord] = []
        self._lock = Lock()

    def record_usage(
        self,
        provider: str,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> float:
        """Record token usage. Returns the estimated cost in USD."""
        pricing = PRICING.get(provider, {}).get(model, (0.0, 0.0))
        cost = (tokens_in * pricing[0] + tokens_out * pricing[1]) / 1_000_000

        record = UsageRecord(
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            timestamp=time.time(),
        )
        with self._lock:
            self._records.append(record)

        return cost

    def get_daily_spend(self) -> float:
        """Get total spend for today in USD."""
        today_start = time.time() - (time.time() % 86400)
        with self._lock:
            return sum(r.cost_usd for r in self._records if r.timestamp >= today_start)

    def get_monthly_spend(self) -> float:
        """Get total spend for the current month in USD."""
        # Approximate: last 30 days
        month_start = time.time() - 30 * 86400
        with self._lock:
            return sum(r.cost_usd for r in self._records if r.timestamp >= month_start)

    def check_budget(self) -> bool:
        """Check if spending is within budget limits.

        Returns True if OK, False if budget exceeded.
        """
        if self._daily_limit > 0 and self.get_daily_spend() >= self._daily_limit:
            logger.warning("Daily budget exceeded: $%.2f >= $%.2f", self.get_daily_spend(), self._daily_limit)
            return False
        if self._monthly_limit > 0 and self.get_monthly_spend() >= self._monthly_limit:
            logger.warning("Monthly budget exceeded: $%.2f >= $%.2f", self.get_monthly_spend(), self._monthly_limit)
            return False
        return True

    def get_summary(self) -> dict[str, float | int]:
        """Get spending summary."""
        return {
            "daily_spend_usd": round(self.get_daily_spend(), 4),
            "daily_limit_usd": self._daily_limit,
            "monthly_spend_usd": round(self.get_monthly_spend(), 4),
            "monthly_limit_usd": self._monthly_limit,
            "total_records": len(self._records),
        }

    def cleanup_old_records(self, max_age_days: int = 60) -> int:
        """Remove records older than max_age_days. Returns count removed."""
        cutoff = time.time() - max_age_days * 86400
        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r.timestamp >= cutoff]
            return before - len(self._records)
