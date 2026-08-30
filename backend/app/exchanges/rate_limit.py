"""Token-bucket rate limiting for exchange APIs.

Exchanges ban keys that exceed their limits, and a banned key during an open position is a real
risk event, not an inconvenience. The limiter is therefore *proactive*: it waits before sending
rather than reacting to 429s.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TokenBucket:
    """Classic token bucket.

    ``capacity`` tokens refill at ``refill_rate`` per second. A request costing more tokens than
    are available waits for the shortfall rather than failing.
    """

    capacity: float
    refill_rate: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        if self.tokens == 0.0:
            self.tokens = self.capacity

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    def try_consume(self, cost: float = 1.0) -> bool:
        """Take tokens if available. Never blocks."""
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def time_until_available(self, cost: float = 1.0) -> float:
        """Seconds until ``cost`` tokens will be available."""
        self._refill()
        if self.tokens >= cost:
            return 0.0
        return (cost - self.tokens) / self.refill_rate


class RateLimiter:
    """Named token buckets with async acquisition.

    Buckets are per endpoint class (``"public"``, ``"private"``, ``"order"``) because venues
    weight those differently.
    """

    def __init__(self, buckets: dict[str, TokenBucket] | None = None) -> None:
        self._buckets: dict[str, TokenBucket] = buckets or {}
        self._lock = asyncio.Lock()
        self._wait_total = 0.0
        self._wait_count = 0

    def add_bucket(self, name: str, capacity: float, refill_rate: float) -> None:
        self._buckets[name] = TokenBucket(capacity=capacity, refill_rate=refill_rate)

    async def acquire(self, name: str = "default", cost: float = 1.0) -> None:
        """Wait until ``cost`` tokens are available in bucket ``name``.

        An unknown bucket name is not an error: it means no limit is configured for that class.
        """
        bucket = self._buckets.get(name)
        if bucket is None:
            return
        while True:
            async with self._lock:
                if bucket.try_consume(cost):
                    return
                delay = bucket.time_until_available(cost)
            self._wait_total += delay
            self._wait_count += 1
            if delay > 1.0:
                logger.warning(
                    "exchange.rate_limit_wait", bucket=name, seconds=round(delay, 2)
                )
            await asyncio.sleep(min(delay, 5.0))

    @property
    def stats(self) -> dict[str, float]:
        return {
            "waits": float(self._wait_count),
            "total_wait_seconds": round(self._wait_total, 3),
        }


def default_limiter(requests_per_minute: int = 600, orders_per_second: int = 5) -> RateLimiter:
    """Conservative defaults suitable for retail exchange keys."""
    limiter = RateLimiter()
    per_second = requests_per_minute / 60.0
    limiter.add_bucket("public", capacity=per_second * 5, refill_rate=per_second)
    limiter.add_bucket("private", capacity=per_second * 2, refill_rate=per_second * 0.5)
    limiter.add_bucket(
        "order", capacity=float(orders_per_second * 2), refill_rate=float(orders_per_second)
    )
    return limiter
