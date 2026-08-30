"""Platform primitives: config-free domain types, errors, logging, time and numerics."""

from app.core.clock import FixedClock, SystemClock, ensure_utc, utcnow
from app.core.logging import configure_logging, get_logger

__all__ = [
    "FixedClock",
    "SystemClock",
    "configure_logging",
    "ensure_utc",
    "get_logger",
    "utcnow",
]
