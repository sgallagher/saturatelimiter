"""Internal rate-limit coordination for saturatelimiter."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import Condition


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value into seconds to wait.

    Args:
        value: Raw ``Retry-After`` header value, or ``None`` if absent.

    Returns:
        Number of seconds to wait before retrying, or ``None`` if the value
        cannot be parsed.
    """
    if value is None:
        return None

    value = value.strip()
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        retry_time = parsedate_to_datetime(value)
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=UTC)
        delta = (retry_time - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError, OverflowError):
        return None


class RateLimitLock:
    """Global rate-limit lock shared across worker threads.

    When any worker receives HTTP 429 with a ``Retry-After`` header, all
    workers block starting new requests until the backoff window expires.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._limited_until = 0.0

    def wait_if_needed(self) -> None:
        """Block until the current global rate-limit window has expired."""
        with self._condition:
            now = time.monotonic()
            while now < self._limited_until:
                self._condition.wait(timeout=self._limited_until - now)
                now = time.monotonic()

    def extend(self, retry_after_seconds: float) -> None:
        """Extend the global rate-limit window.

        Args:
            retry_after_seconds: Seconds to wait before new requests may start.
        """
        with self._condition:
            new_until = time.monotonic() + retry_after_seconds
            if new_until > self._limited_until:
                self._limited_until = new_until
            self._condition.notify_all()
