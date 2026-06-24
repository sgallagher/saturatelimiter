"""Unit tests for rate-limit lock and Retry-After parsing."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

from saturatelimiter._ratelimit import RateLimitLock, parse_retry_after


class TestParseRetryAfter:
    def test_none_returns_none(self) -> None:
        assert parse_retry_after(None) is None

    def test_empty_returns_none(self) -> None:
        assert parse_retry_after("") is None
        assert parse_retry_after("   ") is None

    def test_integer_seconds(self) -> None:
        assert parse_retry_after("30") == 30.0

    def test_float_seconds(self) -> None:
        assert parse_retry_after("1.5") == 1.5

    def test_http_date(self) -> None:
        future = datetime.now(UTC) + timedelta(seconds=60)
        http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        result = parse_retry_after(http_date)
        assert result is not None
        assert 55 <= result <= 65

    def test_invalid_returns_none(self) -> None:
        assert parse_retry_after("not-a-date-or-number") is None


class TestRateLimitLock:
    def test_extend_uses_max_semantics(self) -> None:
        lock = RateLimitLock()
        lock.extend(0.5)
        first_until = lock._limited_until
        lock.extend(0.1)
        assert lock._limited_until == first_until
        lock.extend(2.0)
        assert lock._limited_until > first_until

    def test_wait_if_needed_blocks_until_deadline(self) -> None:
        lock = RateLimitLock()
        lock.extend(0.2)
        start = time.monotonic()
        lock.wait_if_needed()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15

    def test_concurrent_threads_wait_for_global_lock(self) -> None:
        lock = RateLimitLock()
        lock.extend(0.3)
        elapsed_times: list[float] = []
        barrier = threading.Barrier(3)

        def worker() -> None:
            barrier.wait()
            start = time.monotonic()
            lock.wait_if_needed()
            elapsed_times.append(time.monotonic() - start)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2.0)
        assert len(elapsed_times) == 2
        assert all(elapsed >= 0.2 for elapsed in elapsed_times)
