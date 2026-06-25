"""Async HTTP session with thread-pooled requests and global 429 handling."""

from __future__ import annotations

import asyncio
import functools
import os
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from saturatelimiter._ratelimit import RateLimitLock, parse_retry_after
from saturatelimiter._retry import (
    DEFAULT_TRANSIENT_RETRY_ATTEMPTS,
    TransientHTTPError,
    _validate_attempts,
    execute_request,
)


def _default_threads() -> int:
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return count if count is not None else 1


class Session:
    """Thread-pooled HTTP client with global 429 ``Retry-After`` coordination.

    Use as an async context manager (``async with Session(...) as session:``)
    to start a worker thread pool and global rate-limit coordination. Each
    worker thread uses its own ``requests.Session`` so multiple HTTP requests
    can execute concurrently. Submit work through async methods such as
    :meth:`request` and :meth:`get`; each call is executed on a worker thread
    and returns an awaitable ``requests.Response``.

    When any worker receives HTTP 429 and a parseable ``Retry-After`` header,
    all workers pause starting new requests until that window expires, then
    the affected request is retried. Responses with status 429 but no
    ``Retry-After`` header are returned to the caller without retrying.

    Transient server errors (HTTP 500, 502, 503, and 504) and connection
    failures are retried automatically with exponential backoff via
    ``tenacity`` before a response is returned or an error is raised.

    Default request headers may be set at construction time; per-request
    ``headers=`` keyword arguments override those defaults using
    ``requests.Session`` merge semantics.

    Examples:
        >>> import asyncio
        >>> from saturatelimiter import Session
        >>>
        >>> async def fetch():
        ...     async with Session(
        ...         headers={"Accept": "application/json"},
        ...     ) as session:
        ...         return await session.get("https://httpbin.org/get")
        ...
        >>> asyncio.run(fetch()).status_code
        200
    """

    def __init__(
        self,
        *,
        num_threads: int | None = None,
        headers: Mapping[str, str] | None = None,
        transient_retry_attempts: int | None = None,
    ) -> None:
        """
        Configure a new session.

        Enter the async context manager before making requests.

        Args:
            num_threads: Maximum concurrent worker threads. Defaults to
                ``os.process_cpu_count()`` when available, otherwise
                ``os.cpu_count()``, falling back to ``1``.
            headers: Optional default headers applied to every request via each
                worker's ``requests.Session``. Per-request ``headers=``
                override these values by key.
            transient_retry_attempts: Maximum attempts for transient HTTP
                errors (500, 502, 503, 504) and connection failures.
                Defaults to ``DEFAULT_TRANSIENT_RETRY_ATTEMPTS`` (10).
                Per-request ``transient_retry_attempts=`` overrides this
                value.
        """
        if num_threads is not None and num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        attempts = (
            transient_retry_attempts
            if transient_retry_attempts is not None
            else DEFAULT_TRANSIENT_RETRY_ATTEMPTS
        )
        _validate_attempts(attempts)
        self._num_threads = num_threads
        self._headers = dict(headers) if headers is not None else None
        self._transient_retry_attempts = attempts
        self._executor: ThreadPoolExecutor | None = None
        self._rate_limit: RateLimitLock | None = None
        self._thread_local = threading.local()
        self._sessions: list[requests.Session] = []

    async def __aenter__(self) -> Session:
        """Enter the session context and start the worker thread pool.

        Returns:
            This session instance.

        Raises:
            RuntimeError: If the session is already active.
        """
        if self._executor is not None:
            raise RuntimeError("Session is already active")

        self._rate_limit = RateLimitLock()
        self._thread_local = threading.local()
        self._sessions = []
        workers = (
            self._num_threads
            if self._num_threads is not None
            else _default_threads()
        )
        self._executor = ThreadPoolExecutor(max_workers=workers)
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        """Shut down workers and close per-thread sessions."""
        if self._executor is not None:
            executor = self._executor
            self._executor = None
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: executor.shutdown(wait=True, cancel_futures=True),
            )
        for session in self._sessions:
            session.close()
        self._sessions.clear()
        self._rate_limit = None

    def _ensure_active(
        self,
    ) -> tuple[ThreadPoolExecutor, RateLimitLock]:
        if self._executor is None:
            raise RuntimeError(
                "Session is not active; use it as an async context manager"
            )
        assert self._rate_limit is not None
        return self._executor, self._rate_limit

    def _get_thread_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            if self._headers:
                session.headers.update(self._headers)
            self._thread_local.session = session
            self._sessions.append(session)
        return session

    def _sync_request(
        self,
        method: str,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        _, rate_limit = self._ensure_active()
        session = self._get_thread_session()
        attempts = (
            transient_retry_attempts
            if transient_retry_attempts is not None
            else self._transient_retry_attempts
        )
        _validate_attempts(attempts)
        while True:
            rate_limit.wait_if_needed()
            try:
                response = execute_request(
                    session,
                    method,
                    url,
                    attempts=attempts,
                    **kwargs,
                )
            except TransientHTTPError as exc:
                return exc.response
            if response.status_code != 429:
                return response
            retry_after = parse_retry_after(
                response.headers.get("Retry-After")
            )
            if retry_after is None:
                return response
            rate_limit.extend(retry_after)

    async def request(
        self,
        method: str,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Submit an HTTP request and await the response.

        Executes on a worker thread using that thread's ``requests.Session``.
        Accepts the same keyword arguments as ``requests.Session.request``.

        Args:
            method: HTTP method (e.g. ``"GET"``, ``"POST"``).
            url: Request URL.
            transient_retry_attempts: Maximum attempts for transient HTTP
                errors and connection failures for this request only.
                Overrides the session constructor default when set.
            **kwargs: Per-request options passed to
                ``requests.Session.request``, including ``params``, ``data``,
                ``headers``, ``cookies``, ``files``, ``auth``, ``timeout``,
                ``allow_redirects``, ``proxies``, ``hooks``, ``stream``,
                ``verify``, ``cert``, and ``json``.

        Returns:
            The ``requests.Response`` from the server. On HTTP 429, the call
            retries automatically when ``Retry-After`` is present and
            parseable; otherwise the 429 response is returned immediately.
            HTTP 500, 502, 503, and 504 responses are retried with
            exponential backoff; the last response is returned if retries
            are exhausted.

        Raises:
            RuntimeError: If called outside the async context manager.
            requests.RequestException: On connection or transport errors from
                ``requests`` after transient-error retries are exhausted.

        Example:
            >>> import asyncio
            >>> from saturatelimiter import Session
            >>>
            >>> async def example():
            ...     async with Session() as session:
            ...         return await session.request(
            ...             "GET",
            ...             "https://httpbin.org/get",
            ...             headers={"Accept": "application/json"},
            ...         )
            ...
            >>> asyncio.run(example()).status_code
            200
        """
        executor, _ = self._ensure_active()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            executor,
            functools.partial(
                self._sync_request,
                method,
                url,
                transient_retry_attempts=transient_retry_attempts,
                **kwargs,
            ),
        )
        return await asyncio.wrap_future(future)

    async def get(
        self,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Submit an HTTP GET request and await the response.

        Args:
            url: Request URL.
            transient_retry_attempts: Per-request transient retry limit.
                Overrides the session default when set.
            **kwargs: Per-request options passed to :meth:`request`, such as
                ``params``, ``headers``, and ``timeout``.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 and transient-error retry behavior.

        Raises:
            RuntimeError: If called outside the async context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request(
            "GET",
            url,
            transient_retry_attempts=transient_retry_attempts,
            **kwargs,
        )

    async def post(
        self,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Submit an HTTP POST request and await the response.

        Args:
            url: Request URL.
            transient_retry_attempts: Per-request transient retry limit.
                Overrides the session default when set.
            **kwargs: Per-request options passed to :meth:`request`, such as
                ``data``, ``json``, ``headers``, and ``timeout``.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 and transient-error retry behavior.

        Raises:
            RuntimeError: If called outside the async context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request(
            "POST",
            url,
            transient_retry_attempts=transient_retry_attempts,
            **kwargs,
        )

    async def put(
        self,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Submit an HTTP PUT request and await the response.

        Args:
            url: Request URL.
            transient_retry_attempts: Per-request transient retry limit.
                Overrides the session default when set.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 and transient-error retry behavior.

        Raises:
            RuntimeError: If called outside the async context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request(
            "PUT",
            url,
            transient_retry_attempts=transient_retry_attempts,
            **kwargs,
        )

    async def patch(
        self,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Submit an HTTP PATCH request and await the response.

        Args:
            url: Request URL.
            transient_retry_attempts: Per-request transient retry limit.
                Overrides the session default when set.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 and transient-error retry behavior.

        Raises:
            RuntimeError: If called outside the async context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request(
            "PATCH",
            url,
            transient_retry_attempts=transient_retry_attempts,
            **kwargs,
        )

    async def delete(
        self,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Submit an HTTP DELETE request and await the response.

        Args:
            url: Request URL.
            transient_retry_attempts: Per-request transient retry limit.
                Overrides the session default when set.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 and transient-error retry behavior.

        Raises:
            RuntimeError: If called outside the async context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request(
            "DELETE",
            url,
            transient_retry_attempts=transient_retry_attempts,
            **kwargs,
        )

    async def head(
        self,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Submit an HTTP HEAD request and await the response.

        Args:
            url: Request URL.
            transient_retry_attempts: Per-request transient retry limit.
                Overrides the session default when set.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 and transient-error retry behavior.

        Raises:
            RuntimeError: If called outside the async context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request(
            "HEAD",
            url,
            transient_retry_attempts=transient_retry_attempts,
            **kwargs,
        )

    async def options(
        self,
        url: str,
        *,
        transient_retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Submit an HTTP OPTIONS request and await the response.

        Args:
            url: Request URL.
            transient_retry_attempts: Per-request transient retry limit.
                Overrides the session default when set.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 and transient-error retry behavior.

        Raises:
            RuntimeError: If called outside the async context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request(
            "OPTIONS",
            url,
            transient_retry_attempts=transient_retry_attempts,
            **kwargs,
        )
