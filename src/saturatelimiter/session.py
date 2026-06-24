"""Async HTTP session with thread-pooled requests and global 429 handling."""

from __future__ import annotations

import asyncio
import functools
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import requests

from saturatelimiter._ratelimit import RateLimitLock, parse_retry_after


def _default_threads() -> int:
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return count if count is not None else 1


class Session:
    """Thread-pooled HTTP client with global 429 ``Retry-After`` coordination.

    Use as a synchronous context manager (``with Session(...) as session:``) to
    create a shared ``requests.Session``, worker thread pool, and global rate-
    limit lock. Submit work through async methods such as :meth:`request` and
    :meth:`get`; each call is executed on a worker thread and returns an
    awaitable ``requests.Response``.

    When any worker receives HTTP 429 and a parseable ``Retry-After`` header,
    all workers pause starting new requests until that window expires, then
    the affected request is retried. Responses with status 429 but no
    ``Retry-After`` header are returned to the caller without retrying.

    Default request headers may be set at construction time; per-request
    ``headers=`` keyword arguments override those defaults using
    ``requests.Session`` merge semantics.

    Examples:
        >>> import asyncio
        >>> from saturatelimiter import Session
        >>>
        >>> async def fetch():
        ...   with Session(headers={"Accept": "application/json"}) as session:
        ...     return await session.get("https://httpbin.org/get")
        ...
        >>> asyncio.run(fetch()).status_code
        200
    """

    def __init__(
        self,
        *,
        num_threads: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """
        Configure a new session (enter the context manager before requesting).

        Args:
            num_threads: Maximum concurrent worker threads. Defaults to
                ``os.process_cpu_count()`` when available, otherwise
                ``os.cpu_count()``, falling back to ``1``.
            headers: Optional default headers applied to every request via the
                underlying ``requests.Session``. Per-request ``headers=``
                override these values by key.
        """
        if num_threads is not None and num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        self._num_threads = num_threads
        self._headers = dict(headers) if headers is not None else None
        self._session: requests.Session | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._rate_limit: RateLimitLock | None = None
        self._session_lock: Lock | None = None

    def __enter__(self) -> Session:
        """Enter the session context and start the worker thread pool.

        Returns:
            This session instance.

        Raises:
            RuntimeError: If the session is already active.
        """
        if self._executor is not None:
            raise RuntimeError("Session is already active")

        self._session = requests.Session()
        if self._headers:
            self._session.headers.update(self._headers)
        self._rate_limit = RateLimitLock()
        self._session_lock = Lock()
        workers = (
            self._num_threads
            if self._num_threads is not None
            else _default_threads()
        )
        self._executor = ThreadPoolExecutor(max_workers=workers)
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        """Shut down workers and close the underlying ``requests.Session``."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        if self._session is not None:
            self._session.close()
            self._session = None
        self._rate_limit = None
        self._session_lock = None

    def _ensure_active(
        self,
    ) -> tuple[ThreadPoolExecutor, requests.Session, RateLimitLock, Lock]:
        if self._executor is None or self._session is None:
            raise RuntimeError(
                "Session is not active; use it as a context manager"
            )
        assert self._rate_limit is not None
        assert self._session_lock is not None
        return (
            self._executor,
            self._session,
            self._rate_limit,
            self._session_lock,
        )

    def _sync_request(
        self, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        _, session, rate_limit, session_lock = self._ensure_active()
        while True:
            rate_limit.wait_if_needed()
            with session_lock:
                response = session.request(method, url, **kwargs)
            if response.status_code != 429:
                return response
            retry_after = parse_retry_after(
                response.headers.get("Retry-After")
            )
            if retry_after is None:
                return response
            rate_limit.extend(retry_after)

    async def request(
        self, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        """Submit an HTTP request and await the response.

        Executes on a worker thread using the shared ``requests.Session``.
        Accepts the same keyword arguments as ``requests.Session.request``.

        Args:
            method: HTTP method (e.g. ``"GET"``, ``"POST"``).
            url: Request URL.
            **kwargs: Per-request options passed to
                ``requests.Session.request``, including ``params``, ``data``,
                ``headers``, ``cookies``, ``files``, ``auth``, ``timeout``,
                ``allow_redirects``, ``proxies``, ``hooks``, ``stream``,
                ``proxies``, ``hooks``, ``stream``, ``verify``, ``cert``,
                and ``json``.

        Returns:
            The ``requests.Response`` from the server. On HTTP 429, the call
            retries automatically when ``Retry-After`` is present and
            parseable; otherwise the 429 response is returned immediately.

        Raises:
            RuntimeError: If called outside the context manager.
            requests.RequestException: On connection or transport errors from
                ``requests``.

        Example:
            >>> import asyncio
            >>> from saturatelimiter import Session
            >>>
            >>> async def example():
            ...     with Session() as session:
            ...         return await session.request(
            ...             "GET",
            ...             "https://httpbin.org/get",
            ...             headers={"Accept": "application/json"},
            ...         )
            ...
            >>> asyncio.run(example()).status_code
            200
        """
        executor, _, _, _ = self._ensure_active()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            executor,
            functools.partial(self._sync_request, method, url, **kwargs),
        )
        return await asyncio.wrap_future(future)

    async def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Submit an HTTP GET request and await the response.

        Args:
            url: Request URL.
            **kwargs: Per-request options passed to :meth:`request`, such as
                ``params``, ``headers``, and ``timeout``.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 retry behavior.

        Raises:
            RuntimeError: If called outside the context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Submit an HTTP POST request and await the response.

        Args:
            url: Request URL.
            **kwargs: Per-request options passed to :meth:`request`, such as
                ``data``, ``json``, ``headers``, and ``timeout``.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 retry behavior.

        Raises:
            RuntimeError: If called outside the context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> requests.Response:
        """Submit an HTTP PUT request and await the response.

        Args:
            url: Request URL.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 retry behavior.

        Raises:
            RuntimeError: If called outside the context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> requests.Response:
        """Submit an HTTP PATCH request and await the response.

        Args:
            url: Request URL.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 retry behavior.

        Raises:
            RuntimeError: If called outside the context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> requests.Response:
        """Submit an HTTP DELETE request and await the response.

        Args:
            url: Request URL.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 retry behavior.

        Raises:
            RuntimeError: If called outside the context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request("DELETE", url, **kwargs)

    async def head(self, url: str, **kwargs: Any) -> requests.Response:
        """Submit an HTTP HEAD request and await the response.

        Args:
            url: Request URL.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 retry behavior.

        Raises:
            RuntimeError: If called outside the context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request("HEAD", url, **kwargs)

    async def options(self, url: str, **kwargs: Any) -> requests.Response:
        """Submit an HTTP OPTIONS request and await the response.

        Args:
            url: Request URL.
            **kwargs: Per-request options passed to :meth:`request`.

        Returns:
            The ``requests.Response`` from the server. See :meth:`request` for
            429 retry behavior.

        Raises:
            RuntimeError: If called outside the context manager.
            requests.RequestException: On connection or transport errors.
        """
        return await self.request("OPTIONS", url, **kwargs)
