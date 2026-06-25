"""Transient HTTP error retries via tenacity."""

from __future__ import annotations

import requests
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

DEFAULT_TRANSIENT_RETRY_ATTEMPTS = 10
_TRANSIENT_RETRY_WAIT = wait_exponential(multiplier=0.5, min=0.5, max=10)

# Backwards-compatible alias for tests and internal references.
_TRANSIENT_RETRY_ATTEMPTS = DEFAULT_TRANSIENT_RETRY_ATTEMPTS


class TransientHTTPError(Exception):
    """Raised when an HTTP response indicates a transient server error."""

    def __init__(self, response: requests.Response) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


def is_transient_status(status_code: int) -> bool:
    """Return True if ``status_code`` is a transient server error."""
    return status_code in {500, 502, 503, 504}


def _validate_attempts(attempts: int) -> None:
    if attempts < 1:
        raise ValueError("transient_retry_attempts must be at least 1")


def execute_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    attempts: int = DEFAULT_TRANSIENT_RETRY_ATTEMPTS,
    **kwargs: object,
) -> requests.Response:
    """Perform one logical HTTP request with transient-error retries.

    Retries on connection errors from ``requests`` and on HTTP 500, 502, 503,
    and 504 responses. Other status codes (including 429) are returned without
    raising.

    Args:
        session: The ``requests.Session`` for this worker thread.
        method: HTTP method.
        url: Request URL.
        attempts: Maximum number of attempts (including the first request).
        **kwargs: Arguments forwarded to ``session.request``.

    Returns:
        The ``requests.Response`` when a non-transient status is received.

    Raises:
        ValueError: If ``attempts`` is less than 1.
        TransientHTTPError: If transient HTTP errors persist after all
            retry attempts (when ``reraise=True``).
        requests.RequestException: If connection errors persist after all
            retry attempts.
    """
    _validate_attempts(attempts)

    for attempt in Retrying(
        retry=retry_if_exception_type(
            (TransientHTTPError, requests.RequestException)
        ),
        stop=stop_after_attempt(attempts),
        wait=_TRANSIENT_RETRY_WAIT,
        reraise=True,
    ):
        with attempt:
            response = session.request(method, url, **kwargs)
            if is_transient_status(response.status_code):
                raise TransientHTTPError(response)
            return response

    raise RuntimeError("execute_request finished without returning")
