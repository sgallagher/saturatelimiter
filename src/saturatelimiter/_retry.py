"""Transient HTTP error retries via tenacity."""

from __future__ import annotations

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Shared retry policy for server/transient failures (not 429 rate limits).
_TRANSIENT_RETRY_ATTEMPTS = 10
_TRANSIENT_RETRY_WAIT = wait_exponential(multiplier=0.5, min=0.5, max=10)


class TransientHTTPError(Exception):
    """Raised when an HTTP response indicates a transient server error."""

    def __init__(self, response: requests.Response) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


def is_transient_status(status_code: int) -> bool:
    """Return True if ``status_code`` is a transient server error."""
    return status_code in {500, 502, 503, 504}


@retry(
    retry=retry_if_exception_type(
        (TransientHTTPError, requests.RequestException)
    ),
    stop=stop_after_attempt(_TRANSIENT_RETRY_ATTEMPTS),
    wait=_TRANSIENT_RETRY_WAIT,
    reraise=True,
)
def execute_request(
    session: requests.Session,
    method: str,
    url: str,
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
        **kwargs: Arguments forwarded to ``session.request``.

    Returns:
        The ``requests.Response`` when a non-transient status is received.

    Raises:
        TransientHTTPError: If transient HTTP errors persist after all
            retry attempts (when ``reraise=True``).
        requests.RequestException: If connection errors persist after all
            retry attempts.
    """
    response = session.request(method, url, **kwargs)
    if is_transient_status(response.status_code):
        raise TransientHTTPError(response)
    return response
