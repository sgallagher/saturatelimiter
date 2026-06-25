"""Unit tests for transient HTTP retry helpers."""

from __future__ import annotations

import pytest
import requests
from tenacity import wait_fixed

from saturatelimiter._retry import (
    DEFAULT_TRANSIENT_RETRY_ATTEMPTS,
    TransientHTTPError,
    execute_request,
    is_transient_status,
)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (429, False),
        (404, False),
        (200, False),
    ],
)
def test_is_transient_status(status_code: int, expected: bool) -> None:
    assert is_transient_status(status_code) is expected


def test_execute_request_retries_transient_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "saturatelimiter._retry._TRANSIENT_RETRY_WAIT",
        wait_fixed(0),
    )
    calls = {"count": 0}
    session = requests.Session()

    def fake_request(
        method: str, url: str, **kwargs: object
    ) -> requests.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            response = requests.Response()
            response.status_code = 500
            return response
        response = requests.Response()
        response.status_code = 200
        return response

    monkeypatch.setattr(session, "request", fake_request)
    response = execute_request(
        session, "GET", "http://example.com/test", attempts=5
    )
    assert response.status_code == 200
    assert calls["count"] == 3


def test_execute_request_returns_last_transient_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "saturatelimiter._retry._TRANSIENT_RETRY_WAIT",
        wait_fixed(0),
    )
    session = requests.Session()

    def fake_request(
        method: str, url: str, **kwargs: object
    ) -> requests.Response:
        response = requests.Response()
        response.status_code = 503
        return response

    monkeypatch.setattr(session, "request", fake_request)
    with pytest.raises(TransientHTTPError) as exc_info:
        execute_request(
            session,
            "GET",
            "http://example.com/fail",
            attempts=3,
        )
    assert exc_info.value.response.status_code == 503


def test_execute_request_exceeds_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "saturatelimiter._retry._TRANSIENT_RETRY_WAIT",
        wait_fixed(0),
    )
    calls = {"count": 0}
    session = requests.Session()

    def fake_request(
        method: str, url: str, **kwargs: object
    ) -> requests.Response:
        calls["count"] += 1
        response = requests.Response()
        response.status_code = 500
        return response

    monkeypatch.setattr(session, "request", fake_request)
    with pytest.raises(TransientHTTPError) as exc_info:
        execute_request(
            session,
            "GET",
            "http://example.com/fail",
            attempts=4,
        )
    assert exc_info.value.response.status_code == 500
    assert calls["count"] == 4


def test_execute_request_invalid_attempts() -> None:
    session = requests.Session()
    with pytest.raises(ValueError, match="transient_retry_attempts"):
        execute_request(session, "GET", "http://example.com", attempts=0)


def test_default_attempts_constant() -> None:
    assert DEFAULT_TRANSIENT_RETRY_ATTEMPTS == 10
