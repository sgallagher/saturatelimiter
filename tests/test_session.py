"""Integration tests for Session async API and 429 handling."""

from __future__ import annotations

import asyncio
import time

import pytest
from pytest_httpserver import HTTPServer
from tenacity import wait_fixed
from werkzeug.wrappers import Request, Response

from saturatelimiter import Session
from saturatelimiter._retry import _TRANSIENT_RETRY_ATTEMPTS, execute_request


@pytest.fixture
def httpserver(httpserver: HTTPServer) -> HTTPServer:
    httpserver.clear()
    return httpserver


async def test_basic_get(httpserver: HTTPServer) -> None:
    httpserver.expect_request("/get").respond_with_json({"ok": True})
    async with Session() as session:
        response = await session.get(httpserver.url_for("/get"))
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.parametrize(
    ("method_name", "expected_method"),
    [
        ("post", "POST"),
        ("put", "PUT"),
        ("patch", "PATCH"),
        ("delete", "DELETE"),
        ("head", "HEAD"),
        ("options", "OPTIONS"),
    ],
)
async def test_convenience_methods(
    httpserver: HTTPServer,
    method_name: str,
    expected_method: str,
) -> None:
    seen: dict[str, str] = {}

    def handler(request: Request) -> Response:
        seen["method"] = request.method
        return Response("ok", status=200)

    httpserver.expect_request(
        "/verb", method=expected_method
    ).respond_with_handler(handler)
    async with Session() as session:
        method = getattr(session, method_name)
        response = await method(httpserver.url_for("/verb"))
    assert response.status_code == 200
    assert seen["method"] == expected_method


async def test_429_then_200(httpserver: HTTPServer) -> None:
    state = {"count": 0}

    def handler(request: Request) -> Response:
        state["count"] += 1
        if state["count"] == 1:
            return Response(
                "slow down", status=429, headers={"Retry-After": "1"}
            )
        return Response("ok", status=200)

    httpserver.expect_request("/retry").respond_with_handler(handler)
    async with Session() as session:
        response = await session.get(httpserver.url_for("/retry"))
    assert response.status_code == 200
    assert state["count"] == 2


async def test_500_then_200(
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execute_request.retry, "wait", wait_fixed(0))
    state = {"count": 0}

    def handler(request: Request) -> Response:
        state["count"] += 1
        if state["count"] == 1:
            return Response("error", status=500)
        return Response("ok", status=200)

    httpserver.expect_request("/server-error").respond_with_handler(handler)
    async with Session() as session:
        response = await session.get(httpserver.url_for("/server-error"))
    assert response.status_code == 200
    assert state["count"] == 2


async def test_500_exhausted_retries(
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execute_request.retry, "wait", wait_fixed(0))
    state = {"count": 0}

    def handler(request: Request) -> Response:
        state["count"] += 1
        return Response("error", status=500)

    httpserver.expect_request("/always-fail").respond_with_handler(handler)
    async with Session() as session:
        response = await session.get(httpserver.url_for("/always-fail"))
    assert response.status_code == 500
    assert state["count"] == _TRANSIENT_RETRY_ATTEMPTS


async def test_429_without_retry_after(httpserver: HTTPServer) -> None:
    httpserver.expect_request("/limited").respond_with_data(
        "too many",
        status=429,
    )
    async with Session() as session:
        response = await session.get(httpserver.url_for("/limited"))
    assert response.status_code == 429


async def test_global_lock_coordinates_threads(httpserver: HTTPServer) -> None:
    state = {"count": 0}

    def handler(request: Request) -> Response:
        state["count"] += 1
        if state["count"] <= 2:
            return Response(
                "slow down", status=429, headers={"Retry-After": "1"}
            )
        return Response("ok", status=200)

    httpserver.expect_request("/shared").respond_with_handler(handler)
    async with Session(num_threads=2) as session:
        start = time.monotonic()
        responses = await asyncio.gather(
            session.get(httpserver.url_for("/shared")),
            session.get(httpserver.url_for("/shared")),
        )
        elapsed = time.monotonic() - start
    assert all(response.status_code == 200 for response in responses)
    assert elapsed >= 0.9


async def test_context_manager_cleanup(httpserver: HTTPServer) -> None:
    session = Session()
    async with session:
        httpserver.expect_request("/ok").respond_with_data("ok", status=200)
        response = await session.get(httpserver.url_for("/ok"))
        assert response.status_code == 200
    with pytest.raises(RuntimeError, match="not active"):
        await session.get(httpserver.url_for("/ok"))


async def test_session_headers(httpserver: HTTPServer) -> None:
    seen: dict[str, str] = {}

    def handler(request: Request) -> Response:
        seen["user_agent"] = request.headers.get("User-Agent", "")
        seen["accept"] = request.headers.get("Accept", "")
        return Response("ok", status=200)

    httpserver.expect_request("/headers").respond_with_handler(handler)
    async with Session(
        headers={"User-Agent": "test-agent", "Accept": "application/json"},
    ) as session:
        await session.get(httpserver.url_for("/headers"))
    assert seen["user_agent"] == "test-agent"
    assert seen["accept"] == "application/json"


async def test_header_override(httpserver: HTTPServer) -> None:
    seen: dict[str, str] = {}

    def handler(request: Request) -> Response:
        seen["accept"] = request.headers.get("Accept", "")
        return Response("ok", status=200)

    httpserver.expect_request("/override").respond_with_handler(handler)
    async with Session(headers={"Accept": "application/json"}) as session:
        await session.get(
            httpserver.url_for("/override"),
            headers={"Accept": "text/plain"},
        )
    assert seen["accept"] == "text/plain"


async def test_per_request_params(httpserver: HTTPServer) -> None:
    httpserver.expect_request(
        "/params", query_string="key=value"
    ).respond_with_data(
        "ok",
        status=200,
    )
    async with Session() as session:
        response = await session.get(
            httpserver.url_for("/params"),
            params={"key": "value"},
        )
    assert response.status_code == 200


async def test_request_outside_context_manager() -> None:
    session = Session()
    with pytest.raises(RuntimeError, match="not active"):
        await session.request("GET", "http://example.com")


async def test_invalid_num_threads() -> None:
    with pytest.raises(ValueError, match="num_threads"):
        Session(num_threads=0)
