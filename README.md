# saturatelimiter

Have you ever been in a situation where you needed to extract a truly enormous
number of records from an API service, but waiting around to process the
requests one-at-a-time is painfully slow? Then saturatelimiter is for you!

The saturatelimiter module makes concurrent, HTTP requests via the
[requests](https://requests.readthedocs.io/) module with automatic
self-throttling when it encounters 429 `Retry-After` rate limiting headers in
the response.

Transient server errors (HTTP 500, 502, 503, 504) and connection failures are
retried automatically with exponential backoff provided by the
[Tenacity](https://tenacity.readthedocs.io) module.

## Usage

### Fetch many URLs concurrently

This example reads URLs from a generator, submits every request immediately,
and processes each response as it completes. The session's thread pool queues
work internally so at most `num_threads` HTTP requests run at once.

```python
import asyncio
import os
from collections.abc import Iterator

from saturatelimiter import Session


def url_generator() -> Iterator[str]:
    """Yield URLs as they are discovered (files, pagination, etc.)."""
    for item_id in range(0, 11):
        yield f"https://api.myservice.example/v1/{item_id}"


async def fetch_urls(urls: Iterator[str], *, num_threads: int = os.cpu_count()) -> None:
    async with Session(
        num_threads=num_threads,
        headers={
            "User-Agent": "saturatelimiter/0.1",
            "Accept": "application/json",
        },
    ) as session:
        tasks = {
            asyncio.create_task(session.get(url, timeout=30)): url
            for url in urls
        }
        async for task in asyncio.as_completed(tasks):
            url = tasks[task]
            response = await task
            handle_response(url, response)


def handle_response(url: str, response) -> None:
    if response.ok:
        print(url, response.status_code, len(response.content))
    else:
        print(url, "failed:", response.status_code)


if __name__ == "__main__":
    asyncio.run(fetch_urls(url_generator()))
```

`Session` coordinates two kinds of automatic retry:

- **HTTP 429** with a `Retry-After` header pauses *all* workers until the
  backoff window expires, then retries the affected request.
- **Transient errors** (500/502/503/504 and connection failures) are retried
  per request with exponential backoff.

Set `num_threads` to the maximum number of concurrent HTTP requests you want;
additional submitted tasks wait in the thread pool until a worker is free.

## License

Apache License 2.0 — same as python-requests.
