# saturatelimiter

Concurrent HTTP requests via [requests](https://requests.readthedocs.io/) with global
429 `Retry-After` rate limiting across a shared thread pool.

## Usage

```python
import asyncio
from saturatelimiter import Session

async def main():
    with Session(
        num_threads=4,
        headers={"User-Agent": "saturatelimiter/0.1"},
    ) as session:
        response = await session.get("https://httpbin.org/get")
        print(response.status_code)

asyncio.run(main())
```

## License

Apache License 2.0 — same as python-requests.
