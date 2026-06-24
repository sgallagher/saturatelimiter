"""Concurrent HTTP requests with global 429 Retry-After rate limiting.

This package provides :class:`~saturatelimiter.session.Session`, an async
wrapper around :mod:`requests` that executes HTTP calls on a shared thread
pool and coordinates backoff when servers respond with HTTP 429 and a
``Retry-After`` header.
"""

from saturatelimiter.session import Session

__all__ = ["Session"]
__version__ = "0.1.0"
