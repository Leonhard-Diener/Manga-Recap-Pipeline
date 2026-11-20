"""Shared rate-limit handling for synchronous MangaDex API requests."""

from __future__ import annotations

import threading
import time
import math

import requests


# The public API limit is five requests per second. A 250 ms interval leaves
# a modest buffer while keeping search and pagination quick.
API_REQUEST_INTERVAL = 0.25
USER_AGENT = "MangaDownloader/1.0"

_request_lock = threading.Lock()
_next_request = 0.0


def get(url: str, **kwargs) -> requests.Response:
    """GET from api.mangadex.org with shared pacing and 429 handling."""
    global _next_request

    headers = dict(kwargs.pop("headers", {}))
    headers.setdefault("User-Agent", USER_AGENT)

    with _request_lock:
        for attempt in range(5):
            delay = _next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            response = requests.get(url, headers=headers, **kwargs)
            _next_request = time.monotonic() + API_REQUEST_INTERVAL

            if response.status_code != 429:
                return response
            if attempt == 4:
                response.raise_for_status()

            retry_after = response.headers.get("X-RateLimit-Retry-After")
            try:
                wait_time = max(float(retry_after) - time.time(), 1.0)
            except (TypeError, ValueError):
                wait_time = 60.0
            wait_time = min(wait_time, 300.0) if math.isfinite(wait_time) else 60.0

            print(f"MangaDex rate limit reached. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
