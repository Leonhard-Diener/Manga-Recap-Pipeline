"""Network-free tests for MangaDex API rate-limit handling."""

import unittest
from unittest.mock import Mock, patch

from downloader import mangadex_api


class MangaDexApiTests(unittest.TestCase):
    def setUp(self):
        mangadex_api._next_request = 0.0

    def test_retries_429_once_and_keeps_user_agent(self):
        limited = Mock(status_code=429)
        limited.headers = {
            "X-RateLimit-Retry-After": "1001",
        }
        success = Mock(status_code=200)
        success.headers = {}

        with patch.object(
            mangadex_api.requests,
            "get",
            side_effect=[limited, success],
        ) as request, patch.object(
            mangadex_api.time,
            "monotonic",
            side_effect=[0.0, 0.0, 1.0, 1.0],
        ), patch.object(
            mangadex_api.time,
            "time",
            return_value=1000.0,
        ), patch.object(mangadex_api.time, "sleep") as sleep:
            response = mangadex_api.get("https://api.mangadex.org/manga")

        self.assertIs(response, success)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(
            request.call_args.kwargs["headers"]["User-Agent"],
            mangadex_api.USER_AGENT,
        )
        sleep.assert_any_call(1.0)


if __name__ == "__main__":
    unittest.main()
