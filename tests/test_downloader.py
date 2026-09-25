"""Network-free tests for manga download scheduling."""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx

from downloader import download_chapter as downloader


class _FakeClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class DownloadSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_chapters_share_limits_and_only_four_run_at_once(self):
        active = 0
        maximum_active = 0
        clients = set()
        download_semaphores = set()
        at_home_limiters = set()
        image_cooldowns = set()

        async def fake_download(
            chapter_id,
            manga_name,
            chapter_index,
            client,
            download_semaphore,
            at_home_limiter,
            image_cooldown,
        ):
            nonlocal active, maximum_active
            clients.add(id(client))
            download_semaphores.add(id(download_semaphore))
            at_home_limiters.add(id(at_home_limiter))
            image_cooldowns.add(id(image_cooldown))
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return chapter_index

        with patch.object(downloader.httpx, "AsyncClient", return_value=_FakeClient()), \
                patch.object(downloader, "_download_chapter_with_client",
                             side_effect=fake_download):
            result = await downloader.download_chapters(
                [f"chapter-{index}" for index in range(8)],
                "Manga",
            )

        self.assertEqual(result, list(range(1, 9)))
        self.assertEqual(maximum_active, downloader.MAX_CONCURRENT_CHAPTERS)
        self.assertEqual(len(clients), 1)
        self.assertEqual(len(download_semaphores), 1)
        self.assertEqual(len(at_home_limiters), 1)
        self.assertEqual(len(image_cooldowns), 1)

    async def test_global_interval_limiter_spaces_concurrent_requests(self):
        limiter = downloader._IntervalLimiter(0.01)
        loop = asyncio.get_running_loop()

        async def pass_limiter():
            await limiter.wait()
            return loop.time()

        times = await asyncio.gather(*(pass_limiter() for _ in range(3)))
        self.assertGreaterEqual(times[1] - times[0], 0.008)
        self.assertGreaterEqual(times[2] - times[1], 0.008)

    async def test_chapter_source_refreshes_once_per_failed_revision(self):
        initial = {
            "baseUrl": "https://old.example",
            "chapter": {
                "hash": "old-hash",
                "data": ["page.jpg"],
                "chapter": "1",
            },
        }
        refreshed = {
            "baseUrl": "https://new.example",
            "chapter": {
                "hash": "new-hash",
                "data": ["page.jpg"],
                "chapter": "1",
            },
        }
        source = downloader._ChapterSource("chapter-id", initial)

        async def fake_get_server(client, chapter_id, limiter):
            await asyncio.sleep(0)
            return refreshed

        with patch.object(
            downloader,
            "_get_at_home_server",
            side_effect=fake_get_server,
        ) as get_server:
            await asyncio.gather(*(
                source.refresh_if_current(None, None, 0)
                for _ in range(3)
            ))

        self.assertEqual(get_server.await_count, 1)
        self.assertEqual(source.revision, 1)
        self.assertEqual(
            source.image_url(1)[0],
            "https://new.example/data/new-hash/page.jpg",
        )

    async def test_failed_image_refreshes_server_then_retries(self):
        initial = {
            "baseUrl": "https://old.example",
            "chapter": {
                "hash": "old-hash",
                "data": ["page.jpg"],
                "chapter": "1",
            },
        }
        refreshed = {
            "baseUrl": "https://new.example",
            "chapter": {
                "hash": "new-hash",
                "data": ["page.jpg"],
                "chapter": "1",
            },
        }
        source = downloader._ChapterSource("chapter-id", initial)
        requested_urls = []

        class FakeClient:
            async def get(self, url, **kwargs):
                requested_urls.append(url)
                request = httpx.Request("GET", url)
                if url.startswith("https://old.example"):
                    return httpx.Response(500, request=request)
                if url.startswith("https://api.mangadex.org"):
                    return httpx.Response(200, request=request, json=refreshed)
                return httpx.Response(200, request=request, content=b"page")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader.asyncio,
            "sleep",
            return_value=None,
        ):
            image_path = os.path.join(directory, "page.jpg")
            result = await downloader._download_image(
                FakeClient(),
                asyncio.Semaphore(1),
                downloader._SharedCooldown(),
                source,
                1,
                image_path,
                downloader._IntervalLimiter(0),
            )

            self.assertTrue(result)
            with open(image_path, "rb") as image:
                self.assertEqual(image.read(), b"page")

        self.assertEqual(
            requested_urls,
            [
                "https://old.example/data/old-hash/page.jpg",
                "https://api.mangadex.org/at-home/server/chapter-id",
                "https://new.example/data/new-hash/page.jpg",
            ],
        )


if __name__ == "__main__":
    unittest.main()
