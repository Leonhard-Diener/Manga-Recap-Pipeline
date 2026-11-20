import asyncio
import os
import time
import math
import tempfile

import httpx
import config
from common.progress import show_progress


MAX_CONCURRENT_DOWNLOADS = 8
MAX_CONCURRENT_CHAPTERS = 4

# MangaDex allows 40 requests per minute for this endpoint.
# Use about 34/minute to leave a modest buffer below the 40/minute limit.
AT_HOME_INTERVAL = 1.75


class _IntervalLimiter:
    """Space request starts globally, including retries."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._next_request = 0.0

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            delay = self._next_request - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_request = loop.time() + self.interval


class _SharedCooldown:
    """Pause all image workers after any one of them is rate-limited."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._resume_at = 0.0

    async def wait(self) -> None:
        while True:
            async with self._lock:
                delay = self._resume_at - asyncio.get_running_loop().time()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    async def defer(self, seconds: float) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            self._resume_at = max(self._resume_at, loop.time() + seconds)


async def _get_at_home_server(
    client: httpx.AsyncClient,
    chapter_id: str,
    limiter: _IntervalLimiter,
):
    """Retrieve the MangaDex@Home server assigned to a chapter."""
    url = f"https://api.mangadex.org/at-home/server/{chapter_id}"

    # Keep the entire response/retry cycle serialized. If this request receives
    # 429, no other chapter can slip another API call into its cooldown window.
    async with limiter._request_lock:
        for attempt in range(5):
            await limiter.wait()
            response = await client.get(url, timeout=30)

            if response.status_code == 429:
                if attempt == 4:
                    response.raise_for_status()
                retry_after = response.headers.get(
                    "X-RateLimit-Retry-After"
                )

                if retry_after:
                    try:
                        wait_time = max(
                            float(retry_after) - time.time(),
                            1,
                        )
                    except ValueError:
                        wait_time = 60
                else:
                    wait_time = 60
                wait_time = min(wait_time, 300) if math.isfinite(wait_time) else 60

                print(
                    f"Rate limit reached. "
                    f"Retrying in {wait_time:.1f}s..."
                )

                await asyncio.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()


class _ChapterSource:
    """Current, refreshable MangaDex@Home metadata for one chapter."""

    def __init__(self, chapter_id: str, data: dict):
        self.chapter_id = chapter_id
        self._refresh_lock = asyncio.Lock()
        self.revision = 0
        self._apply(data)

    def _apply(self, data: dict) -> None:
        chapter = data["chapter"]
        self.base_url = data["baseUrl"]
        self.chapter_hash = chapter["hash"]
        self.page_filenames = list(chapter["data"])
        self.chapter_number = chapter.get("chapter")

    def image_url(self, page_index: int) -> tuple[str, int]:
        filename = self.page_filenames[page_index - 1]
        return (
            f"{self.base_url}/data/{self.chapter_hash}/{filename}",
            self.revision,
        )

    async def refresh_if_current(
        self,
        client: httpx.AsyncClient,
        limiter: _IntervalLimiter,
        failed_revision: int,
    ) -> None:
        """Refresh once when several workers fail against the same base URL."""
        async with self._refresh_lock:
            if failed_revision != self.revision:
                return
            data = await _get_at_home_server(client, self.chapter_id, limiter)
            self._apply(data)
            self.revision += 1


async def _download_image(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    cooldown: _SharedCooldown,
    source: _ChapterSource,
    page_index: int,
    image_path: str,
    at_home_limiter: _IntervalLimiter,
):
    """Download a single page while respecting the concurrency limit."""

    if os.path.isfile(image_path) and os.path.getsize(image_path) > 0:
        return False

    async with semaphore:
        for attempt in range(3):
            image_url, source_revision = source.image_url(page_index)
            retry_wait = 2 ** attempt
            try:
                await cooldown.wait()
                response = await client.get(
                    image_url,
                    timeout=30,
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")

                    try:
                        wait_time = float(retry_after)
                    except (TypeError, ValueError):
                        wait_time = 5
                    wait_time = min(max(wait_time, 0), 300) if math.isfinite(wait_time) else 5

                    await cooldown.defer(wait_time)
                    retry_wait = 0

                elif response.status_code >= 400:
                    response.raise_for_status()

                else:
                    await asyncio.to_thread(
                        _write_file,
                        image_path,
                        response.content,
                    )

                    return True

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as error:
                if attempt == 2:
                    print(
                        f"Failed to download "
                        f"{os.path.basename(image_path)}: {error}"
                    )
                    return False

            if attempt == 2:
                print(f"Failed to download {os.path.basename(image_path)}: HTTP 429")
                return False

            # MangaDex asks clients to obtain a fresh server assignment after
            # any image failure. Concurrent failures share this single refresh.
            try:
                await source.refresh_if_current(
                    client,
                    at_home_limiter,
                    source_revision,
                )
            except httpx.HTTPError as error:
                print(f"Could not refresh image server: {error}")

            if retry_wait:
                await asyncio.sleep(retry_wait)

    return False


def _write_file(path: str, content: bytes):
    """Write downloaded data without blocking the event loop."""
    if not content:
        raise ValueError(f"Empty downloaded page: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix=".part", delete=False) as file:
            temporary = file.name
            file.write(content)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


async def _download_chapter_with_client(
    chapter_id: str,
    manga_name: str,
    chapter_index: int,
    client: httpx.AsyncClient,
    download_semaphore: asyncio.Semaphore,
    at_home_limiter: _IntervalLimiter,
    image_cooldown: _SharedCooldown,
):
    """Download one chapter through shared connection and rate limits."""
    manga_folder = os.path.join(
        config.INPUT_DIR,
        manga_name,
    )

    chapter_folder = os.path.join(
        manga_folder,
        f"chapter_{chapter_index:03d}",
    )

    os.makedirs(
        chapter_folder,
        exist_ok=True,
    )

    data = await _get_at_home_server(
        client,
        chapter_id,
        at_home_limiter,
    )

    source = _ChapterSource(chapter_id, data)

    tasks = []
    paths = []
    completed = 0
    total = len(source.page_filenames)
    label = f"Download chapter {chapter_index} pages"
    show_progress(label, 0, total)

    async def download_page(page_index, image_path):
        nonlocal completed
        result = await _download_image(
            client, download_semaphore, image_cooldown, source, page_index,
            image_path, at_home_limiter,
        )
        if os.path.isfile(image_path) and os.path.getsize(image_path) > 0:
            completed += 1
            show_progress(label, completed, total)
        return result

    for page_index, filename in enumerate(source.page_filenames, start=1):
        extension = os.path.splitext(filename)[1]
        image_path = os.path.join(
            chapter_folder,
            f"{page_index:05d}{extension}",
        )
        paths.append(image_path)
        tasks.append(download_page(page_index, image_path))

    results = await asyncio.gather(*tasks)
    downloaded = sum(results)
    missing = [path for path in paths if not os.path.isfile(path) or os.path.getsize(path) == 0]
    if missing:
        raise RuntimeError(f"Chapter {chapter_index}: {len(missing)} pages failed to download. Rerun to retry missing pages.")
    skipped = len(results) - downloaded

    print(
        f"Chapter {source.chapter_number}: "
        f"{downloaded} downloaded, "
        f"{skipped} skipped"
    )
    return len(source.page_filenames)


def _client_limits() -> httpx.Limits:
    # Reserve a few connections for paced API requests while all eight image
    # slots are busy. Image concurrency remains controlled by the semaphore.
    connections = MAX_CONCURRENT_DOWNLOADS + MAX_CONCURRENT_CHAPTERS
    return httpx.Limits(
        max_connections=connections,
        max_keepalive_connections=connections,
    )


async def download_chapter(
    chapter_id: str,
    manga_name: str,
    chapter_index: int,
):
    """Download one chapter; retained for existing callers."""
    async with httpx.AsyncClient(
        limits=_client_limits(),
        headers={"User-Agent": "MangaDownloader/1.0"},
    ) as client:
        return await _download_chapter_with_client(
            chapter_id,
            manga_name,
            chapter_index,
            client,
            asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS),
            _IntervalLimiter(AT_HOME_INTERVAL),
            _SharedCooldown(),
        )


async def download_chapters(
    chapter_ids: list[str],
    manga_name: str,
) -> list[int]:
    """Pipeline chapters with global, MangaDex-friendly concurrency limits."""
    if not chapter_ids:
        return []

    download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    chapter_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHAPTERS)
    at_home_limiter = _IntervalLimiter(AT_HOME_INTERVAL)
    image_cooldown = _SharedCooldown()

    async with httpx.AsyncClient(
        limits=_client_limits(),
        headers={"User-Agent": "MangaDownloader/1.0"},
    ) as client:
        async def run_chapter(chapter_index: int, chapter_id: str) -> int:
            async with chapter_semaphore:
                print(f"Starting chapter {chapter_index}/{len(chapter_ids)}")
                return await _download_chapter_with_client(
                    chapter_id,
                    manga_name,
                    chapter_index,
                    client,
                    download_semaphore,
                    at_home_limiter,
                    image_cooldown,
                )

        return await asyncio.gather(*(
            run_chapter(chapter_index, chapter_id)
            for chapter_index, chapter_id in enumerate(chapter_ids, start=1)
        ))
