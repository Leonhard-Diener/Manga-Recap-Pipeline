import asyncio
import os
import time

import httpx
import config


MAX_CONCURRENT_DOWNLOADS = 8

# MangaDex allows 40 requests per minute for this endpoint.
AT_HOME_INTERVAL = 60 / 40


async def _get_at_home_server(
    client: httpx.AsyncClient,
    chapter_id: str,
):
    """Retrieve the MangaDex@Home server assigned to a chapter."""
    await asyncio.sleep(AT_HOME_INTERVAL)

    url = f"https://api.mangadex.org/at-home/server/{chapter_id}"

    while True:
        response = await client.get(url)

        if response.status_code == 429:
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

            print(
                f"Rate limit reached. "
                f"Retrying in {wait_time:.1f}s..."
            )

            await asyncio.sleep(wait_time)
            continue

        response.raise_for_status()
        return response.json()


async def _download_image(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    image_url: str,
    image_path: str,
):
    """Download a single page while respecting the concurrency limit."""

    if os.path.exists(image_path):
        return False

    async with semaphore:
        for attempt in range(3):
            try:
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

                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()

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
            ) as error:

                if attempt == 2:
                    print(
                        f"Failed to download "
                        f"{os.path.basename(image_path)}: {error}"
                    )
                    return False

                await asyncio.sleep(2 ** attempt)

    return False


def _write_file(path: str, content: bytes):
    """Write downloaded data without blocking the event loop."""

    with open(path, "wb") as file:
        file.write(content)


async def download_chapter(
    chapter_id: str,
    manga_name: str,
    chapter_index: int,
):
    """
    Download all pages of one chapter concurrently.

    Each chapter is stored in its own directory:
        input/<manga_name>/chapter_001/
        input/<manga_name>/chapter_002/
        ...
    """

    # --------------------------------------------------------
    # Create manga and chapter directories
    # --------------------------------------------------------

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

    limits = httpx.Limits(
        max_connections=MAX_CONCURRENT_DOWNLOADS,
        max_keepalive_connections=MAX_CONCURRENT_DOWNLOADS,
    )

    async with httpx.AsyncClient(
        limits=limits,
        headers={
            "User-Agent": "MangaDownloader/1.0",
        },
    ) as client:

        # ----------------------------------------------------
        # Get MangaDex@Home server
        # ----------------------------------------------------

        data = await _get_at_home_server(
            client,
            chapter_id,
        )

        base_url = data["baseUrl"]

        chapter = data["chapter"]
        chapter_hash = chapter["hash"]
        page_filenames = chapter["data"]

        chapter_number = chapter.get("chapter")

        # ----------------------------------------------------
        # Create download tasks
        # ----------------------------------------------------

        semaphore = asyncio.Semaphore(
            MAX_CONCURRENT_DOWNLOADS
        )

        tasks = []

        for page_index, filename in enumerate(
            page_filenames,
            start=1,
        ):
            image_url = (
                f"{base_url}/data/"
                f"{chapter_hash}/"
                f"{filename}"
            )

            extension = os.path.splitext(filename)[1]

            image_path = os.path.join(
                chapter_folder,
                f"{page_index:05d}{extension}",
            )

            tasks.append(
                _download_image(
                    client,
                    semaphore,
                    image_url,
                    image_path,
                )
            )

        # ----------------------------------------------------
        # Download all pages
        # ----------------------------------------------------

        results = await asyncio.gather(*tasks)

        downloaded = sum(results)
        skipped = len(results) - downloaded

        print(
            f"Chapter {chapter_number}: "
            f"{downloaded} downloaded, "
            f"{skipped} skipped"
        )

        return len(page_filenames)