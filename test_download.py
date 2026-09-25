"""Standalone test for step 1: search MangaDex and download a manga's
chapters into input/<manga title>/chapter_XXX/.

Usage
-----
    python test_download.py ["Manga title"]
"""

from __future__ import annotations

import asyncio
import sys

import config
from downloader.download_chapter import download_chapters
from downloader.get_chapter_ids import get_chapter_ids
from downloader.get_manga_id import get_manga_id


async def download_manga(title: str, all_chapters: bool) -> None:
    manga_id = get_manga_id(title)
    chapter_ids = get_chapter_ids(manga_id)

    print(f"\nFound {len(chapter_ids)} chapters.")
    if not chapter_ids:
        print("No chapters found.")
        return

    chosen = chapter_ids if all_chapters else chapter_ids[:1]
    await download_chapters(chosen, title)

    print("\nDownload complete.")


def main() -> None:
    if len(sys.argv) > 1:
        title = sys.argv[1]
    elif config.MANGA_TITLE:
        title = config.MANGA_TITLE
    else:
        raise SystemExit("Usage: python test_download.py \"Manga title\"")

    asyncio.run(download_manga(title, config.DOWNLOAD_ALL_CHAPTERS))


if __name__ == "__main__":
    main()
