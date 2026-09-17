import asyncio

from downloader.get_manga_id import get_manga_id
from downloader.get_chapter_ids import get_chapter_ids
from downloader.download_chapter import download_chapter

import config


async def main():

    manga_name = (
        "Maou-sama no Machizukuri! "
        "~Saikyou no Dungeon wa Kindai Toshi~"
    )

    manga_id = get_manga_id(manga_name)

    chapter_ids = get_chapter_ids(manga_id)

    print(f"\nFound {len(chapter_ids)} chapters.")

    if not chapter_ids:
        print("No chapters found.")
        return

    # Download either all chapters or only the first one.
    chapters_to_download = (
        chapter_ids
        if config.DOWNLOAD_ALL_CHAPTERS
        else chapter_ids[:1]
    )

    for index, chapter_id in enumerate(
        chapters_to_download,
        start=1,
    ):
        print(
            f"\nDownloading chapter "
            f"{index}/{len(chapters_to_download)}"
        )

        await download_chapter(
            chapter_id,
            manga_name,
            index,
        )

    print("\nDownload complete.")


if __name__ == "__main__":
    asyncio.run(main())