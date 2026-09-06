import asyncio

from downloader.get_manga_id import get_manga_id
from downloader.get_chapter_ids import get_chapter_ids
from downloader.download_chapter import download_chapter


async def main():
    manga_name = "Maou-sama no Machizukuri! ~Saikyou no Dungeon wa Kindai Toshi~"

    manga_id = get_manga_id(manga_name)
    chapter_ids = get_chapter_ids(manga_id)

    print(f"\nFound {len(chapter_ids)} chapters.")

    await download_chapter(
        chapter_ids[0],
        manga_name,
    )


if __name__ == "__main__":
    asyncio.run(main())