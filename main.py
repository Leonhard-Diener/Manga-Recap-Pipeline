"""Run the full manga recap pipeline for one manga title, end to end:

    1. Download all (or just the first) chapter(s) from MangaDex.
    2. Remove non-story pages (credits, shoutouts, intros, ...) from the
       start/end of every chapter.
    3. Detect panels, crop them, and save numbered page previews.
    4. Describe what happens on every page, panel by panel.

Each step's model is loaded once and reused across all chapters before
moving on to the next step.

Usage
-----
    python main.py "Manga title"
    python main.py "Manga title" --no-all-chapters
"""

from __future__ import annotations

import argparse
import asyncio

import config
from cleanup.classifier import clean_manga
from common.utils import find_chapter_dirs, find_manga_dir
from description.describer import describe_manga
from downloader.download_chapter import download_chapter
from downloader.get_chapter_ids import get_chapter_ids
from downloader.get_manga_id import get_manga_id
from panel_extractor.pipeline import extract_panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manga recap pipeline")
    parser.add_argument("title", help="Manga title to search for on MangaDex")
    parser.add_argument(
        "--no-all-chapters",
        action="store_true",
        help="Only download the first chapter instead of all chapters",
    )
    return parser.parse_args()


async def _download_manga(title: str, all_chapters: bool) -> None:
    """Step 1: download the manga's chapters."""
    manga_id = get_manga_id(title)
    chapter_ids = get_chapter_ids(manga_id)
    if not chapter_ids:
        raise RuntimeError("No chapters found.")

    chosen = chapter_ids if all_chapters else chapter_ids[:1]
    print(f"\nDownloading {len(chosen)} chapter(s)...")
    for index, chapter_id in enumerate(chosen, start=1):
        print(f"\nChapter {index}/{len(chosen)}")
        await download_chapter(chapter_id, title, index)


def _extract_panels_for_all_chapters(manga_dir, chapter_dirs) -> None:
    """Step 3: detect, crop, and number panels for every chapter."""
    for chapter_dir in chapter_dirs:
        chapter_output = config.OUTPUT_DIR / manga_dir.name / chapter_dir.name
        print(f"\n── {chapter_dir.name} ──")
        panels = extract_panels(
            input_dir=chapter_dir,
            output_dir=chapter_output,
            model_path=config.MODEL_PATH,
            **config.panel_extraction_kwargs(),
        )
        print(f"  {len(panels)} panels extracted.")


def main() -> None:
    args = parse_args()
    all_chapters = not args.no_all_chapters

    print("=" * 60)
    print("STEP 1/4: Download")
    print("=" * 60)
    asyncio.run(_download_manga(args.title, all_chapters))

    manga_dir = find_manga_dir(config.INPUT_DIR, args.title)
    chapter_dirs = find_chapter_dirs(manga_dir)

    print("\n" + "=" * 60)
    print("STEP 2/4: Remove non-story pages")
    print("=" * 60)
    clean_manga(chapter_dirs)

    # Step 2 deletes files in place, so re-read the chapter list in case a
    # chapter ended up with no pages left (nothing to do for it in step 3/4).
    chapter_dirs = find_chapter_dirs(manga_dir)

    print("\n" + "=" * 60)
    print("STEP 3/4: Detect, crop and number panels")
    print("=" * 60)
    _extract_panels_for_all_chapters(manga_dir, chapter_dirs)

    print("\n" + "=" * 60)
    print("STEP 4/4: Describe pages")
    print("=" * 60)
    manga_output_dir = config.OUTPUT_DIR / manga_dir.name
    output_file = config.OUTPUT_DIR / f"{manga_dir.name}_descriptions.json"
    describe_manga(manga_output_dir, chapter_dirs, output_file)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
