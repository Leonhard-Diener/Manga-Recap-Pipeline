"""Standalone page description and story-page filtering runner.

Requires step 3 (test_panels.py) to have already written
output/<manga>/<chapter>/page_previews/.

Writes one output/<manga>_descriptions.json containing usable story descriptions,
all-page filtering metadata, and transformations. The same combined object is
returned by describe_manga; JSON export belongs to this runner.

Usage
-----
    python test_description.py ["Manga title"]
"""

from __future__ import annotations

import sys

import config
from common.utils import find_chapter_dirs, find_manga_dir
from description.describer import describe_manga
from common.json_io import write_json


def main() -> dict:
    title = sys.argv[1] if len(sys.argv) > 1 else (config.MANGA_TITLE or None)

    manga_dir = find_manga_dir(config.INPUT_DIR, title)
    chapter_dirs = find_chapter_dirs(manga_dir)

    manga_output_dir = config.OUTPUT_DIR / manga_dir.name
    output_file = config.OUTPUT_DIR / f"{manga_dir.name}_descriptions.json"

    print(f"Manga: {manga_dir.name}  |  Chapters: {len(chapter_dirs)}\n")
    result = describe_manga(
        manga_output_dir,
        chapter_dirs,
        filter_non_story=True,
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(output_file, result)
    pages = [page for chapter in result["page_filter"]["chapters"].values()
             for page in chapter.values()]
    included = sum(page["included_in_story"] for page in pages)
    print(f"\nStory filter: {included} included, {len(pages) - included} excluded")
    print(f"\nSaved descriptions and character events to {output_file}")
    return result


if __name__ == "__main__":
    main()
