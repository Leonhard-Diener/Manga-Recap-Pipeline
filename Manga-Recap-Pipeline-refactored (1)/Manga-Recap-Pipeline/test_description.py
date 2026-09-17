"""Standalone test for step 4: describe what happens on each page.

Requires step 3 (test_panels.py) to have already written
output/<manga>/<chapter>/page_previews/.

Usage
-----
    python test_description.py ["Manga title"]
"""

from __future__ import annotations

import sys

import config
from common.utils import find_chapter_dirs, find_manga_dir
from description.describer import describe_manga


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else (config.MANGA_TITLE or None)

    manga_dir = find_manga_dir(config.INPUT_DIR, title)
    chapter_dirs = find_chapter_dirs(manga_dir)

    manga_output_dir = config.OUTPUT_DIR / manga_dir.name
    output_file = config.OUTPUT_DIR / f"{manga_dir.name}_descriptions.json"

    print(f"Manga: {manga_dir.name}  |  Chapters: {len(chapter_dirs)}\n")
    describe_manga(manga_output_dir, chapter_dirs, output_file)


if __name__ == "__main__":
    main()
