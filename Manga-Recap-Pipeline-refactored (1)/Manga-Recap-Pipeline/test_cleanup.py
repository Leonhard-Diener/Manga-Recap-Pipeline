"""Standalone test for step 2: classify pages and delete the extra
(credits/shoutout/intro) pages at the start and end of every chapter.

This permanently deletes files under input/<manga>/<chapter>/. Run
test_download.py first if the chapters aren't downloaded yet.

Usage
-----
    python test_cleanup.py ["Manga title"]

If no title is given, config.MANGA_TITLE is used, or (if that's empty) the
single manga folder already present under input/.
"""

from __future__ import annotations

import sys

import config
from cleanup.classifier import clean_manga
from common.utils import find_chapter_dirs, find_manga_dir


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else (config.MANGA_TITLE or None)

    manga_dir = find_manga_dir(config.INPUT_DIR, title)
    chapter_dirs = find_chapter_dirs(manga_dir)

    print(f"Manga: {manga_dir.name}  |  Chapters: {len(chapter_dirs)}\n")
    clean_manga(chapter_dirs)


if __name__ == "__main__":
    main()
