"""Detect panels and beings, preserve source pages, and export hashed manifests.

Usage: python test_panels.py ["Manga title"]
"""
import sys

import config
from common.utils import find_manga_dir, find_chapter_dirs
from panel_extractor.runner import extract_chapter


def main():
    title = sys.argv[1] if len(sys.argv) > 1 else (config.MANGA_TITLE or None)
    manga = find_manga_dir(config.INPUT_DIR, title)
    chapters = find_chapter_dirs(manga)
    for chapter in chapters:
        panels, count = extract_chapter(chapter, config.OUTPUT_DIR / manga.name / chapter.name,
                                       config.MODEL_PATH, **config.panel_extraction_kwargs())
        print(f"{chapter.name}: {len(panels)} panels, {count} body proposals.")


if __name__ == "__main__":
    main()
