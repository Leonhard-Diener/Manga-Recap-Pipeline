"""Standalone test for step 3: panel detection, cropping, and numbered
page previews, combined into a single pass per chapter.

For each chapter this writes two things under output/<manga>/<chapter>/:
    panel_0001.jpg, panel_0002.jpg, ...   the cropped panels
    page_previews/<page>.jpg              original pages with numbered
                                           borders around each panel, in
                                           manga reading order

Usage
-----
    python test_panels.py ["Manga title"]
"""

from __future__ import annotations

import sys

import config
from common.utils import find_chapter_dirs, find_manga_dir
from panel_extractor.pipeline import extract_panels


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else (config.MANGA_TITLE or None)

    manga_dir = find_manga_dir(config.INPUT_DIR, title)
    chapter_dirs = find_chapter_dirs(manga_dir)

    print(f"Manga: {manga_dir.name}  |  Chapters: {len(chapter_dirs)}")

    total_panels = 0
    for chapter_dir in chapter_dirs:
        chapter_output = config.OUTPUT_DIR / manga_dir.name / chapter_dir.name
        print(f"\n── {chapter_dir.name} ──")

        panels = extract_panels(
            input_dir=chapter_dir,
            output_dir=chapter_output,
            model_path=config.MODEL_PATH,
            **config.panel_extraction_kwargs(),
        )

        print(f"  {len(panels)} panels cropped -> {chapter_output}")
        total_panels += len(panels)

    print(f"\nDone. {total_panels} panels extracted across {len(chapter_dirs)} chapters.")


if __name__ == "__main__":
    main()
