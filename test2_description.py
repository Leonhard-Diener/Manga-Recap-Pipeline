"""Standalone test for step 4: describe what happens on each page.

Requires step 3 (test_panels.py) to have already written
output/<manga>/<chapter>/page_previews/.

Writes output/<manga>_descriptions.json and output/<manga>_transformations.json.
Both datasets are returned by describe_manga; JSON export belongs to this runner.

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
    transformations_file = config.OUTPUT_DIR / f"{manga_dir.name}_transformations.json"

    print(f"Manga: {manga_dir.name}  |  Chapters: {len(chapter_dirs)}\n")
    result = describe_manga(manga_output_dir, chapter_dirs)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(output_file, result["descriptions"])
    write_json(transformations_file, result["transformations"])
    print(f"\nSaved descriptions to {output_file}")
    print(f"Saved proposed character events to {transformations_file}")
    return result


if __name__ == "__main__":
    main()
