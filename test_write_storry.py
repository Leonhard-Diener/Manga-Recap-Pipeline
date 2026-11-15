"""Write or resume the grounded manga recap story.

Usage:
    python test_write_storry.py ["Manga title"]
    python test_write_storry.py ["Manga title"] --restart

Run test_panels.py, test_description.py, and test_character_reid.py first.
"""

from __future__ import annotations

import argparse

import config
from common.utils import find_chapter_dirs, find_manga_dir
from story_writer import run_story_writer


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default=config.MANGA_TITLE or None)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Archive compatible saved progress and generate the story again.",
    )
    args = parser.parse_args(argv)

    manga_dir = find_manga_dir(config.INPUT_DIR, args.title)
    chapters = find_chapter_dirs(manga_dir)
    print(f"Manga: {manga_dir.name}  |  Chapters: {len(chapters)}")
    story = run_story_writer(
        manga_dir,
        chapters,
        config.OUTPUT_DIR,
        restart=args.restart,
    )

    segments = story.get("segments", [])
    words = sum(int(segment.get("word_count", 0)) for segment in segments)
    print(f"Story complete: {len(segments)} panels, {words} words")
    print(f"Narration: {config.OUTPUT_DIR / f'{manga_dir.name}_story.txt'}")
    print(f"Story JSON: {config.OUTPUT_DIR / f'{manga_dir.name}_story.json'}")
    print(f"Timeline: {config.OUTPUT_DIR / f'{manga_dir.name}_panel_timeline.json'}")
    return story


if __name__ == "__main__":
    main()
