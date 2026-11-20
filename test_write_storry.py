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
from story_writer.display import refresh_panel_timeline


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default=config.MANGA_TITLE or None)
    parser.add_argument("--style", choices=("scene", "panel"), default="scene",
                        help="Scene prose (default) or legacy per-panel narration.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--restart",
        action="store_true",
        help="Archive compatible saved progress and generate the story again.",
    )
    mode.add_argument("--refresh-timeline", action="store_true",
                      help="Refresh aspect-aware panel display groups from saved story output; no model run.")
    args = parser.parse_args(argv)

    manga_dir = find_manga_dir(config.INPUT_DIR, args.title)
    if args.refresh_timeline:
        timeline = refresh_panel_timeline(
            config.OUTPUT_DIR / f"{manga_dir.name}_story.json",
            config.OUTPUT_DIR / f"{manga_dir.name}_panel_timeline.json",
        )
        print(f"Refreshed {len(timeline['display_groups'])} display groups without regenerating narration.")
        return timeline
    chapters = find_chapter_dirs(manga_dir)
    print(f"Manga: {manga_dir.name}  |  Chapters: {len(chapters)}")
    story = run_story_writer(
        manga_dir,
        chapters,
        config.OUTPUT_DIR,
        restart=args.restart,
        narration_style=args.style,
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
