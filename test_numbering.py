"""Run panel detection on all chapters and save numbered preview images + detections.json.

Usage
-----
    python test_numbering.py

Input structure expected
------------------------
    input/
      <manga title>/
        chapter_001/
          00001.jpg ...
        chapter_002/
          ...

What it does
------------
1. Finds every chapter_* subfolder under input/<manga title>/.
2. Runs the RT-DETR detector on each chapter in order.
3. Sorts the final panel boxes into manga reading order (right-to-left, top-to-bottom).
4. Saves an annotated preview for each page to:
       output/<manga title>/<chapter>/page_previews/<page>.jpg
5. Writes output/detections.json so test_cropping.py can crop without re-running
   the detector.

No panel crops are written by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from panel_extractor.pipeline import detect_panels

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUTPUT_DIR = config.OUTPUT_DIR
DETECTIONS_FILE = OUTPUT_DIR / "detections.json"


def _find_manga_dir() -> Path:
    """Return the single manga title directory inside config.INPUT_DIR."""
    base = config.INPUT_DIR
    if not base.is_dir():
        raise NotADirectoryError(f"Input directory not found: {base}")
    subdirs = [p for p in sorted(base.iterdir()) if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No manga title folder found in {base}")
    if len(subdirs) > 1:
        print(f"Multiple manga folders found, using first: {subdirs[0].name}")
    return subdirs[0]


def _find_chapter_dirs(manga_dir: Path) -> list[Path]:
    """Return all chapter subdirectories sorted by name."""
    chapters = sorted(
        p for p in manga_dir.iterdir()
        if p.is_dir() and any(
            f.suffix.lower() in IMAGE_SUFFIXES for f in p.iterdir()
        )
    )
    if not chapters:
        raise FileNotFoundError(f"No chapter directories with images found in {manga_dir}")
    return chapters


def main() -> None:
    manga_dir = _find_manga_dir()
    chapter_dirs = _find_chapter_dirs(manga_dir)

    print(f"Manga : {manga_dir.name}")
    print(f"Chapters found: {len(chapter_dirs)}")

    all_detections = []
    total_pages = 0
    total_panels = 0

    for chapter_dir in chapter_dirs:
        # Output previews go into output/<manga>/<chapter>/page_previews/
        chapter_output = OUTPUT_DIR / manga_dir.name / chapter_dir.name

        print(f"\n── {chapter_dir.name} ──")
        detections = detect_panels(
            input_dir=chapter_dir,
            model_path=config.MODEL_PATH,
            output_dir=chapter_output,
            image_size=config.IMAGE_SIZE,
            panel_confidence=config.PANEL_CONFIDENCE,
            text_confidence=config.TEXT_CONFIDENCE,
            body_confidence=config.BODY_CONFIDENCE,
            enable_tiled_fallback=config.ENABLE_TILED_FALLBACK,
            fallback_uncovered_ratio=config.FALLBACK_UNCOVERED_RATIO,
            fallback_grid=config.FALLBACK_GRID,
            fallback_overlap=config.FALLBACK_OVERLAP,
            minimum_panel_ratio=config.MINIMUM_PANEL_RATIO,
            duplicate_iou=config.DUPLICATE_IOU,
            duplicate_containment=config.DUPLICATE_CONTAINMENT,
            enable_deduplication=config.ENABLE_DEDUPLICATION,
            enable_small_panel_filter=config.ENABLE_SMALL_PANEL_FILTER,
            enable_content_protection=config.ENABLE_CONTENT_PROTECTION,
            enable_nested_cleanup=config.ENABLE_NESTED_CLEANUP,
            text_padding_x=config.TEXT_PADDING_X,
            text_padding_y=config.TEXT_PADDING_Y,
            text_safety_pixels=config.TEXT_SAFETY_PIXELS,
            body_padding_x=config.BODY_PADDING_X,
            body_padding_y=config.BODY_PADDING_Y,
            body_safety_pixels=config.BODY_SAFETY_PIXELS,
            max_content_expansion_ratio=config.MAX_CONTENT_EXPANSION_RATIO,
            jpeg_quality=config.JPEG_QUALITY,
            save_debug=config.SAVE_DEBUG,
        )

        chapter_panels = sum(len(d.boxes) for d in detections)
        total_pages += len(detections)
        total_panels += chapter_panels
        print(f"  {len(detections)} pages | {chapter_panels} panels")

        all_detections.extend(detections)

    # Serialise to JSON.
    payload = [
        {
            "page_path": str(d.page_path.resolve()),
            "boxes": d.boxes,    # list of [x1, y1, x2, y2]
            "scores": d.scores,  # list of floats, same order
        }
        for d in all_detections
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"\n{'='*50}\n"
        f"Done.  {total_pages} pages | {total_panels} panels detected.\n"
        f"Previews : {OUTPUT_DIR / manga_dir.name} (per chapter subfolder)\n"
        f"JSON     : {DETECTIONS_FILE}"
    )


if __name__ == "__main__":
    main()
