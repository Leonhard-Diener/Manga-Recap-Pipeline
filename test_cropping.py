"""Crop panel images from a detections.json produced by test_numbering.py.

Usage
-----
    python test_cropping.py

What it does
------------
1. Reads ``output/detections.json`` (written by ``test_numbering.py``).
2. Calls ``crop_panels()`` per chapter and saves panel crops to:
       output/<manga title>/<chapter>/panels/panel_XXXX.jpg
   The numbering is global across all chapters so panel_0001 is always the
   first panel of the first chapter.
3. The detector is NOT re-run; positions come entirely from the JSON file.

Run ``test_numbering.py`` first if the JSON file does not exist yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from panel_extractor.pipeline import crop_panels
from panel_extractor.types import PageDetection

OUTPUT_DIR = config.OUTPUT_DIR
DETECTIONS_FILE = OUTPUT_DIR / "detections.json"


def load_detections(path: Path) -> list[PageDetection]:
    """Deserialise a detections.json file into PageDetection objects."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Detections file not found: {path}\n"
            "Run test_numbering.py first to generate it."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        PageDetection(
            page_path=Path(entry["page_path"]),
            boxes=entry["boxes"],
            scores=entry["scores"],
        )
        for entry in raw
    ]


def _chapter_key(page_path: Path) -> Path:
    """Return the chapter directory for a given page path."""
    return page_path.parent


def main() -> None:
    print(f"Loading detections from: {DETECTIONS_FILE}")
    detections = load_detections(DETECTIONS_FILE)

    total_panels = sum(len(d.boxes) for d in detections)
    print(f"{len(detections)} pages | {total_panels} panels to crop.\n")

    # Derive the manga title from the first page path.
    # Expected: .../<manga title>/<chapter>/<page>.jpg
    first_page = detections[0].page_path
    manga_name = first_page.parent.parent.name

    # Group pages by chapter to write crops into chapter subfolders.
    # We keep the global panel index running across all chapters.
    next_index = 1
    current_chapter: Path | None = None
    chapter_detections: list[PageDetection] = []
    all_panels = []

    def _flush_chapter(chap_dir: Path, pages: list[PageDetection], start_index: int) -> tuple[list, int]:
        """Crop one chapter and return (panels, next_index)."""
        crop_output = OUTPUT_DIR / manga_name / chap_dir.name / "panels"
        # crop_panels always starts its own counter at 1 internally, so we
        # need to rename after the fact — simpler to just collect and rename.
        panels = crop_panels(pages, crop_output, jpeg_quality=config.JPEG_QUALITY)
        # Rename files so numbering is globally sequential.
        renamed = []
        for panel in panels:
            new_name = crop_output / f"panel_{start_index:04d}.jpg"
            panel.output_path.rename(new_name)
            from dataclasses import replace  # noqa: PLC0415
            renamed.append(
                type(panel)(
                    index=start_index,
                    source_page=panel.source_page,
                    confidence=panel.confidence,
                    box=panel.box,
                    output_path=new_name,
                )
            )
            start_index += 1
        return renamed, start_index

    for det in detections:
        chap_dir = _chapter_key(det.page_path)
        if current_chapter is None:
            current_chapter = chap_dir
        if chap_dir != current_chapter:
            panels, next_index = _flush_chapter(current_chapter, chapter_detections, next_index)
            all_panels.extend(panels)
            current_chapter = chap_dir
            chapter_detections = []
        chapter_detections.append(det)

    if chapter_detections and current_chapter is not None:
        panels, next_index = _flush_chapter(current_chapter, chapter_detections, next_index)
        all_panels.extend(panels)

    for panel in all_panels:
        print(
            f"Panel {panel.index:04d} | "
            f"Page: {panel.source_page} | "
            f"Confidence: {panel.confidence:.3f} | "
            f"Saved: {panel.output_path}"
        )

    print(f"\nDone.  {len(all_panels)} panels saved under {OUTPUT_DIR / manga_name}")


if __name__ == "__main__":
    main()
