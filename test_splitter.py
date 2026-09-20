"""Standalone test for the merged-panel splitter (``panel_extractor/splitter.py``).

The frame-line path needs no model, so this test runs offline against synthetic
manga pages.  The AI fallback is disabled because it requires the ONNX model.

Usage
-----
    python test_splitter.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from panel_extractor.splitter import (
    SplitSettings,
    _max_internal_line_ratio,
    split_merged_panels,
)

PAGE_W = 600
PAGE_H = 800
FRAME_WIDTH = 3

# Frame-line only; no model / no inference.
SETTINGS = SplitSettings(
    enable_frame_split=True,
    enable_ai_fallback=False,
)


def _blank_page(size: tuple[int, int] = (PAGE_W, PAGE_H)) -> Image.Image:
    return Image.new("RGB", size, "white")


def _draw_frames(image: Image.Image, cells: list[tuple[int, int, int, int]]) -> None:
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in cells:
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline="black", width=FRAME_WIDTH)


def _split(image: Image.Image, box: tuple[int, int, int, int]):
    from panel_extractor.types import Detection

    detection = Detection(np.array(box, dtype=np.float32), 0.9, 2)
    return split_merged_panels(
        image,
        [detection],
        session=None,
        image_size=1280,
        panel_confidence=0.35,
        text_confidence=0.20,
        body_confidence=0.20,
        settings=SETTINGS,
    )


def test_stacked_panels() -> bool:
    """Two vertically stacked touching panels must split into two."""
    image = _blank_page()
    _draw_frames(image, [(50, 50, 550, 400), (50, 400, 550, 750)])

    panels, extra = _split(image, (50, 50, 550, 750))
    ok = len(panels) == 2 and extra == 1
    print(f"stacked panels       : {len(panels)} boxes (expected 2) -> {'OK' if ok else 'FAIL'}")

    if ok:
        ys = sorted(round(float(p.box[1])) for p in panels)
        for panel in panels:
            print(f"    box={[round(float(v)) for v in panel.box]}")
        del ys
    return ok


def test_side_by_side_panels() -> bool:
    """Two side-by-side touching panels must split into two."""
    image = _blank_page()
    _draw_frames(image, [(50, 50, 300, 750), (300, 50, 550, 750)])

    panels, extra = _split(image, (50, 50, 550, 750))
    ok = len(panels) == 2 and extra == 1
    print(f"side-by-side panels  : {len(panels)} boxes (expected 2) -> {'OK' if ok else 'FAIL'}")
    for panel in panels:
        print(f"    box={[round(float(v)) for v in panel.box]}")
    return ok


def test_tall_stacked_panels() -> bool:
    """Stacked panels in a tall box (touching top/bottom edges) must split.

    Regression test: the line search once compared candidate row indices
    against the box width instead of its height, so any shared border far down
    a tall box was wrongly discarded.
    """
    image = _blank_page((400, 1000))
    _draw_frames(image, [(50, 50, 350, 500), (50, 500, 350, 950)])

    panels, extra = _split(image, (50, 50, 350, 950))
    ok = len(panels) == 2 and extra == 1
    print(f"tall stacked panels  : {len(panels)} boxes (expected 2) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_wide_side_by_side_panels() -> bool:
    """Side-by-side panels in a wide box must split (same regression)."""
    image = _blank_page((1000, 400))
    _draw_frames(image, [(50, 50, 500, 350), (500, 50, 950, 350)])

    panels, extra = _split(image, (50, 50, 950, 350))
    ok = len(panels) == 2 and extra == 1
    print(f"wide side-by-side    : {len(panels)} boxes (expected 2) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_grid_recursion() -> bool:
    """A 2x2 grid of touching panels must split recursively into four."""
    image = _blank_page()
    _draw_frames(
        image,
        [
            (50, 50, 300, 400),
            (300, 50, 550, 400),
            (50, 400, 300, 750),
            (300, 400, 550, 750),
        ],
    )

    panels, extra = _split(image, (50, 50, 550, 750))
    ok = len(panels) == 4 and extra == 3
    print(f"2x2 grid             : {len(panels)} boxes (expected 4) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_thin_strip_panel_at_border() -> bool:
    """A thin strip panel merged with a tall neighbour must still split.

    Regression test: the interior was once trimmed by a 6% margin, which hid a
    shared border that sits close to the edge of the merged box -- exactly what
    happens when a short panel touches the page border above a tall panel.
    """
    image = _blank_page()
    _draw_frames(image, [(50, 50, 550, 90), (50, 90, 550, 750)])

    panels, extra = _split(image, (50, 50, 550, 750))
    ok = len(panels) == 2 and extra == 1
    print(f"thin border strip    : {len(panels)} boxes (expected 2) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_single_panel_with_short_art_line() -> bool:
    """A single panel with a short interior stroke must not be split."""
    image = _blank_page()
    _draw_frames(image, [(50, 50, 550, 750)])

    draw = ImageDraw.Draw(image)
    # Deliberately short horizontal stroke: not a full-span frame line.
    draw.line((200, 400, 400, 400), fill="black", width=3)

    panels, extra = _split(image, (50, 50, 550, 750))
    ok = len(panels) == 1 and extra == 0
    print(f"short art line       : {len(panels)} boxes (expected 1) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_single_panel_with_speech_box() -> bool:
    """A wide rectangular speech/text box must not become three panels.

    Its top and bottom strokes cover more than 70% of the panel interior, so
    this reproduces the false-positive that a simple line-coverage rule made.
    Unlike a true separator, those strokes stop before the parent frame edges.
    """
    image = _blank_page()
    _draw_frames(image, [(50, 50, 550, 750)])

    draw = ImageDraw.Draw(image)
    draw.rectangle((90, 250, 510, 460), outline="black", width=FRAME_WIDTH)

    panels, extra = _split(image, (50, 50, 550, 750))
    ok = len(panels) == 1 and extra == 0
    print(f"speech/text box      : {len(panels)} boxes (expected 1) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_hidden_line_triggers_ai() -> bool:
    """A frame line broken by breakout art must not split, but must flag AI.

    The AI fallback itself needs the ONNX model, so this test only checks the
    evidence that would trigger it.
    """
    image = _blank_page()
    _draw_frames(image, [(50, 50, 550, 400), (50, 400, 550, 750)])

    # A mid-grey blob covers the shared border, as a breaking-out character
    # would. Grey 128 is above dark_threshold, so it is not counted as ink.
    draw = ImageDraw.Draw(image)
    draw.rectangle((250, 390, 350, 410), fill=(128, 128, 128))

    panels, extra = _split(image, (50, 50, 550, 750))
    geometric_ok = len(panels) == 1 and extra == 0

    box = np.array([50, 50, 550, 750], dtype=np.float32)
    ink = np.asarray(image.convert("L"), dtype=np.uint8) < SETTINGS.dark_threshold
    evidence = _max_internal_line_ratio(ink, box, SplitSettings(enable_ai_fallback=True))
    trigger_ok = evidence >= SplitSettings().ai_min_line_ratio

    ok = geometric_ok and trigger_ok
    print(
        f"hidden frame line    : boxes={len(panels)} (expected 1), "
        f"evidence={evidence:.2f} (>= {SplitSettings().ai_min_line_ratio}) "
        f"-> {'OK' if ok else 'FAIL'}"
    )
    return ok


def test_single_panel_plain() -> bool:
    """A plain single panel must be left untouched."""
    image = _blank_page()
    _draw_frames(image, [(50, 50, 550, 750)])

    panels, extra = _split(image, (50, 50, 550, 750))
    ok = len(panels) == 1 and extra == 0
    print(f"plain single panel   : {len(panels)} boxes (expected 1) -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> None:
    results = [
        test_stacked_panels(),
        test_side_by_side_panels(),
        test_tall_stacked_panels(),
        test_wide_side_by_side_panels(),
        test_grid_recursion(),
        test_thin_strip_panel_at_border(),
        test_single_panel_with_short_art_line(),
        test_single_panel_with_speech_box(),
        test_hidden_line_triggers_ai(),
        test_single_panel_plain(),
    ]

    passed = sum(results)
    print(f"\n{passed}/{len(results)} splitter tests passed.")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
