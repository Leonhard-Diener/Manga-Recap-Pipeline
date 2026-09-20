"""Offline tests for overlap resolution in ``panel_extractor/postprocess.py``.

These cover the containment cases that used to lose or duplicate real panels:

* A panel detected as a full box plus a contained, higher-scoring fragment.
  The fragment won and was then discarded by the size filter, so the panel
  vanished.
* A detector fragment that sits almost entirely inside its parent frame.
* A full-page box with a fully nested detector fragment.

The tests need no model: detections are constructed directly.

Usage
-----
    python test_postprocess.py
"""

from __future__ import annotations

import numpy as np

from panel_extractor.geometry import box_area
from panel_extractor.postprocess import deduplicate, final_nested_cleanup
from panel_extractor.types import Detection

PAGE_W = 1360
PAGE_H = 1920
PAGE_AREA = PAGE_W * PAGE_H


def _det(box: tuple[int, int, int, int], score: float) -> Detection:
    return Detection(np.array(box, dtype=np.float32), score, 2)


def test_panel_fragments_merge() -> bool:
    """A full box plus a contained higher-scoring fragment must become one box.

    Mirrors a real failure: the detector returned the whole panel at 0.575 and a
    right-hand fragment at 0.722.  The old code kept the fragment (higher score)
    and dropped the full box, then filtered the fragment away as too small.
    """
    full = _det((0, 1592, 490, 1920), 0.575)
    fragment = _det((129, 1593, 491, 1918), 0.722)

    kept, removed = deduplicate(
        [full, fragment],
        iou_threshold=0.80,
        containment_threshold=0.90,
        page_area=PAGE_AREA,
    )

    ok = len(kept) == 1 and removed == 1
    if ok:
        box = kept[0].box
        ok = (
            abs(float(box[0]) - 0) <= 2
            and abs(float(box[1]) - 1592) <= 2
            and abs(float(box[2]) - 491) <= 2
            and abs(float(box[3]) - 1920) <= 2
        )
    print(f"fragments merge      : kept={len(kept)} removed={removed} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_deeply_nested_panels_merge() -> bool:
    """A fragment that is 90% inside another frame becomes one panel."""
    big = _det((0, 420, 1303, 1916), 0.858)
    small = _det((2, 1396, 1031, 1919), 0.876)

    kept, _ = deduplicate(
        [big, small],
        iou_threshold=0.80,
        containment_threshold=0.90,
        page_area=PAGE_AREA,
    )

    ok = len(kept) == 1
    print(f"deep nesting merges  : kept={len(kept)} (expected 1) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_background_box_and_inset_merge() -> bool:
    """A whole-page box and fully nested fragment become one panel."""
    background = _det((0, 0, PAGE_W, PAGE_H), 0.594)
    panel = _det((994, 104, 1185, 435), 0.863)

    kept, _ = deduplicate(
        [background, panel],
        iou_threshold=0.80,
        containment_threshold=0.90,
        page_area=PAGE_AREA,
    )

    ok = len(kept) == 1 and box_area(kept[0].box) >= PAGE_AREA * 0.9
    print(f"page + nested merge  : kept={len(kept)} (expected 1) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_identical_boxes_merge() -> bool:
    """Near-identical duplicates must be merged into a single detection."""
    first = _det((100, 100, 500, 500), 0.9)
    second = _det((102, 101, 501, 499), 0.6)

    kept, removed = deduplicate(
        [first, second],
        iou_threshold=0.80,
        containment_threshold=0.90,
        page_area=PAGE_AREA,
    )

    ok = len(kept) == 1 and removed == 1
    print(f"identical merge      : kept={len(kept)} removed={removed} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_nested_cleanup_merges_deep_nesting() -> bool:
    """The final cleanup also unifies deeply nested fragments."""
    big = _det((0, 420, 1303, 1916), 0.858)
    small = _det((2, 1396, 1031, 1919), 0.876)

    kept, removed = final_nested_cleanup(
        [(big.box.copy(), big), (small.box.copy(), small)],
        containment_threshold=0.90,
        iou_threshold=0.80,
        page_area=PAGE_AREA,
    )

    ok = len(kept) == 1 and removed == 1
    print(f"nested merges deep   : kept={len(kept)} (expected 1) -> {'OK' if ok else 'FAIL'}")
    return ok


def test_nested_cleanup_drops_duplicate() -> bool:
    """A near-identical crop must still be removed by the final cleanup."""
    first = _det((100, 100, 500, 500), 0.9)
    second = _det((101, 100, 501, 501), 0.5)

    kept, removed = final_nested_cleanup(
        [(first.box.copy(), first), (second.box.copy(), second)],
        containment_threshold=0.90,
        iou_threshold=0.80,
        page_area=PAGE_AREA,
    )

    ok = len(kept) == 1 and removed == 1
    print(f"nested drops dup     : kept={len(kept)} removed={removed} -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> None:
    results = [
        test_panel_fragments_merge(),
        test_deeply_nested_panels_merge(),
        test_background_box_and_inset_merge(),
        test_identical_boxes_merge(),
        test_nested_cleanup_merges_deep_nesting(),
        test_nested_cleanup_drops_duplicate(),
    ]

    passed = sum(results)
    print(f"\n{passed}/{len(results)} post-processing tests passed.")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
