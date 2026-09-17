"""Save an annotated page image with numbered panel borders in manga reading order.

The output is written to a ``page_previews/`` sub-directory inside the given
output directory so it never overwrites panel crops or debug images.

Reading order
-------------
Manga pages are read right-to-left, top-to-bottom.  Panels are grouped into
rows using a 30% vertical-overlap threshold (applied to either panel's height),
with transitive grouping via union-find.  Rows are then sorted top-to-bottom
and panels within each row right-to-left by x1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── constants ────────────────────────────────────────────────────────────────

# Border drawn around every panel.
_BORDER_COLOR = (0, 255, 0)         # lime — matches the existing debug colour
_BORDER_WIDTH = 4

# Label badge colours.
_LABEL_BG = (148, 0, 211)           # vivid purple — visible on any manga page
_LABEL_FG = (255, 255, 255)         # white text
_LABEL_PAD = 10                     # pixels of padding inside the badge
_FONT_SIZE = 36                     # points for the panel number


# ── reading-order sort ────────────────────────────────────────────────────────


def _manga_reading_order(boxes: list[np.ndarray]) -> list[int]:
    """Return panel indices in manga reading order.

    Manga is read from right to left and then from top to bottom.

    Panels are grouped into rows by transitive vertical-overlap: two
    panels belong to the same row if their vertical extents overlap by
    at least 30% of the shorter panel's height, and row membership is
    transitive (union-find), so a chain of panels connects into one row
    even if no single pair perfectly aligns. This also correctly
    handles a tall panel that spans several stacked panels, e.g.:

        +-------------+-------------+
        |      2      |      1      |
        |             |             |
        +-------------+             |
        |      3      |             |
        |             |             |
        +-------------+-------------+

    which is one row (1, 2 and 3 all overlap panel 1's height), read as
    1 -> 2 -> 3: right to left by each panel's left edge, and top to
    bottom as a tie-break for panels stacked in the same column.

    Rows themselves are then sorted top-to-bottom by their topmost edge.

    Note: an earlier version of this function used a step-by-step
    nearest-neighbour walk instead of this row grouping. That approach
    re-evaluated "is this the same row?" relative to whichever panel it
    had *just* visited rather than to the row as a whole. If that panel
    happened to be short (e.g. a thin strip panel), the row-overlap
    test — 30% of the *shorter* panel's height — became very strict for
    its neighbours, so a tall panel one column over could fail to
    register as "same row" and be misclassified as a lower row. Once
    misclassified, it competed for a "closest row" slot against
    unrelated panels and could keep losing, ending up read far later
    than it should have been — occasionally dead last. Grouping rows
    transitively up front avoids this entirely: row membership no
    longer depends on which panel happened to be visited right before it.
    """

    if not boxes:
        return []

    panels = []

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box[:4])

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        panels.append(
            {
                "index": index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "h": max(1.0, y2 - y1),
            }
        )

    n = len(panels)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def same_vertical_band(a: dict, b: dict) -> bool:
        """Check whether two panels occupy approximately the same vertical band."""

        overlap = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))

        if overlap <= 0:
            return False

        return overlap / min(a["h"], b["h"]) >= 0.30

    # Union every pair of panels that share a vertical band. Transitive
    # via union-find, so a chain of overlapping panels ends up in one
    # row even without every pair overlapping each other directly.
    for i in range(n):
        for j in range(i + 1, n):
            if same_vertical_band(panels[i], panels[j]):
                union(i, j)

    rows: dict[int, list[dict]] = {}
    for i, panel in enumerate(panels):
        rows.setdefault(find(i), []).append(panel)

    ordered_rows = sorted(rows.values(), key=lambda row: min(p["y1"] for p in row))

    result: list[int] = []

    for row in ordered_rows:
        if not row:
            continue

        # Group panels that occupy approximately the same horizontal column.
        # Small x-offsets are common in AI detections and should not change
        # the vertical reading order of stacked panels.
        widths = [p["x2"] - p["x1"] for p in row]

        column_tolerance = max(
            10.0,
            np.median(widths) * 0.10,
        )

        columns: list[list[dict]] = []

        # Build loose vertical columns from left-edge positions.
        for panel in sorted(row, key=lambda p: p["x1"]):
            placed = False

            for column in columns:
                reference_x = np.mean([p["x1"] for p in column])

                if abs(panel["x1"] - reference_x) <= column_tolerance:
                    column.append(panel)
                    placed = True
                    break

            if not placed:
                columns.append([panel])

        # Manga reads columns from right to left.
        columns.sort(
            key=lambda column: max(p["x2"] for p in column),
            reverse=True,
        )

        for column in columns:
            # Panels stacked in the same column are read top-to-bottom.
            column.sort(key=lambda p: p["y1"])

            result.extend(
                p["index"]
                for p in column
            )

    return result





# ── font helper ───────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a truetype font; fall back to the built-in bitmap font."""
    candidates = [
        "arialbd.ttf",
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


# ── badge helper ──────────────────────────────────────────────────────────────

def _draw_badge(
    draw: ImageDraw.ImageDraw,
    label: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    preferred_x: int,
    preferred_y: int,
    img_w: int,
    img_h: int,
) -> None:
    """Draw a filled badge with *label* text, guaranteed fully inside the image.

    The badge is placed so its top-left corner is at (*preferred_x*, *preferred_y*).
    If it would overflow the right or bottom edge it is shifted left / up just
    enough to fit.  It is never clipped.
    """
    # Measure text.
    try:
        lb = font.getbbox(label)            # (left, top, right, bottom)
        text_w = lb[2] - lb[0]
        text_h = lb[3] - lb[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)  # type: ignore[attr-defined]

    badge_w = text_w + _LABEL_PAD * 2
    badge_h = text_h + _LABEL_PAD * 2

    # Start at the preferred position.
    bx1 = preferred_x
    by1 = preferred_y

    # Slide left / up if the badge would overflow the image edge.
    if bx1 + badge_w > img_w:
        bx1 = img_w - badge_w
    if by1 + badge_h > img_h:
        by1 = img_h - badge_h

    # Never go negative (top-left corner of image).
    bx1 = max(0, bx1)
    by1 = max(0, by1)

    bx2 = bx1 + badge_w
    by2 = by1 + badge_h

    # Draw background.
    draw.rectangle((bx1, by1, bx2, by2), fill=_LABEL_BG)

    # Draw text centred in the badge.
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2
    try:
        draw.text((cx, cy), label, fill=_LABEL_FG, font=font, anchor="mm")
    except TypeError:
        # Older Pillow without anchor support.
        draw.text((bx1 + _LABEL_PAD, by1 + _LABEL_PAD), label, fill=_LABEL_FG, font=font)


# ── public API ────────────────────────────────────────────────────────────────

def save_page_preview(
    image: Image.Image,
    final_boxes: list[np.ndarray],
    page_path: Path,
    output_dir: Path,
    *,
    jpeg_quality: int = 92,
) -> Path:
    """Draw numbered panel borders on *image* and save the result.

    The annotated image is saved to ``<output_dir>/page_previews/`` using the
    original page filename so filenames stay consistent with the source pages.

    Parameters
    ----------
    image:
        The original (unannotated) page image.
    final_boxes:
        All final panel boxes for this page as xyxy numpy arrays, in any order.
        They are re-sorted into manga reading order internally.
    page_path:
        Path to the source page file — used only to derive the output filename.
    output_dir:
        Root output directory (``page_previews/`` is created inside it).
    jpeg_quality:
        JPEG quality for the saved preview.

    Returns
    -------
    Path to the saved preview image.
    """
    preview_dir = output_dir / "page_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(_FONT_SIZE)
    img_w, img_h = annotated.size

    # Sort boxes into manga reading order.
    order = _manga_reading_order(final_boxes)

    for reading_number, box_index in enumerate(order, start=1):
        box = final_boxes[box_index]
        x1 = int(round(float(box[0])))
        y1 = int(round(float(box[1])))
        x2 = int(round(float(box[2])))
        y2 = int(round(float(box[3])))

        # Draw the panel border.
        draw.rectangle((x1, y1, x2, y2), outline=_BORDER_COLOR, width=_BORDER_WIDTH)

        # Place badge inside the panel, just below-right of the top-left corner.
        inset = _BORDER_WIDTH + 4
        px = x1 + inset
        py = y1 + inset

        _draw_badge(draw, str(reading_number), font, px, py, img_w, img_h)

    out_path = preview_dir / page_path.name
    annotated.save(out_path, format="JPEG", quality=jpeg_quality)
    return out_path