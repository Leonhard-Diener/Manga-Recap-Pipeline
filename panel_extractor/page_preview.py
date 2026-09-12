"""Save an annotated page image with numbered panel borders in manga reading order.

The output is written to a ``page_previews/`` sub-directory inside the given
output directory so it never overwrites panel crops or debug images.

Reading order
-------------
Manga pages are read right-to-left, top-to-bottom.  Panels are first grouped
into rows by checking whether their vertical centres overlap (with a small
tolerance band), then sorted right-to-left within each row.
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

# Two panels are considered to be in the same row when their vertical overlap
# covers at least this fraction of the smaller panel's height.
_ROW_OVERLAP_RATIO = 0.5


# ── reading-order sort ────────────────────────────────────────────────────────

def _manga_reading_order(boxes: list[np.ndarray]) -> list[int]:
    """Return panel indices sorted in manga reading order (right-to-left, top-to-bottom).

    Row grouping uses vertical overlap rather than centre-distance so that a
    tall panel spanning multiple small panels on the opposite side is not
    incorrectly merged into one of those rows.  Two panels are considered to be
    in the same row when they share at least ``_ROW_OVERLAP_RATIO`` of the
    smaller panel's height as vertical overlap.
    """
    if not boxes:
        return []

    # Work with (original_index, x_centre, y1, y2) tuples.
    items = [
        (i, (b[0] + b[2]) * 0.5, float(b[1]), float(b[3]))
        for i, b in enumerate(boxes)
    ]
    # Process top-to-bottom by the top edge of each panel.
    items.sort(key=lambda c: c[2])

    # Each row stores (original_index, cx, y1, y2) and tracks its own y-span.
    rows: list[list[tuple[int, float, float, float]]] = []
    row_spans: list[tuple[float, float]] = []   # (row_y1, row_y2) for each row

    for idx, cx, y1, y2 in items:
        h = max(1.0, y2 - y1)
        placed = False
        for r_idx, (row_y1, row_y2) in enumerate(row_spans):
            # Vertical overlap between this panel and the current row span.
            overlap = max(0.0, min(y2, row_y2) - max(y1, row_y1))
            row_h = max(1.0, row_y2 - row_y1)
            # Belongs to the row if overlap covers enough of the smaller height.
            if overlap / min(h, row_h) >= _ROW_OVERLAP_RATIO:
                rows[r_idx].append((idx, cx, y1, y2))
                # Expand the row's y-span to include this panel.
                row_spans[r_idx] = (min(row_y1, y1), max(row_y2, y2))
                placed = True
                break
        if not placed:
            rows.append([(idx, cx, y1, y2)])
            row_spans.append((y1, y2))

    result: list[int] = []
    for row in rows:
        # Within each row sort right-to-left by x centre.
        row.sort(key=lambda c: -c[1])
        result.extend(c[0] for c in row)

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
