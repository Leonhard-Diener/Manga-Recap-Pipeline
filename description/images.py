"""Temporary inference-only image preparation; source artifacts stay untouched."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter


def _intersection_ratio(box, protected) -> float:
    area = max(1, (protected[2] - protected[0]) * (protected[3] - protected[1]))
    overlap = max(0, min(box[2], protected[2]) - max(box[0], protected[0])) * max(
        0, min(box[3], protected[3]) - max(box[1], protected[1])
    )
    return overlap / area


def mask_text_regions(image_path: Path, regions: list, destination: Path,
                      protected_boxes=None) -> Path:
    """Blur and cover detected text so the VLM cannot copy it as story evidence."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    protected_boxes = [
        row.get("box") if isinstance(row, dict) else row
        for row in (protected_boxes or [])
    ]
    for raw in regions:
        box = raw.get("box") if isinstance(raw, dict) else raw
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        padding = max(2, round(min(max(1, x2 - x1), max(1, y2 - y1)) * .06))
        bounds = (
            max(0, x1 - padding), max(0, y1 - padding),
            min(width, x2 + padding), min(height, y2 + padding),
        )
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        if any(
            isinstance(protected, (list, tuple)) and len(protected) == 4
            and _intersection_ratio(bounds, protected) > .20
            for protected in protected_boxes
        ):
            continue
        blurred = image.crop(bounds).filter(
            ImageFilter.GaussianBlur(radius=max(10, padding * 3))
        )
        image.paste(blurred, bounds)
    image.save(destination, quality=95)
    return destination


def crop_panel(image_path: Path, box: list, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        bounds = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        image.crop(bounds).save(destination, quality=95)
    return destination
