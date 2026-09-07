"""Conservative panel post-processing for the RT-DETR detector."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image

from .geometry import box_area, containment, intersection_area, iou
from .types import Detection


def deduplicate(
    detections: Iterable[Detection],
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> tuple[list[Detection], int]:
    """Remove duplicate detections while keeping the highest-confidence box."""
    ordered = sorted(detections, key=lambda detection: detection.score, reverse=True)
    kept: list[Detection] = []
    removed = 0

    for candidate in ordered:
        duplicate = any(
            iou(candidate.box, existing.box) >= iou_threshold
            or containment(candidate.box, existing.box) >= containment_threshold
            for existing in kept
        )
        if duplicate:
            removed += 1
        else:
            kept.append(candidate)

    return kept, removed


def estimate_coverage(detections: list[Detection], width: int, height: int) -> float:
    """Estimate union coverage of boxes without creating a full image mask."""
    if not detections:
        return 0.0

    xs = sorted({0, width, *[int(d.box[0]) for d in detections], *[int(d.box[2]) for d in detections]})
    ys = sorted({0, height, *[int(d.box[1]) for d in detections], *[int(d.box[3]) for d in detections]})

    covered = 0.0
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            if x1 <= x0 or y1 <= y0:
                continue
            center_x = (x0 + x1) * 0.5
            center_y = (y0 + y1) * 0.5
            if any(
                d.box[0] <= center_x <= d.box[2]
                and d.box[1] <= center_y <= d.box[3]
                for d in detections
            ):
                covered += (x1 - x0) * (y1 - y0)

    return covered / max(1, width * height)


def filter_small_panels(
    detections: Iterable[Detection],
    *,
    page_area: int,
    minimum_ratio: float,
) -> tuple[list[Detection], int]:
    """Remove panel detections smaller than the configured page-area ratio."""
    kept: list[Detection] = []
    removed = 0
    for detection in detections:
        if box_area(detection.box) / max(1, page_area) < minimum_ratio:
            removed += 1
        else:
            kept.append(detection)
    return kept, removed


def expand_box(box: np.ndarray, pad_x: float, pad_y: float, width: int, height: int) -> np.ndarray:
    """Expand an xyxy box by proportional padding and clamp it to the image."""
    x1, y1, x2, y2 = map(float, box)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    return np.array(
        [
            max(0.0, x1 - box_w * pad_x),
            max(0.0, y1 - box_h * pad_y),
            min(float(width), x2 + box_w * pad_x),
            min(float(height), y2 + box_h * pad_y),
        ],
        dtype=np.float32,
    )


def content_intersects(panel: np.ndarray, content: np.ndarray) -> bool:
    """Return whether a content box overlaps a panel box."""
    return box_area(panel) > 0 and box_area(content) > 0 and intersection_area(panel, content) > 0


def protect_content(
    panel: np.ndarray,
    *,
    texts: list[Detection],
    bodies: list[Detection],
    image_size: tuple[int, int],
    text_padding: tuple[float, float],
    text_safety: int,
    body_padding: tuple[float, float],
    body_safety: int,
    max_expand_ratio: float,
) -> np.ndarray:
    """Expand a panel only where detected text/body boxes cross its boundary.

    Content that is already fully inside the RT-DETR frame does not change the
    crop. This keeps strong frame detections intact while protecting details
    that the frame detector boxed too tightly.
    """
    width, height = image_size
    original = panel.copy()
    result = panel.copy()

    panel_w = max(1.0, original[2] - original[0])
    panel_h = max(1.0, original[3] - original[1])
    max_dx = panel_w * max_expand_ratio
    max_dy = panel_h * max_expand_ratio

    def apply(content: np.ndarray, padding: tuple[float, float], safety: int) -> None:
        nonlocal result
        if not content_intersects(original, content):
            return

        left_out = max(0.0, original[0] - content[0])
        top_out = max(0.0, original[1] - content[1])
        right_out = max(0.0, content[2] - original[2])
        bottom_out = max(0.0, content[3] - original[3])

        if left_out == top_out == right_out == bottom_out == 0.0:
            return

        content_w = max(1.0, content[2] - content[0])
        content_h = max(1.0, content[3] - content[1])
        pad_x = content_w * padding[0] + safety
        pad_y = content_h * padding[1] + safety

        if left_out > 0:
            result[0] = max(original[0] - max_dx, min(result[0], original[0] - left_out - pad_x))
        if top_out > 0:
            result[1] = max(original[1] - max_dy, min(result[1], original[1] - top_out - pad_y))
        if right_out > 0:
            result[2] = min(original[2] + max_dx, max(result[2], original[2] + right_out + pad_x))
        if bottom_out > 0:
            result[3] = min(original[3] + max_dy, max(result[3], original[3] + bottom_out + pad_y))

    for text in texts:
        apply(text.box, text_padding, text_safety)
    for body in bodies:
        apply(body.box, body_padding, body_safety)

    return result


def final_nested_cleanup(
    pairs: list[tuple[np.ndarray, Detection]],
    *,
    containment_threshold: float,
) -> tuple[list[tuple[np.ndarray, Detection]], int]:
    """Remove final crops that are almost completely contained by another crop."""
    ordered = sorted(pairs, key=lambda pair: pair[1].score, reverse=True)
    kept: list[tuple[np.ndarray, Detection]] = []
    removed = 0

    for candidate_box, candidate_source in ordered:
        if any(
            containment(candidate_box, kept_box) >= containment_threshold
            for kept_box, _ in kept
        ):
            removed += 1
        else:
            kept.append((candidate_box, candidate_source))

    return kept, removed


def debug_image(
    image: Image.Image,
    *,
    raw_panels: list[Detection],
    texts: list[Detection],
    bodies: list[Detection],
    final_boxes: list[np.ndarray],
) -> Image.Image:
    """Create a debug overlay: orange=raw panel, magenta=text, cyan=body, lime=final."""
    debug = image.copy()
    from PIL import ImageDraw

    draw = ImageDraw.Draw(debug)
    for detection in raw_panels:
        draw.rectangle(tuple(map(int, np.round(detection.box))), outline="orange", width=3)
    for detection in texts:
        draw.rectangle(tuple(map(int, np.round(detection.box))), outline="magenta", width=2)
    for detection in bodies:
        draw.rectangle(tuple(map(int, np.round(detection.box))), outline="cyan", width=2)
    for box in final_boxes:
        draw.rectangle(tuple(map(int, np.round(box))), outline="lime", width=4)
    return debug
