"""Geometry helpers for detection post-processing."""

from __future__ import annotations

import numpy as np


def box_area(box: np.ndarray) -> float:
    """Return the area of an xyxy bounding box."""
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    """Return the intersection area of two xyxy boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    intersection = intersection_area(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Return the fraction of ``inner`` covered by ``outer``."""
    area = box_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip an xyxy box to image bounds."""
    return np.array(
        [
            max(0.0, min(float(width), float(box[0]))),
            max(0.0, min(float(height), float(box[1]))),
            max(0.0, min(float(width), float(box[2]))),
            max(0.0, min(float(height), float(box[3]))),
        ],
        dtype=np.float32,
    )
