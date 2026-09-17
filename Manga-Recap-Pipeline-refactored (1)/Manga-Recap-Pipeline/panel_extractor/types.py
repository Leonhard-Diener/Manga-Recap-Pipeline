"""Data structures shared by the detector and post-processing modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Detection:
    """A single RT-DETR detection in original-image coordinates."""

    box: np.ndarray
    score: float
    class_id: int


@dataclass
class PageDetection:
    """All final panel boxes for one page, sorted in manga reading order.

    This is the intermediate result produced by ``detect_panels()`` and
    consumed by ``crop_panels()``.  The JSON test files serialise and
    deserialise this structure so the two steps can run independently.

    Attributes
    ----------
    page_path:
        Absolute path to the source page image.
    boxes:
        Final panel boxes in manga reading order, each as [x1, y1, x2, y2].
    scores:
        Detector confidence score for each box (same order as *boxes*).
    """

    page_path: Path
    boxes: list[list[int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


@dataclass(slots=True)
class PanelDetection:
    """Final exported panel metadata."""

    index: int
    source_page: str
    confidence: float
    box: tuple[int, int, int, int]
    output_path: Path
