"""Data structures shared by the detector and post-processing modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Detection:
    """A single RT-DETR detection in original-image coordinates."""

    box: np.ndarray
    score: float
    class_id: int


@dataclass(slots=True)
class PanelDetection:
    """Final exported panel metadata."""

    index: int
    source_page: str
    confidence: float
    box: tuple[int, int, int, int]
    output_path: Path
