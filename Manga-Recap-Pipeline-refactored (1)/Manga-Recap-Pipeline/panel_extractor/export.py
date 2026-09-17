"""Panel crop and debug-image export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .types import Detection, PanelDetection


def save_panel_crop(
    image: Image.Image,
    box: np.ndarray,
    output_path: Path,
    *,
    jpeg_quality: int,
) -> tuple[int, int, int, int]:
    """Crop and save one final panel box."""
    width, height = image.size
    x1, y1, x2, y2 = map(int, np.round(box))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    image.crop((x1, y1, x2, y2)).save(
        output_path,
        format="JPEG",
        quality=jpeg_quality,
    )
    return x1, y1, x2, y2


def build_result(
    index: int,
    page_name: str,
    confidence: float,
    box: tuple[int, int, int, int],
    output_path: Path,
) -> PanelDetection:
    """Create the public metadata object for an exported panel."""
    return PanelDetection(
        index=index,
        source_page=page_name,
        confidence=confidence,
        box=box,
        output_path=output_path,
    )
