# Image processing helpers.

import cv2
import numpy as np
from PIL import Image

from config import DARK_PIXEL_THRESHOLD, TRIM_MARGIN


def _find_content_bounds(
    dark_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """Find the bounding box of visible content."""
    height, width = dark_mask.shape

    mask = dark_mask.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return 0, 0, width - 1, height - 1

    min_area = max(4, int(width * height * 0.000005))

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    if not valid_contours:
        return 0, 0, width - 1, height - 1

    left = width
    top = height
    right = 0
    bottom = 0

    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)

        left = min(left, x)
        top = min(top, y)
        right = max(right, x + w - 1)
        bottom = max(bottom, y + h - 1)

    return left, top, right, bottom


def trim_white_border(image: Image.Image) -> Image.Image:
    """Remove white margins while preserving a safety area around content."""
    gray = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2GRAY,
    )

    dark_mask = gray < DARK_PIXEL_THRESHOLD

    left, top, right, bottom = _find_content_bounds(dark_mask)

    height, width = gray.shape

    left = max(0, left - TRIM_MARGIN)
    top = max(0, top - TRIM_MARGIN)
    right = min(width - 1, right + TRIM_MARGIN)
    bottom = min(height - 1, bottom + TRIM_MARGIN)

    if left == 0 and top == 0 and right == width - 1 and bottom == height - 1:
        return image.copy()

    return image.crop(
        (
            left,
            top,
            right + 1,
            bottom + 1,
        )
    )