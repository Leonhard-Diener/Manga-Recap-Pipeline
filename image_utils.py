# Image processing helpers.

import cv2
import numpy as np
from PIL import Image

from config import DARK_PIXEL_THRESHOLD, DARK_RATIO_THRESHOLD, TRIM_MARGIN


def _find_content_bounds(dark_mask: np.ndarray) -> tuple[int, int, int, int]:
    # Locate the first/last row and column that contain enough dark pixels.
    height, width = dark_mask.shape

    col_ratios = dark_mask.mean(axis=0)
    row_ratios = dark_mask.mean(axis=1)

    cols_with_content = np.where(col_ratios > DARK_RATIO_THRESHOLD)[0]
    rows_with_content = np.where(row_ratios > DARK_RATIO_THRESHOLD)[0]

    left = int(cols_with_content[0]) if len(cols_with_content) else 0
    right = int(cols_with_content[-1]) if len(cols_with_content) else width - 1
    top = int(rows_with_content[0]) if len(rows_with_content) else 0
    bottom = int(rows_with_content[-1]) if len(rows_with_content) else height - 1

    return left, top, right, bottom


def trim_white_border(image: Image.Image) -> Image.Image:
    # Crop the white margin surrounding the actual panel content.
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    dark_mask = gray < DARK_PIXEL_THRESHOLD

    left, top, right, bottom = _find_content_bounds(dark_mask)
    height, width = gray.shape

    # Re-apply the margin, clamping to valid image bounds
    left = min(left + TRIM_MARGIN, width - 1)
    top = min(top + TRIM_MARGIN, height - 1)
    right = max(right - TRIM_MARGIN, left + 1)
    bottom = max(bottom - TRIM_MARGIN, top + 1)

    return image.crop((left, top, right + 1, bottom + 1))