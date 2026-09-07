# Image processing helpers.

from PIL import Image


def trim_white_border(image: Image.Image) -> Image.Image:
    """Return the model crop without additional content-based cropping."""
    return image.copy()