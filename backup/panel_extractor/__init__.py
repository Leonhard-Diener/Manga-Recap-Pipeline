"""Public API for RT-DETRv4-X Manga panel extraction."""

from .page_preview import save_page_preview
from .pipeline import crop_panels, detect_panels, extract_panels
from .types import PageDetection, PanelDetection

__all__ = [
    "PageDetection",
    "PanelDetection",
    "crop_panels",
    "detect_panels",
    "extract_panels",
    "save_page_preview",
]
