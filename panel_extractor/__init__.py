"""Public API for RT-DETRv4-X Manga panel extraction."""

from .pipeline import extract_panels
from .types import PanelDetection

__all__ = ["PanelDetection", "extract_panels"]
