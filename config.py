''' Panel Extraction Settings '''
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = BASE_DIR / "models" / "model.onnx"

# RT-DETRv4-X Manga109-s inference.
IMAGE_SIZE = 1280
PANEL_CONFIDENCE = 0.35
TEXT_CONFIDENCE = 0.20
BODY_CONFIDENCE = 0.20

# Recover panels when full-page inference leaves too much of the page
# unexplained. Tiles are used only as page-level recovery, never recursively.
ENABLE_TILED_FALLBACK = True
FALLBACK_UNCOVERED_RATIO = 0.35
FALLBACK_GRID = (2, 2)
FALLBACK_OVERLAP = 0.20

# Conservative panel filtering and duplicate removal.
MINIMUM_PANEL_RATIO = 0.05
DUPLICATE_IOU = 0.80
DUPLICATE_CONTAINMENT = 0.90

# Content protection. These values only affect a panel when detected content
# actually crosses the panel boundary.
TEXT_PADDING_X = 0.10
TEXT_PADDING_Y = 0.15
TEXT_SAFETY_PIXELS = 18
BODY_PADDING_X = 0.04
BODY_PADDING_Y = 0.05
BODY_SAFETY_PIXELS = 10
MAX_CONTENT_EXPANSION_RATIO = 0.15

# Export.
JPEG_QUALITY = 95
SAVE_DEBUG = True

''' Downloader settings '''
BASE_URL = "https://api.mangadex.org/manga"