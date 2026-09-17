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

# ── Post-processing step toggles ──────────────────────────────────────────────
# Set any flag to False to disable that specific post-processing step entirely.
# Useful for diagnosing which step causes wrong panel areas.

# Remove overlapping / near-identical panel detections after inference (and
# after the tiled fallback if it runs).
ENABLE_DEDUPLICATION = True

# Discard panels whose area is smaller than MINIMUM_PANEL_RATIO × page area.
ENABLE_SMALL_PANEL_FILTER = True

# Expand panel boxes to include text / body content that crosses the boundary.
ENABLE_CONTENT_PROTECTION = False

# Final pass that drops crops almost completely contained by another crop.
ENABLE_NESTED_CLEANUP = True

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
SAVE_DEBUG = False

def panel_extraction_kwargs() -> dict:
    """All panel-extraction settings, ready to unpack into
    ``detect_panels()`` / ``crop_panels()`` / ``extract_panels()``.
    """
    return dict(
        image_size=IMAGE_SIZE,
        panel_confidence=PANEL_CONFIDENCE,
        text_confidence=TEXT_CONFIDENCE,
        body_confidence=BODY_CONFIDENCE,
        enable_tiled_fallback=ENABLE_TILED_FALLBACK,
        fallback_uncovered_ratio=FALLBACK_UNCOVERED_RATIO,
        fallback_grid=FALLBACK_GRID,
        fallback_overlap=FALLBACK_OVERLAP,
        minimum_panel_ratio=MINIMUM_PANEL_RATIO,
        duplicate_iou=DUPLICATE_IOU,
        duplicate_containment=DUPLICATE_CONTAINMENT,
        enable_deduplication=ENABLE_DEDUPLICATION,
        enable_small_panel_filter=ENABLE_SMALL_PANEL_FILTER,
        enable_content_protection=ENABLE_CONTENT_PROTECTION,
        enable_nested_cleanup=ENABLE_NESTED_CLEANUP,
        text_padding_x=TEXT_PADDING_X,
        text_padding_y=TEXT_PADDING_Y,
        text_safety_pixels=TEXT_SAFETY_PIXELS,
        body_padding_x=BODY_PADDING_X,
        body_padding_y=BODY_PADDING_Y,
        body_safety_pixels=BODY_SAFETY_PIXELS,
        max_content_expansion_ratio=MAX_CONTENT_EXPANSION_RATIO,
        jpeg_quality=JPEG_QUALITY,
        save_debug=SAVE_DEBUG,
    )


''' Downloader settings '''
BASE_URL = "https://api.mangadex.org/manga"
# Download all chapters or only the first chapter
DOWNLOAD_ALL_CHAPTERS = True

# Optional. If empty, the pipeline auto-detects the single manga folder
# already present under INPUT_DIR. Set this if you keep multiple manga
# side by side, or pass a title on the command line to main.py / the
# individual test_*.py scripts instead.
MANGA_TITLE = ""

''' Qwen3.5-2B settings (steps 2 and 4) '''
# Same local model, used both to classify pages (MANGA/EXTRA) and to
# describe panels. Run download_model.py or place the model manually at
# this path.
QWEN_MODEL_PATH = BASE_DIR / "models" / "Qwen3.5-2B"

# Step 2: page classification (removal of credits/shoutouts/intros).
CLASSIFIER_MIN_PIXELS = 256 * 28 * 28
CLASSIFIER_MAX_PIXELS = 512 * 28 * 28
CLEANUP_MAX_NEW_TOKENS = 4

# Step 4: per-panel page description.
DESCRIPTION_MIN_PIXELS = 256 * 28 * 28
DESCRIPTION_MAX_PIXELS = 384 * 28 * 28
DESCRIPTION_MAX_NEW_TOKENS = 48
DESCRIPTION_ENABLE_THINKING = False
