# Detects manga panels using the ShadowB segmentation model.

import os
from dataclasses import dataclass

from PIL import Image
from ultralytics import YOLO

import config
from panel_extractor_B.image_utils import trim_white_border
from panel_extractor_B.reading_order import sort_reading_order


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Panel:
    index: int
    box: tuple[int, int, int, int]
    confidence: float
    output_path: str
    source_page: str


def _list_page_images(input_dir: str) -> list[str]:
    """Return all supported page images sorted by filename."""
    paths = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not paths:
        raise FileNotFoundError(
            f"No supported page images found in '{input_dir}'"
        )

    return sorted(paths)


def detect_panels(model: YOLO, image_path: str):
    """Run inference and return detections for the frame class."""
    results = model.predict(
        image_path,
        conf=config.CONFIDENCE,
        classes=[0],
        verbose=False,
    )

    return results[0].boxes


def _extract_page_panels(
    image_path: str,
    model: YOLO,
    start_index: int,
) -> list[Panel]:
    """Detect, order, crop, and save panels from one page."""
    image = Image.open(image_path).convert("RGB")
    boxes = detect_panels(model, image_path)

    detections = [
        (
            int(box.xyxy[0][0]),
            int(box.xyxy[0][1]),
            int(box.xyxy[0][2]),
            int(box.xyxy[0][3]),
            float(box.conf[0]),
        )
        for box in boxes
    ]

    ordered_detections = sort_reading_order(
        detections,
        box_key=lambda detection: detection[:4],
    )

    page_name = os.path.basename(image_path)
    panels = []

    for offset, (x1, y1, x2, y2, confidence) in enumerate(
        ordered_detections
    ):
        panel_image = trim_white_border(
            image.crop((x1, y1, x2, y2))
        )

        index = start_index + offset
        output_path = os.path.join(
            config.OUTPUT_DIR,
            f"{index}.jpg",
        )

        panel_image.save(
            output_path,
            quality=config.JPEG_QUALITY,
        )

        panels.append(
            Panel(
                index=index,
                box=(x1, y1, x2, y2),
                confidence=confidence,
                output_path=output_path,
                source_page=page_name,
            )
        )

    return panels


def extract_panels(input_dir: str = None) -> list[Panel]:
    """Process all pages and return the extracted panels."""
    input_dir = input_dir or config.INPUT_DIR

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model = YOLO(config.MODEL_PATH)

    all_panels = []
    next_index = 1

    for image_path in _list_page_images(input_dir):
        page_panels = _extract_page_panels(
            image_path,
            model,
            next_index,
        )

        all_panels.extend(page_panels)
        next_index += len(page_panels)

    return all_panels