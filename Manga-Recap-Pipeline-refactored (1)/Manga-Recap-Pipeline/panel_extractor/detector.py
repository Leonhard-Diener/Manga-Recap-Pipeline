"""RT-DETRv4-X Manga109-s inference and optional tiled recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

from .types import Detection

BODY_CLASS_ID = 0
TEXT_CLASS_ID = 1
FRAME_CLASS_ID = 2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_session(model_path: Path) -> ort.InferenceSession:
    """Load the ONNX model and prefer CUDA when an execution provider exists."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"RT-DETR model not found: {model_path}. Run download_model.py first."
        )

    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(str(model_path), providers=providers)


def _preprocess(image: Image.Image, image_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize a page to the model input size and return CHW float32 data."""
    original_size = image.size
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = array.transpose(2, 0, 1)[None, ...]
    return tensor, original_size


def predict(
    session: ort.InferenceSession,
    image: Image.Image,
    *,
    image_size: int,
    panel_confidence: float,
    text_confidence: float,
    body_confidence: float,
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Run RT-DETR and split its detections into panels, text and bodies."""
    tensor, original_size = _preprocess(image, image_size)
    target_sizes = np.asarray([[original_size[0], original_size[1]]], dtype=np.int64)

    inputs = session.get_inputs()
    input_names = {item.name for item in inputs}
    feed: dict[str, np.ndarray] = {
        "images" if "images" in input_names else inputs[0].name: tensor
    }
    if "orig_target_sizes" in input_names:
        feed["orig_target_sizes"] = target_sizes

    outputs = session.run(None, feed)
    if len(outputs) < 3:
        raise RuntimeError(
            f"Unexpected RT-DETR output count: {len(outputs)}; expected labels, boxes and scores."
        )

    labels = np.asarray(outputs[0])[0]
    boxes = np.asarray(outputs[1])[0]
    scores = np.asarray(outputs[2])[0]

    panels: list[Detection] = []
    texts: list[Detection] = []
    bodies: list[Detection] = []

    for label, box, score in zip(labels, boxes, scores):
        class_id = int(label)
        score = float(score)
        detection = Detection(np.asarray(box, dtype=np.float32), score, class_id)

        if class_id == FRAME_CLASS_ID and score >= panel_confidence:
            panels.append(detection)
        elif class_id == TEXT_CLASS_ID and score >= text_confidence:
            texts.append(detection)
        elif class_id == BODY_CLASS_ID and score >= body_confidence:
            bodies.append(detection)

    return panels, texts, bodies


def page_tiles(
    image: Image.Image,
    session: ort.InferenceSession,
    *,
    image_size: int,
    panel_confidence: float,
    text_confidence: float,
    body_confidence: float,
    grid: tuple[int, int],
    overlap: float,
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Run recovery inference on overlapping page tiles and restore coordinates."""
    width, height = image.size
    columns, rows = grid
    panels: list[Detection] = []
    texts: list[Detection] = []
    bodies: list[Detection] = []

    for row in range(rows):
        for column in range(columns):
            x0 = int(width * column / columns)
            x1 = int(width * (column + 1) / columns)
            y0 = int(height * row / rows)
            y1 = int(height * (row + 1) / rows)

            tile_w = x1 - x0
            tile_h = y1 - y0
            pad_x = int(tile_w * overlap / 2)
            pad_y = int(tile_h * overlap / 2)

            tx0 = max(0, x0 - pad_x)
            ty0 = max(0, y0 - pad_y)
            tx1 = min(width, x1 + pad_x)
            ty1 = min(height, y1 + pad_y)

            tile = image.crop((tx0, ty0, tx1, ty1))
            tile_panels, tile_texts, tile_bodies = predict(
                session,
                tile,
                image_size=image_size,
                panel_confidence=panel_confidence,
                text_confidence=text_confidence,
                body_confidence=body_confidence,
            )

            offset = np.array([tx0, ty0, tx0, ty0], dtype=np.float32)
            for detection in tile_panels:
                detection.box += offset
            for detection in tile_texts:
                detection.box += offset
            for detection in tile_bodies:
                detection.box += offset

            panels.extend(tile_panels)
            texts.extend(tile_texts)
            bodies.extend(tile_bodies)

    return panels, texts, bodies
