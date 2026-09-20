"""Detect people on manga pages and save inspectable crops.

This is deliberately separate from ``test_character_reid.py``. Run this
first, inspect the saved crops, then run the ReID script to assign provisional
identity labels.

Usage
-----
    python test_people_detection.py "Manga title"

Optional second argument selects one chapter:
    python test_people_detection.py "Manga title" chapter_001
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import config
from common.utils import find_chapter_dirs, find_manga_dir


PERSON_MODEL_PATH = Path("./models/person_detection_model.onnx")
PERSON_DETECTION_FILE_NAME = "person_detections.json"
DEBUG_ROOT = Path("./debug/person_detection")

# The model is evaluated on whole pages. This retains enough context for
# profiles, foreshortened bodies, and partially visible characters.
PERSON_DETECTION_SIZE = 640
PERSON_CONF_THRESHOLD = 0.30
PERSON_IOU_THRESHOLD = 0.50
# Retain small candidates. The shared ReID pipeline marks low-detail crops as
# unsuitable prototypes instead of deleting possible story characters. This
# detector is still person-trained; broad body/entity proposals supplement it.
MIN_PERSON_AREA_RATIO = 0.0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _providers() -> list[str]:
    available = ort.get_available_providers()
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )


def load_person_session() -> ort.InferenceSession:
    if not PERSON_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Anime person model not found: {PERSON_MODEL_PATH}.")
    session = ort.InferenceSession(str(PERSON_MODEL_PATH), providers=_providers())
    print(f"Person detector: {session.get_providers()}")
    return session


def _nms_indices(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Return score-ordered boxes after standard IoU suppression."""
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        current_box = boxes[current]
        others = boxes[order[1:]]
        x1 = np.maximum(current_box[0], others[:, 0])
        y1 = np.maximum(current_box[1], others[:, 1])
        x2 = np.minimum(current_box[2], others[:, 2])
        y2 = np.minimum(current_box[3], others[:, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        iou = intersection / np.maximum(1e-6, current_area + other_areas - intersection)
        order = order[1:][iou <= PERSON_IOU_THRESHOLD]
    return kept


def detect_people(
    session: ort.InferenceSession,
    image: Image.Image,
) -> list[tuple[list[float], float]]:
    """Return full-page person boxes in original-image xyxy coordinates."""
    width, height = image.size
    resized = image.convert("RGB").resize(
        (PERSON_DETECTION_SIZE, PERSON_DETECTION_SIZE),
        Image.Resampling.BILINEAR,
    )
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])
    input_name = session.get_inputs()[0].name
    output = np.asarray(session.run(None, {input_name: tensor})[0])[0]

    if output.ndim != 2 or output.shape[0] != 5:
        raise RuntimeError(f"Unexpected person-detector output shape: {output.shape}")
    candidates = output.T
    candidates = candidates[candidates[:, 4] >= PERSON_CONF_THRESHOLD]
    if not len(candidates):
        return []

    boxes = candidates[:, :4].copy()
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes[:, [0, 2]] *= width / PERSON_DETECTION_SIZE
    boxes[:, [1, 3]] *= height / PERSON_DETECTION_SIZE
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

    scores = candidates[:, 4]
    page_area = max(1, width * height)
    detections: list[tuple[list[float], float]] = []
    for index in _nms_indices(boxes, scores):
        box = boxes[index]
        area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / page_area
        if area_ratio >= MIN_PERSON_AREA_RATIO:
            detections.append(([float(value) for value in box], float(scores[index])))
    return detections


def _crop_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def process_chapter(
    chapter_dir: Path,
    chapter_output: Path,
    session: ort.InferenceSession,
) -> dict:
    manga_name = chapter_output.parent.name
    crops_dir = DEBUG_ROOT / manga_name / chapter_dir.name / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    # These are derived inspection files owned by this script. Remove only
    # its own old crop naming pattern so the folder always mirrors the JSON.
    for stale_crop in crops_dir.glob("*_person_*.jpg"):
        stale_crop.unlink()
    detections: list[dict] = []
    start = time.perf_counter()

    for page_path in sorted(
        path for path in chapter_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        with Image.open(page_path) as opened:
            image = opened.convert("RGB")
            people = detect_people(session, image)
            for index, (box, score) in enumerate(people, start=1):
                crop_box = _crop_box(box, *image.size)
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    continue
                crop_path = crops_dir / f"{page_path.stem}_person_{index:03d}.jpg"
                # This is the untouched crop used by DINO; no label or overlay.
                image.crop(crop_box).save(crop_path, quality=95)
                detections.append(
                    {
                        "page": str(page_path.resolve()),
                        "character_index": index,
                        "box": [round(value, 2) for value in box],
                        "detector_score": round(score, 4),
                        "crop_image": str(crop_path),
                    }
                )

    result = {
        "chapter": chapter_dir.name,
        "model": str(PERSON_MODEL_PATH),
        "confidence_threshold": PERSON_CONF_THRESHOLD,
        "iou_threshold": PERSON_IOU_THRESHOLD,
        "minimum_area_ratio": MIN_PERSON_AREA_RATIO,
        "detection_count": len(detections),
        "detections": detections,
    }
    chapter_output.mkdir(parents=True, exist_ok=True)
    output_file = chapter_output / PERSON_DETECTION_FILE_NAME
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  People detected: {len(detections)} ({time.perf_counter() - start:.2f}s)")
    print(f"  Saved: {output_file}")
    print(f"  Clean crops: {crops_dir}")
    return result


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else (config.MANGA_TITLE or None)
    requested_chapter = sys.argv[2] if len(sys.argv) > 2 else None
    manga_dir = find_manga_dir(config.INPUT_DIR, title)
    chapter_dirs = find_chapter_dirs(manga_dir)
    if requested_chapter:
        chapter_dirs = [chapter for chapter in chapter_dirs if chapter.name == requested_chapter]
        if not chapter_dirs:
            raise FileNotFoundError(f"Chapter '{requested_chapter}' not found in {manga_dir}.")

    print(f"Manga: {manga_dir.name}  |  Chapters: {len(chapter_dirs)}")
    session = load_person_session()
    total = 0
    for chapter_dir in chapter_dirs:
        print(f"\n-- {chapter_dir.name} --")
        chapter_output = config.OUTPUT_DIR / manga_dir.name / chapter_dir.name
        total += process_chapter(chapter_dir, chapter_output, session)["detection_count"]
    print(f"\nDone. {total} people detected.")


if __name__ == "__main__":
    main()
