"""Standalone manga-wide ReID runner; reusable matching lives in story_entities.

python test_character_reid.py "Manga title"
python test_character_reid.py "Manga title" --reviews reviews.json
See CHARACTER_REID_USAGE.md for conservative acceptance, galleries, and calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

import config
from common.utils import find_chapter_dirs, find_manga_dir
from story_entities.runner import publish_result, export_review_gallery
from common.json_io import write_json
from story_entities.pipeline import Encoder, classify_manga, load_calibration, read_json

# Compatibility helpers remain available to the separate CCIP/Qwen experiments.
# The production runner below does not execute those experiments.
MODEL_PATH = config.BASE_DIR / "models" / "DINOv3" / "dino_v3.onnx"
PERSON_DETECTION_FILE_NAME = "person_detections.json"
CLASSIFICATION_FILE_NAME = "character_classification.json"
IDENTITY_OVERRIDE_FILE_NAME = "character_identity_overrides.json"
TARGET_SIZE = (448, 448)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def identity_label(index: int) -> str:
    index += 1
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result

def placeholder_name(index: int) -> str:
    """Neutral debug label for legacy experiment imports; never a person name."""
    return f"Character_{index + 1:03d}"

def _providers() -> list[str]:
    """Return the preferred ONNX Runtime providers for DINOv3."""
    available = ort.get_available_providers()
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )

# ---------------------------------------------------------------------------
# DINOv3
# ---------------------------------------------------------------------------


def load_dino_session() -> ort.InferenceSession:
    """Load the exported DINOv3 ONNX model using CUDA when available."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"DINOv3 model not found: {MODEL_PATH}\n"
            "Expected both dino_v3.onnx and dino_v3.onnx.data in models/DINOv3/."
        )

    providers = _providers()
    if providers[0] == "CUDAExecutionProvider":
        print("DINOv3: CUDAExecutionProvider")
    else:
        print("DINOv3: CPUExecutionProvider")
        print("WARNING: onnxruntime-gpu is not available, so this runs on CPU.")

    session = ort.InferenceSession(str(MODEL_PATH), providers=providers)

    inp = session.get_inputs()[0]
    print(f"Model input : {inp.name}")
    print(f"Model shape : {inp.shape}")
    print(f"Providers   : {session.get_providers()}")

    return session


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the preprocessing used by the published Anime-ReID ONNX code."""
    padded = ImageOps.pad(
        image.convert("RGB"),
        TARGET_SIZE,
        color=(128, 128, 128),
    )

    array = np.asarray(padded, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(array, dtype=np.float32)


def embed_batch(
    session: ort.InferenceSession,
    images: list[Image.Image],
    batch_size: int = 16,
) -> np.ndarray:
    """Return one DINOv3 embedding per supplied crop."""
    if not images:
        return np.empty((0, 0), dtype=np.float32)

    input_name = session.get_inputs()[0].name
    output_chunks: list[np.ndarray] = []

    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        batch = np.stack(
            [preprocess(image) for image in batch_images],
            axis=0,
        )

        output = session.run(
            None,
            {input_name: batch},
        )

        if not output:
            raise RuntimeError("DINOv3 returned no output tensor.")

        embeddings = np.asarray(output[0], dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[0] != len(batch_images):
            raise RuntimeError(
                f"Unexpected embedding shape {embeddings.shape}; "
                f"expected {len(batch_images)} rows."
            )

        output_chunks.append(embeddings)

    return np.concatenate(output_chunks, axis=0)


def sanitize_filename(value: str) -> str:
    """Make a string safe to use as a Windows filename."""
    invalid = '<>:"/\\|?*'
    result = "".join("_" if char in invalid else char for char in value)
    return result.strip(" .") or "character"


def load_identity_overrides(chapter_output: Path) -> dict[tuple[str, int], str]:
    """Load user-confirmed identity seeds for transformations and disguises.

    The optional JSON file contains ``overrides`` entries with ``page``,
    ``character_index``, and an existing identity label (for example ``A``).
    A seed is deliberately explicit: a transformed character can look less
    similar than two entirely different people, so automatically forcing that
    merge would create more wrong identities than it fixes.
    """
    override_file = chapter_output / IDENTITY_OVERRIDE_FILE_NAME
    if not override_file.is_file():
        return {}
    data = json.loads(override_file.read_text(encoding="utf-8"))
    entries = data.get("overrides", []) if isinstance(data, dict) else []
    if not isinstance(entries, list):
        raise ValueError(f"Invalid overrides list in {override_file}.")

    overrides: dict[tuple[str, int], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        page = entry.get("page")
        identity = entry.get("identity")
        try:
            index = int(entry.get("character_index"))
        except (TypeError, ValueError):
            continue
        if isinstance(page, str) and isinstance(identity, str) and identity:
            overrides[(Path(page).name, index)] = identity.upper()
    return overrides


# ---------------------------------------------------------------------------
# Person-detection input
# ---------------------------------------------------------------------------


def load_person_detections(detection_file: Path) -> list[dict]:
    """Load validated full-page person boxes from test_people_detection.py."""
    if not detection_file.is_file():
        raise FileNotFoundError(
            f"Missing {detection_file}. Run test_people_detection.py first."
        )

    data = json.loads(detection_file.read_text(encoding="utf-8"))
    raw_detections = data.get("detections", []) if isinstance(data, dict) else []
    if not isinstance(raw_detections, list):
        raise ValueError(f"Invalid detections list in {detection_file}.")

    detections: list[dict] = []
    for item in raw_detections:
        if not isinstance(item, dict):
            continue
        page = Path(str(item.get("page", "")))
        if not page.is_file():
            page = detection_file.parent.parent.parent / page.name
        box = item.get("box")
        if not page.is_file() or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            values = [float(value) for value in box]
            score = float(item.get("detector_score", 0.0))
            index = int(item.get("character_index", 0))
        except (TypeError, ValueError):
            continue
        if values[2] <= values[0] or values[3] <= values[1]:
            continue
        detections.append(
            {
                "page": page,
                "character_index": index,
                "box": values,
                "detector_score": score,
                "person_detection_crop": item.get("crop_image"),
            }
        )
    return detections



def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default=config.MANGA_TITLE or None)
    parser.add_argument("chapter", nargs="?", help="Legacy argument; all chapters still participate in matching.")
    parser.add_argument("--encoder", choices=("ccip", "dino"), default="dino")
    parser.add_argument("--style-linker", choices=("ccip", "none"), default="ccip",
                        help="Second-model agreement for simple/detailed style links (DINO primary only).")
    parser.add_argument("--group-distance", type=float,
                        help="Visual grouping tolerance; larger joins more variants (DINO: 0.20, CCIP: 0.14).")
    parser.add_argument("--provider", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--reviews", type=Path, help="Explicit instance/form assignments and reviewed names.")
    parser.add_argument("--calibration", type=Path, help="Validated model-specific assignment thresholds.")
    parser.add_argument("--export-dir", type=Path, help="Alternate output root for non-destructive experiments.")
    parser.add_argument("--no-gallery", action="store_true")
    args = parser.parse_args(argv)
    if args.group_distance is not None and (not np.isfinite(args.group_distance) or args.group_distance < 0):
        parser.error("--group-distance must be finite and nonnegative")
    manga_dir = find_manga_dir(config.INPUT_DIR, args.title)
    chapters = find_chapter_dirs(manga_dir)
    if args.chapter and args.chapter not in {p.name for p in chapters}:
        raise FileNotFoundError(args.chapter)
    if args.chapter:
        print("Matching all available chapters to preserve global identities.")
    source_output = config.OUTPUT_DIR / manga_dir.name
    destination = (args.export_dir / manga_dir.name) if args.export_dir else source_output
    review_path = args.reviews or (source_output / "character_reviews.json")
    if args.reviews and not review_path.is_file():
        raise FileNotFoundError(review_path)
    reviews = read_json(review_path, {"schema_version": 1, "assignments": [], "different": [], "names": []})
    encoder = Encoder(config.BASE_DIR / "models", args.encoder, args.provider)
    style_encoder = None
    if args.style_linker == "ccip":
        if args.encoder != "dino":
            parser.error("--style-linker ccip requires --encoder dino")
        style_encoder = Encoder(config.BASE_DIR / "models", "ccip", args.provider)
    if args.group_distance is not None:
        encoder.default_threshold = args.group_distance
    calibration = load_calibration(args.calibration, encoder)
    analysis_data = read_json(config.OUTPUT_DIR / f"{manga_dir.name}_descriptions.json", {})
    event_data = analysis_data.get("transformations") if isinstance(analysis_data, dict) else None
    description_data = analysis_data.get("descriptions") if isinstance(analysis_data, dict) else None
    # Old split artifacts remain readable, but normal description runs now
    # create only the combined file above.
    if event_data is None:
        event_data = read_json(config.OUTPUT_DIR / f"{manga_dir.name}_transformations.json")
    result, crops, _ = classify_manga(
        manga_dir, chapters, source_output, encoder=encoder,
        cache_dir=source_output / "reid_cache", reviews=reviews,
        calibration=calibration, transformations=event_data, descriptions=description_data,
        style_encoder=style_encoder,
    )
    publish_result(result, crops, destination, chapters, reviews=reviews, calibration=calibration,
                   gallery=not args.no_gallery, source_output=source_output)
    return result


if __name__ == "__main__":
    main()
