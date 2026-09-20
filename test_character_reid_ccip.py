"""Alternative manga character ReID runner using CCIP.

CCIP is an anime-character similarity model. It is kept separate from the
DINO runner so their outputs can be compared without replacing either method.

Run person detection first:
    python test_people_detection.py "Manga title" chapter_001
    python test_character_reid_ccip.py "Manga title" chapter_001
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import AgglomerativeClustering

import config
from common.utils import find_chapter_dirs, find_manga_dir
from test_character_reid import (
    IDENTITY_OVERRIDE_FILE_NAME,
    PERSON_DETECTION_FILE_NAME,
    identity_label,
    load_identity_overrides,
    load_person_detections,
    placeholder_name,
    sanitize_filename,
)


# This is the strongest CCIP checkpoint downloaded for this project.  Keep the
# lighter checkpoint on disk for comparisons, but do not silently use it for
# the production form-clustering pass.
CCIP_MODEL_DIR = Path("./models/CCIP/ccip-caformer_b36-24")
CCIP_FEATURE_MODEL = CCIP_MODEL_DIR / "model_feat.onnx"
CCIP_METRIC_MODEL = CCIP_MODEL_DIR / "model_metrics.onnx"
CCIP_METRICS_FILE = CCIP_MODEL_DIR / "metrics.json"
CLASSIFICATION_FILE_NAME = "character_classification_ccip.json"
DEBUG_ROOT = Path("./debug/character_reid_ccip")

# Complete linkage avoids an A~B~C similarity chain collapsing all characters
# into one cluster. 0.20 lets pose/expression variants of one manga character
# join while still requiring every pair in a cluster to agree.
CCIP_CLUSTER_MAX_DISTANCE = 0.20
CCIP_SIZE = 384
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _providers() -> list[str]:
    available = ort.get_available_providers()
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )


def load_sessions() -> tuple[ort.InferenceSession, ort.InferenceSession, dict]:
    missing = [path for path in (CCIP_FEATURE_MODEL, CCIP_METRIC_MODEL, CCIP_METRICS_FILE) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing CCIP model files: " + ", ".join(str(path) for path in missing)
        )
    providers = _providers()
    feature_session = ort.InferenceSession(str(CCIP_FEATURE_MODEL), providers=providers)
    metric_session = ort.InferenceSession(str(CCIP_METRIC_MODEL), providers=providers)
    print(f"CCIP feature providers: {feature_session.get_providers()}")
    return feature_session, metric_session, json.loads(CCIP_METRICS_FILE.read_text(encoding="utf-8"))


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((CCIP_SIZE, CCIP_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray((array - MEAN[:, None, None]) / STD[:, None, None])


def extract_features(session: ort.InferenceSession, crops: list[Image.Image]) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(crops), 8):
        batch = np.stack([preprocess(crop) for crop in crops[start : start + 8]])
        batches.append(np.asarray(session.run(None, {"input": batch})[0], dtype=np.float32))
    return np.concatenate(batches, axis=0)


def cluster_differences(differences: np.ndarray) -> list[list[int]]:
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CCIP_CLUSTER_MAX_DISTANCE,
        metric="precomputed",
        linkage="complete",
    ).fit_predict(differences)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)
    return sorted(grouped.values(), key=lambda members: min(members))


def apply_overrides(
    clusters: list[list[int]],
    detections: list[dict],
    overrides: dict[tuple[str, int], str],
) -> tuple[list[list[int]], set[int]]:
    """Move explicitly confirmed forms into their target cluster."""
    seeded: set[int] = set()
    for index, detection in enumerate(detections):
        target_label = overrides.get((detection["page"].name, detection["character_index"]))
        if target_label is None:
            continue
        target_index = ord(target_label) - ord("A")
        if target_index < 0 or target_index >= len(clusters):
            raise ValueError(
                f"CCIP override target '{target_label}' is not an existing cluster."
            )
        for cluster in clusters:
            if index in cluster:
                cluster.remove(index)
                break
        clusters[target_index].append(index)
        seeded.add(index)
    return [cluster for cluster in clusters if cluster], seeded


def save_debug_crop(
    crop: Image.Image,
    output_dir: Path,
    page_name: str,
    character_index: int,
    label: str,
    name: str,
    distance: float | None,
    status: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (crop.width, crop.height + 70), "white")
    canvas.paste(crop.convert("RGB"), (0, 70))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), f"{label} | {name} | {status}", fill="black", font=font)
    distance_text = "n/a" if distance is None else f"{distance:.4f}"
    draw.text((8, 28), f"CCIP distance to cluster: {distance_text}", fill="black", font=font)
    draw.text((8, 46), f"page: {page_name} | idx: {character_index}", fill="black", font=font)
    output_path = output_dir / f"{sanitize_filename(Path(page_name).stem)}_char_{character_index:03d}_{label}_{sanitize_filename(name)}.jpg"
    canvas.save(output_path, quality=95)
    return output_path


def process_chapter(chapter_dir: Path, chapter_output: Path, feature_session: ort.InferenceSession, metric_session: ort.InferenceSession, metrics: dict) -> dict:
    detections = load_person_detections(chapter_output / PERSON_DETECTION_FILE_NAME)
    if not detections:
        return {"chapter": chapter_dir.name, "characters": [], "identities": []}

    crops: list[Image.Image] = []
    valid: list[dict] = []
    for detection in detections:
        with Image.open(detection["page"]) as opened:
            image = opened.convert("RGB")
            x1, y1, x2, y2 = (int(round(value)) for value in detection["box"])
            box = (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crops.append(image.crop(box))
            valid.append(detection)

    start = time.perf_counter()
    features = extract_features(feature_session, crops)
    differences = np.asarray(metric_session.run(None, {"input": features})[0], dtype=np.float32)
    clusters = cluster_differences(differences)
    clusters, seeded = apply_overrides(clusters, valid, load_identity_overrides(chapter_output))
    elapsed = time.perf_counter() - start

    member_to_cluster = {member: cluster_index for cluster_index, cluster in enumerate(clusters) for member in cluster}
    debug_dir = DEBUG_ROOT / sanitize_filename(chapter_output.parent.name) / sanitize_filename(chapter_dir.name)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for stale in debug_dir.glob("*_char_*.jpg"):
        stale.unlink()

    classified: list[dict] = []
    for index, (detection, crop) in enumerate(zip(valid, crops)):
        cluster_index = member_to_cluster[index]
        members = clusters[cluster_index]
        label = identity_label(cluster_index)
        name = placeholder_name(cluster_index)
        other_members = [member for member in members if member != index]
        distance = min((float(differences[index, member]) for member in other_members), default=None)
        status = "manual_seed" if index in seeded else ("confirmed" if len(members) >= 2 else "unsure")
        debug_label = label if status != "unsure" else "UNSURE"
        debug_name = name if status != "unsure" else "single appearance"
        debug_path = save_debug_crop(crop, debug_dir, detection["page"].name, detection["character_index"], debug_label, debug_name, distance, status)
        classified.append({
            "page": str(detection["page"]), "character_index": detection["character_index"],
            "box": [round(float(value), 2) for value in detection["box"]],
            "detector_score": round(float(detection["detector_score"]), 4),
            "person_detection_crop": detection.get("person_detection_crop"),
            "identity": label, "name": name, "identity_status": status,
            "ccip_distance_to_cluster": round(distance, 4) if distance is not None else None,
            "debug_image": str(debug_path),
        })

    result = {
        "chapter": chapter_dir.name, "feature_model": str(CCIP_FEATURE_MODEL), "metric_model": str(CCIP_METRIC_MODEL),
        "ccip_published_same_character_threshold": metrics.get("threshold"), "complete_linkage_max_distance": CCIP_CLUSTER_MAX_DISTANCE,
        "person_detections": str(chapter_output / PERSON_DETECTION_FILE_NAME), "embedding_count": len(valid), "identity_count": len(clusters),
        "identities": [{"identity": identity_label(i), "name": placeholder_name(i), "embedding_count": len(members)} for i, members in enumerate(clusters)],
        "characters": classified,
    }
    output_file = chapter_output / CLASSIFICATION_FILE_NAME
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  CCIP: {len(valid)} crops -> {len(clusters)} identities ({elapsed:.2f}s)")
    print(f"  Saved: {output_file}")
    return result


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else (config.MANGA_TITLE or None)
    requested_chapter = sys.argv[2] if len(sys.argv) > 2 else None
    manga_dir = find_manga_dir(config.INPUT_DIR, title)
    chapters = find_chapter_dirs(manga_dir)
    if requested_chapter:
        chapters = [chapter for chapter in chapters if chapter.name == requested_chapter]
    if not chapters:
        raise FileNotFoundError(f"Requested chapter not found in {manga_dir}.")
    feature_session, metric_session, metrics = load_sessions()
    for chapter in chapters:
        print(f"\n-- {chapter.name} --")
        process_chapter(chapter, config.OUTPUT_DIR / manga_dir.name / chapter.name, feature_session, metric_session, metrics)


if __name__ == "__main__":
    main()
