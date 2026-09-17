#!/usr/bin/env python3
"""Standalone test for the fine-tuned DINOv3 anime character Re-ID model.

The script does not modify the Manga-Recap-Pipeline and does not perform
character detection. It expects character crops as input and compares them
against a reference gallery.

Directory layout
----------------
character_reid_test/
├── references/
│   ├── Character_A/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── Character_B/
│       ├── 001.jpg
│       └── ...
├── query/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── gender.json                 # optional

Example gender.json
-------------------
{
    "Character_A": "he",
    "Character_B": "she"
}

For an uncertain identity, the script uses the reference character's
configured pronoun when available. If there is no usable gender mapping,
it returns "he/she" instead of inventing a character name.

Usage
-----
    python test_character_reid.py

The default paths can be changed in the configuration section below.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path("./models/DINOv3/dino_v3.onnx")

TEST_ROOT = Path("./character_reid_test")
REFERENCE_DIR = TEST_ROOT / "references"
QUERY_DIR = TEST_ROOT / "query"
GENDER_FILE = TEST_ROOT / "gender.json"
RESULTS_FILE = TEST_ROOT / "results.json"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

# These are the thresholds used by the referenced anime Re-ID project.
# >= 0.80: confident match
# 0.69-0.79: uncertain match -> top-k voting
# < 0.69: reject the identity match
THRESHOLD_CONFIDENT = 0.80
THRESHOLD_UNCERTAIN = 0.69
TOP_K_VOTE = 5
MIN_EMBEDDINGS_UNCERTAIN = 5

# The published ONNX inference code uses 448x448 padded inputs.
TARGET_SIZE = (448, 448)
PAD_COLOR = (128, 128, 128)

# ImageNet normalization used by the published implementation.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# With an 8 GB GPU this is intentionally conservative. Increase it if VRAM
# usage stays comfortably below the limit.
BATCH_SIZE = 8


# ============================================================
# File helpers
# ============================================================

def get_images(directory: Path) -> list[Path]:
    """Return supported images from one directory in stable order."""
    if not directory.is_dir():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_gallery() -> dict[str, list[Path]]:
    """Load one reference folder per character identity."""
    if not REFERENCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Reference directory does not exist: {REFERENCE_DIR}\n"
            "Create character subfolders such as references/Character_A/."
        )

    gallery: dict[str, list[Path]] = {}

    for character_dir in sorted(REFERENCE_DIR.iterdir()):
        if not character_dir.is_dir():
            continue

        images = get_images(character_dir)
        if images:
            gallery[character_dir.name] = images

    if not gallery:
        raise FileNotFoundError(
            f"No reference images found below {REFERENCE_DIR}."
        )

    return gallery


def load_gender_map() -> dict[str, str]:
    """Load optional character -> pronoun mappings."""
    if not GENDER_FILE.is_file():
        return {}

    with GENDER_FILE.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        raise ValueError("gender.json must contain a JSON object.")

    gender: dict[str, str] = {}
    for character, pronoun in raw.items():
        if pronoun not in {"he", "she"}:
            raise ValueError(
                f"Invalid pronoun for {character!r}: {pronoun!r}. "
                "Use 'he' or 'she'."
            )
        gender[str(character)] = pronoun

    return gender


# ============================================================
# Preprocessing
# ============================================================

def preprocess(image: Image.Image) -> np.ndarray:
    """Convert one image to the exact CHW float32 format used by the model."""
    image = ImageOps.pad(
        image.convert("RGB"),
        TARGET_SIZE,
        method=Image.Resampling.BILINEAR,
        color=PAD_COLOR,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1))

    return np.ascontiguousarray(array, dtype=np.float32)


def prepare_batch(paths: list[Path]) -> np.ndarray:
    """Load and preprocess a batch of images."""
    tensors = []

    for path in paths:
        with Image.open(path) as image:
            tensors.append(preprocess(image))

    return np.stack(tensors, axis=0).astype(np.float32, copy=False)


# ============================================================
# Model
# ============================================================

def load_model() -> ort.InferenceSession:
    """Load the ONNX model with CUDA when available."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Expected dino_v3.onnx and dino_v3.onnx.data in models/DINOv3/."
        )

    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Execution provider: CUDA")
    else:
        providers = ["CPUExecutionProvider"]
        print("Execution provider: CPU")
        print(
            "WARNING: onnxruntime cannot see CUDAExecutionProvider. "
            "Install/use onnxruntime-gpu if you want CUDA inference."
        )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=providers,
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"Input : {input_info.name} | {input_info.shape} | {input_info.type}")
    print(f"Output: {output_info.name} | {output_info.shape} | {output_info.type}")
    print(f"Providers: {session.get_providers()}")

    return session


def embed_images(
    session: ort.InferenceSession,
    paths: list[Path],
    batch_size: int = BATCH_SIZE,
) -> list[np.ndarray]:
    """Generate one embedding for every image."""
    if not paths:
        return []

    input_name = session.get_inputs()[0].name
    embeddings: list[np.ndarray] = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = prepare_batch(batch_paths)

        outputs = session.run(
            None,
            {input_name: batch},
        )

        if not outputs:
            raise RuntimeError("The DINOv3 ONNX model returned no outputs.")

        batch_embeddings = np.asarray(outputs[0], dtype=np.float32)
        if batch_embeddings.ndim == 1:
            batch_embeddings = batch_embeddings.reshape(1, -1)

        if batch_embeddings.shape[0] != len(batch_paths):
            raise RuntimeError(
                f"Unexpected output shape {batch_embeddings.shape}; "
                f"expected first dimension {len(batch_paths)}."
            )

        for embedding in batch_embeddings:
            if not np.all(np.isfinite(embedding)):
                raise RuntimeError("Model produced a non-finite embedding.")
            embeddings.append(embedding.reshape(-1))

    return embeddings


# ============================================================
# Reference index
# ============================================================

def build_reference_index(
    session: ort.InferenceSession,
    gallery: dict[str, list[Path]],
) -> list[dict]:
    """Create one embedding record per reference image."""
    all_paths: list[Path] = []
    path_to_character: dict[Path, str] = {}

    for character, paths in gallery.items():
        for path in paths:
            all_paths.append(path)
            path_to_character[path] = character

    print(
        f"\nBuilding reference index: {len(all_paths)} images "
        f"across {len(gallery)} characters..."
    )

    start = time.perf_counter()
    embeddings = embed_images(session, all_paths)
    elapsed = time.perf_counter() - start

    records = []
    for path, embedding in zip(all_paths, embeddings):
        records.append(
            {
                "character": path_to_character[path],
                "path": path,
                "embedding": embedding,
            }
        )

    print(f"Reference embeddings: {elapsed:.2f}s")
    if records:
        print(f"Embedding dimension: {len(records[0]['embedding'])}")

    return records


# ============================================================
# Matching
# ============================================================

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate dot-product similarity, matching the published database code."""
    return float(np.dot(a, b))


def pronoun_for_candidate(
    character: str | None,
    gender_map: dict[str, str],
) -> str:
    """Return a known pronoun or the neutral fallback requested for uncertainty."""
    if character and character in gender_map:
        return gender_map[character]
    return "he/she"


def match_character(
    query_embedding: np.ndarray,
    reference_records: list[dict],
    gender_map: dict[str, str],
) -> dict:
    """Match a query embedding with conservative identity assignment."""
    ranked = []

    for record in reference_records:
        ranked.append(
            {
                "character": record["character"],
                "path": str(record["path"]),
                "similarity": similarity(
                    query_embedding,
                    record["embedding"],
                ),
            }
        )

    ranked.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    top_k = ranked[:TOP_K_VOTE]
    if not top_k:
        return {
            "result": "he/she",
            "status": "no-reference",
            "similarity": None,
            "top_matches": [],
        }

    best = top_k[0]
    best_character = best["character"]
    best_similarity = float(best["similarity"])

    # High-confidence match.
    if best_similarity >= THRESHOLD_CONFIDENT:
        return {
            "result": best_character,
            "status": "confident",
            "similarity": best_similarity,
            "top_matches": top_k,
        }

    # Uncertain zone: require repeated support from the same character.
    if best_similarity >= THRESHOLD_UNCERTAIN:
        candidates = [
            item
            for item in top_k
            if item["similarity"] >= THRESHOLD_UNCERTAIN
        ]

        votes = Counter(
            item["character"]
            for item in candidates
        )

        if votes:
            winner, _ = votes.most_common(1)[0]
            observation_count = sum(
                1
                for record in reference_records
                if record["character"] == winner
            )

            if observation_count >= MIN_EMBEDDINGS_UNCERTAIN:
                winner_similarity = max(
                    item["similarity"]
                    for item in candidates
                    if item["character"] == winner
                )

                return {
                    "result": winner,
                    "status": "uncertain-voted",
                    "similarity": float(winner_similarity),
                    "top_matches": top_k,
                }

    # Below the acceptance criteria: do not guess a name.
    return {
        "result": pronoun_for_candidate(
            best_character,
            gender_map,
        ),
        "status": "uncertain-fallback",
        "similarity": best_similarity,
        "best_candidate": best_character,
        "top_matches": top_k,
    }


# ============================================================
# Test
# ============================================================

def main() -> None:
    print("=" * 72)
    print("DINOv3 ANIME CHARACTER RE-ID TEST")
    print("=" * 72)

    gallery = load_gallery()
    query_paths = get_images(QUERY_DIR)
    gender_map = load_gender_map()

    if not query_paths:
        raise FileNotFoundError(
            f"No query images found in {QUERY_DIR}."
        )

    print(f"Characters : {len(gallery)}")
    print(
        "References : "
        f"{sum(len(paths) for paths in gallery.values())}"
    )
    print(f"Queries    : {len(query_paths)}")
    print(f"Model      : {MODEL_PATH}")

    session = load_model()
    reference_records = build_reference_index(session, gallery)

    print("\nGenerating query embeddings...")
    query_start = time.perf_counter()
    query_embeddings = embed_images(session, query_paths)
    query_elapsed = time.perf_counter() - query_start
    print(f"Query embeddings: {query_elapsed:.2f}s")

    results: dict[str, dict] = {}

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    for path, embedding in zip(query_paths, query_embeddings):
        result = match_character(
            embedding,
            reference_records,
            gender_map,
        )
        results[path.name] = result

        score = result["similarity"]
        score_text = "n/a" if score is None else f"{score:.4f}"

        print(f"\n{path.name}")
        print(f"  Result     : {result['result']}")
        print(f"  Status     : {result['status']}")
        print(f"  Similarity : {score_text}")

        if result.get("best_candidate"):
            print(
                f"  Candidate  : {result['best_candidate']}"
            )

        print("  Top matches:")
        for index, item in enumerate(result["top_matches"], start=1):
            print(
                f"    {index}. {item['character']:<24} "
                f"{item['similarity']:.4f}  "
                f"{Path(item['path']).name}"
            )

    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
