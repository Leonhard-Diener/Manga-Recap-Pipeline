"""Original-page provenance; no source image is modified or deleted."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image

import config
from .matching import stable_id


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _area(box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _overlap_metrics(a, b) -> tuple[float, float, float]:
    intersection = _intersection(a, b)
    area_a, area_b = _area(a), _area(b)
    union = area_a + area_b - intersection
    smaller, larger = min(area_a, area_b), max(area_a, area_b)
    return (
        intersection / union if union > 0 else 0.0,
        intersection / smaller if smaller > 0 else 0.0,
        smaller / larger if larger > 0 else 0.0,
    )


def _same_character_box(first: dict, second: dict, *, iou_threshold: float,
                        containment_threshold: float,
                        minimum_area_ratio: float,
                        aligned_coverage_threshold: float,
                        edge_tolerance: float) -> bool:
    if first["panel_number"] != second["panel_number"]:
        return False
    iou, containment, area_ratio = _overlap_metrics(first["box"], second["box"])
    if iou >= iou_threshold or (
        containment >= containment_threshold and area_ratio >= minimum_area_ratio
    ):
        return True

    # Tile boundaries can truncate one side of a body, producing boxes with
    # modest IoU even though two other edges line up almost exactly. Requiring
    # both strong coverage and two aligned edges avoids merging ordinary
    # overlapping characters.
    if containment < aligned_coverage_threshold:
        return False
    a, b = first["box"], second["box"]
    width = min(a[2] - a[0], b[2] - b[0])
    height = min(a[3] - a[1], b[3] - b[1])
    aligned_edges = sum((
        abs(a[0] - b[0]) <= width * edge_tolerance,
        abs(a[2] - b[2]) <= width * edge_tolerance,
        abs(a[1] - b[1]) <= height * edge_tolerance,
        abs(a[3] - b[3]) <= height * edge_tolerance,
    ))
    return aligned_edges >= 2


def _panel_for_box(box: list[int], panels: list[dict]) -> int | None:
    """Return one unambiguous panel containing most of a character box."""
    size = _area(box)
    if size <= 0:
        return None
    matches = sorted(
        ((_intersection(box, panel["box"]) / size, panel["panel_number"])
         for panel in panels),
        reverse=True,
    )
    if not matches or matches[0][0] < .75:
        return None
    if len(matches) > 1 and matches[0][0] - matches[1][0] < .15:
        return None
    return matches[0][1]


def build_manifest(detections, previous=None) -> dict:
    previous_pages = {Path(p["page"]).name: p for p in (previous or {}).get("pages", [])}
    pages = []
    for order, detection in enumerate(detections):
        path = detection.page_path
        digest = file_hash(path)
        prior = previous_pages.get(path.name, {})
        if prior.get("source_hash") != digest:
            prior = {}
        with Image.open(path) as image:
            width, height = image.size
        page_id = stable_id("page_", path.parent.parent.name, path.parent.name, path.name, digest)
        provisional = getattr(detection, "provisional", False)
        panels = [{"panel_number": i, "panel_id": stable_id("panel_", page_id, box),
                   "box": box, "provisional": provisional,
                   "story_panel_id": stable_id("story_panel_", digest, box)}
                  for i, box in enumerate(detection.boxes, start=1)]
        pages.append({"page": str(path.resolve()), "page_id": page_id,
                      "source_hash": digest, "width": width, "height": height,
                      "page_order": order, "reading_direction": "right_to_left",
                      "coordinate_space": "original_page_pixels", "layout": prior.get("layout", "page"),
                      "page_role": prior.get("page_role", "uncertain"),
                      "page_role_source": prior.get("page_role_source", "review" if prior else "unknown"),
                      "panels": panels, "geometry_revision": stable_id("geometry_", panels),
                      "text_regions": [
                          {"box": list(box)}
                          for box in getattr(detection, "text_boxes", [])
                      ],
                      "preview_hash": file_hash(path_preview) if (
                          path_preview := getattr(detection, "preview_path", None)) else None})
    return {"schema_version": 2, "pages": pages}


def serialise_proposals(body_detections: dict, manifest: dict, *,
                        min_score: float = config.CHARACTER_LABEL_CONFIDENCE,
                        duplicate_iou: float = .75,
                        duplicate_containment: float = .90,
                        duplicate_min_area_ratio: float = .40,
                        duplicate_aligned_coverage: float = .80,
                        duplicate_edge_tolerance: float = .10,
                        duplicate_larger_score_margin: float = .08) -> list:
    """Create deterministic request-local C labels for body proposals.

    Labels are deliberately page-local. ``detection_id`` is the durable bridge
    from a Qwen response to the later ReID instance, even if ReID combines an
    overlapping proposal from another detector source.
    """
    pages = {Path(p["page"]).name: p for p in manifest["pages"]}
    result = []
    for path, items in body_detections.items():
        page = pages[path.name]
        width, height = int(page.get("width", 0)), int(page.get("height", 0))
        candidates = []
        for detection in items:
            score = float(detection.score)
            if not math.isfinite(score) or score < min_score:
                continue
            raw_box = [int(round(float(value))) for value in detection.box]
            box = [
                max(0, min(width, raw_box[0])),
                max(0, min(height, raw_box[1])),
                max(0, min(width, raw_box[2])),
                max(0, min(height, raw_box[3])),
            ]
            if _area(box) <= 0:
                continue
            panel_number = _panel_for_box(box, page["panels"])
            # A box outside one unambiguous panel cannot safely ground an
            # action, transformation endpoint, or ReID instance.
            if panel_number is None:
                continue
            candidates.append({"box": box, "score": score,
                               "panel_number": panel_number})
        candidates.sort(key=lambda row: (-row["score"], row["box"]))

        # Full-page and tiled inference often return slightly shifted or nested
        # views of one body. Suppress only within the same panel; the area-ratio
        # guard avoids merging a small foreground character into a larger one.
        unique = []
        for candidate in candidates:
            duplicate_indexes = [
                index for index, existing in enumerate(unique)
                if _same_character_box(
                    candidate,
                    existing,
                    iou_threshold=duplicate_iou,
                    containment_threshold=duplicate_containment,
                    minimum_area_ratio=duplicate_min_area_ratio,
                    aligned_coverage_threshold=duplicate_aligned_coverage,
                    edge_tolerance=duplicate_edge_tolerance,
                )
            ]
            if not duplicate_indexes:
                unique.append(candidate)
                continue

            cluster = [candidate] + [unique[index] for index in duplicate_indexes]
            strongest_score = max(row["score"] for row in cluster)
            credible = [
                row for row in cluster
                if row["score"] >= strongest_score - duplicate_larger_score_margin
            ]
            # Prefer a fuller crop only when its detector confidence is close
            # to the strongest duplicate. This prevents a weak, oversized box
            # from replacing a reliable character detection.
            survivor = max(credible, key=lambda row: (
                _area(row["box"]), row["score"], tuple(-value for value in row["box"]),
            ))
            unique = [
                row for index, row in enumerate(unique)
                if index not in duplicate_indexes
            ]
            unique.append(survivor)

        unique.sort(key=lambda row: (
            row["panel_number"] if row["panel_number"] is not None else 10**9,
            (row["box"][1] + row["box"][3]) / 2,
            -(row["box"][0] + row["box"][2]) / 2,
            row["box"],
        ))
        characters = []
        for index, candidate in enumerate(unique, start=1):
            box = candidate["box"]
            characters.append({
                "index": index,
                "label": f"C{index}",
                "detection_id": stable_id(
                    "det_", path.parent.parent.name, path.parent.name,
                    path.name, page["source_hash"], box,
                ),
                "box": box,
                "panel_number": candidate["panel_number"],
                "score": candidate["score"],
            })
        result.append({
            "page": str(path.resolve()),
            "source_hash": page["source_hash"],
            "geometry_revision": page["geometry_revision"],
            "grounding_revision": stable_id(
                "grounding_", page["source_hash"], page["geometry_revision"], characters,
            ),
            "characters": characters,
        })
    return result
