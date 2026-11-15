"""Shared panel-stage runner: derived crops, manifests and body proposals."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import config
from common.json_io import write_json
from story_entities.manifest import build_manifest, serialise_proposals
from story_entities.manifest import file_hash
from .page_preview import save_grounded_page_preview, save_page_preview
from .pipeline import detect_panels, crop_panels


def _apply_reviewed_panel_corrections(detections, raw_manifest: dict,
                                      correction_path: Path, destination: Path,
                                      jpeg_quality: int) -> dict[str, dict]:
    """Apply source/revision-bound boxes, order, and roles; retain raw geometry."""
    if not correction_path.is_file():
        return {}
    data = json.loads(correction_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"Unsupported panel correction schema: {correction_path}")
    entries = data.get("pages", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid panel correction pages: {correction_path}")
    raw_by_name = {Path(row["page"]).name: row for row in raw_manifest["pages"]}
    detections_by_name = {row.page_path.name: row for row in detections}
    applied = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("page"), str):
            raise ValueError("Panel correction must identify one page.")
        name = Path(entry["page"]).name
        raw = raw_by_name.get(name)
        detection = detections_by_name.get(name)
        if raw is None or detection is None:
            raise ValueError(f"Panel correction references missing page: {name}")
        if entry.get("status") == "unresolved" and entry.get("critical") is True:
            raise ValueError(
                f"Critical panel correction is unresolved for {name}: "
                f"{entry.get('reason', 'review required')}"
            )
        if entry.get("status") != "accepted_reviewed":
            continue
        if (entry.get("source_hash") != raw.get("source_hash")
                or entry.get("base_geometry_revision") != raw.get("geometry_revision")):
            raise ValueError(f"Stale panel correction for {name}.")
        boxes = entry.get("boxes")
        if boxes is not None:
            if (not isinstance(boxes, list) or not boxes
                    or any(not isinstance(box, list) or len(box) != 4 for box in boxes)):
                raise ValueError(f"Invalid corrected boxes for {name}.")
            normalized = []
            for box in boxes:
                values = [int(round(float(value))) for value in box]
                values = [
                    max(0, min(raw["width"], values[0])),
                    max(0, min(raw["height"], values[1])),
                    max(0, min(raw["width"], values[2])),
                    max(0, min(raw["height"], values[3])),
                ]
                if values[2] <= values[0] or values[3] <= values[1]:
                    raise ValueError(f"Empty corrected box for {name}.")
                normalized.append(values)
            order = entry.get("reading_order", list(range(len(normalized))))
            if sorted(order) != list(range(len(normalized))):
                raise ValueError(f"Invalid corrected reading order for {name}.")
            detection.boxes = [normalized[index] for index in order]
            detection.scores = [1.0] * len(detection.boxes)
            detection.provisional = False
            with Image.open(detection.page_path) as source:
                detection.preview_path = save_page_preview(
                    source.convert("RGB"), detection.boxes, detection.page_path,
                    destination, jpeg_quality=jpeg_quality, preserve_order=True,
                )
        applied[name] = entry
    return applied


def extract_chapter(chapter: Path, destination: Path, model_path: Path, **settings):
    body_detections = {}
    detections = detect_panels(chapter, model_path, destination,
                               body_detections=body_detections, **settings)
    manifest_path = destination / "panel_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    raw_manifest = build_manifest(detections, previous)
    corrections = _apply_reviewed_panel_corrections(
        detections,
        raw_manifest,
        destination / "panel_review_corrections.json",
        destination,
        settings["jpeg_quality"],
    )
    manifest = build_manifest(detections, previous)
    raw_by_name = {Path(row["page"]).name: row for row in raw_manifest["pages"]}
    for page in manifest["pages"]:
        name = Path(page["page"]).name
        correction = corrections.get(name)
        if not correction:
            continue
        page["detector_panels"] = raw_by_name[name]["panels"]
        page["geometry_review"] = {
            "status": "accepted_reviewed",
            "base_geometry_revision": correction["base_geometry_revision"],
            "reason": correction.get("reason"),
        }
        role = correction.get("page_role")
        if role is not None:
            if role not in {"story", "reference", "extra", "uncertain"}:
                raise ValueError(f"Invalid reviewed page role for {name}.")
            page["page_role"] = role
            page["page_role_source"] = "review"
    panels = crop_panels(detections, destination, jpeg_quality=settings["jpeg_quality"])
    proposals = serialise_proposals(
        body_detections,
        manifest,
        min_score=config.CHARACTER_LABEL_CONFIDENCE,
        duplicate_iou=config.CHARACTER_DUPLICATE_IOU,
        duplicate_containment=config.CHARACTER_DUPLICATE_CONTAINMENT,
        duplicate_min_area_ratio=config.CHARACTER_DUPLICATE_MIN_AREA_RATIO,
        duplicate_aligned_coverage=config.CHARACTER_DUPLICATE_ALIGNED_COVERAGE,
        duplicate_edge_tolerance=config.CHARACTER_DUPLICATE_EDGE_TOLERANCE,
        duplicate_larger_score_margin=config.CHARACTER_DUPLICATE_LARGER_SCORE_MARGIN,
    )
    proposals_by_page = {Path(row["page"]).name: row for row in proposals}
    detections_by_page = {row.page_path.name: row for row in detections}
    for page in manifest["pages"]:
        name = Path(page["page"]).name
        proposal = proposals_by_page.get(name, {"characters": []})
        detection = detections_by_page[name]
        with Image.open(page["page"]) as source:
            grounded_path = save_grounded_page_preview(
                source.convert("RGB"),
                detection.boxes,
                proposal["characters"],
                Path(page["page"]),
                destination,
                jpeg_quality=settings["jpeg_quality"],
                preserve_order=bool(page.get("geometry_review")),
            )
        page["grounded_preview"] = str(grounded_path.resolve())
        page["grounded_preview_hash"] = file_hash(grounded_path)
        page["character_labels"] = [
            {key: character.get(key) for key in (
                "label", "detection_id", "box", "panel_number", "score"
            )}
            for character in proposal["characters"]
        ]
        page["grounding_revision"] = proposal.get("grounding_revision")
    write_json(destination / "character_detections.json", proposals)
    write_json(manifest_path, manifest)
    return panels, sum(len(page["characters"]) for page in proposals)
