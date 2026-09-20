"""Shared panel-stage runner: derived crops, manifests and body proposals."""
from __future__ import annotations

import json
from pathlib import Path

from common.json_io import write_json
from story_entities.manifest import build_manifest, serialise_proposals
from .pipeline import detect_panels, crop_panels


def extract_chapter(chapter: Path, destination: Path, model_path: Path, **settings):
    body_detections = {}
    detections = detect_panels(chapter, model_path, destination,
                               body_detections=body_detections, **settings)
    manifest_path = destination / "panel_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    manifest = build_manifest(detections, previous)
    panels = crop_panels(detections, destination, jpeg_quality=settings["jpeg_quality"])
    write_json(destination / "character_detections.json", serialise_proposals(body_detections, manifest))
    write_json(manifest_path, manifest)
    return panels, sum(map(len, body_detections.values()))
