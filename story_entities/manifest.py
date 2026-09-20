"""Original-page provenance; no source image is modified or deleted."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from .matching import stable_id


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                      "preview_hash": file_hash(path_preview) if (
                          path_preview := getattr(detection, "preview_path", None)) else None})
    return {"schema_version": 2, "pages": pages}


def serialise_proposals(body_detections: dict, manifest: dict) -> list:
    pages = {Path(p["page"]).name: p for p in manifest["pages"]}
    return [{"page": str(path.resolve()), "source_hash": pages[path.name]["source_hash"],
             "characters": [{"index": i, "box": [int(round(float(v))) for v in d.box],
                             "score": float(d.score)} for i, d in enumerate(items, start=1)]}
            for path, items in body_detections.items()]
