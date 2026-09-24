"""Normalize entity proposals and run cached, manga-wide visual matching."""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common.utils import natural_sort_key, list_images
from .matching import resolve, stable_id
from .manifest import file_hash
from .naming import assign_group_display_names


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def collect_instances(manga_dir: Path, chapters: list[Path], output_dir: Path):
    """Combine broad/body and legacy person proposals; never filter by species."""
    instances, crops, warnings = [], [], []
    seen_sources = set()
    for chapter_index, chapter in enumerate(sorted(chapters, key=natural_sort_key)):
        page_order = {path.name: i for i, path in enumerate(list_images(chapter))}
        chapter_output = output_dir / chapter.name
        manifest = read_json(chapter_output / "panel_manifest.json", {})
        page_panels = {Path(p["page"]).name: p for p in manifest.get("pages", [])}
        if not page_panels:
            warnings.append(f"{chapter.name}: no panel_manifest.json; counts stay unknown. Run test_panels.py.")
        if (chapter_output / "character_identity_overrides.json").is_file():
            warnings.append(f"{chapter.name}: legacy A/B identity overrides ignored; review exact instance IDs instead.")
        proposals = {}
        sources = (
            ("character_detections.json", "body"),
            ("person_detections.json", "legacy_person"),
            ("entity_detections.json", "entity"),
        )
        found = False
        for filename, source in sources:
            data = read_json(chapter_output / filename)
            if data is None:
                continue
            found = True
            seen_sources.add(source)
            if source == "body":
                rows = [dict(c, page=p["page"], source_hash=p.get("source_hash"))
                        for p in data for c in p.get("characters", [])]
            else:
                rows = data.get("detections", [])
            for entry in rows:
                name = Path(entry.get("page", "")).name
                path = chapter / name
                if not path.is_file():
                    warnings.append(f"Missing source image: {chapter.name}/{name}")
                    continue
                box = entry.get("box")
                if not isinstance(box, list) or len(box) != 4:
                    warnings.append(f"Invalid box: {chapter.name}/{name}")
                    continue
                try:
                    box = [float(x) for x in box]
                    score = float(entry.get("detector_score", entry.get("score", 0.0)))
                except (ValueError, TypeError):
                    continue
                if not np.isfinite(box + [score]).all():
                    continue
                proposals.setdefault(path, []).append({"box": box, "score": score, "source": source,
                                                       "source_hash": entry.get("source_hash")})
        if not found:
            raise FileNotFoundError(f"No entity proposals for {chapter.name}; run test_panels.py first.")
        for page_index, path in enumerate(sorted(proposals, key=natural_sort_key)):
            digest = file_hash(path)
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                valid = []
                for proposal in sorted(proposals[path], key=lambda p: (-p["score"], p["box"])):
                    if proposal["source_hash"] and proposal["source_hash"] != digest:
                        warnings.append(f"Stale detection ignored: {chapter.name}/{path.name}")
                        continue
                    x1, y1, x2, y2 = proposal["box"]
                    box = [max(0, min(width, round(x1))), max(0, min(height, round(y1))),
                           max(0, min(width, round(x2))), max(0, min(height, round(y2)))]
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    duplicate = next((p for p in valid if intersection(box, p["box"]) /
                                      (area(box) + area(p["box"]) - intersection(box, p["box"])) >= 0.90), None)
                    if duplicate:
                        duplicate["sources"].add(proposal["source"])
                        duplicate["verified_source"] |= proposal["source_hash"] == digest
                        continue
                    valid.append({"box": box, "score": proposal["score"], "sources": {proposal["source"]},
                                  "verified_source": proposal["source_hash"] == digest})
                panel_data = page_panels.get(path.name, {})
                # Do not use stale geometry from a different edition/image.
                if panel_data.get("source_hash") != digest:
                    panel_data = {}
                panels = panel_data.get("panels", [])
                for index, proposal in enumerate(sorted(valid, key=lambda p: p["box"])):
                    box = proposal["box"]
                    matched = sorted(((intersection(box, p["box"]) / area(box), p["panel_number"])
                                      for p in panels), reverse=True)
                    panel = matched[0][1] if matched and matched[0][0] >= .75 and (
                        len(matched) == 1 or matched[0][0] - matched[1][0] >= .15) else None
                    flags = []
                    if not proposal["verified_source"]:
                        flags.append("unverified_detection_source")
                    if min(box[2] - box[0], box[3] - box[1]) < 48 or area(box) < 4096:
                        flags.append("tiny")
                    if proposal["score"] < .65:
                        flags.append("low_detector_score")
                    if panel is None:
                        flags.append("uncertain_panel")
                    panel_record = next((p for p in panels if p["panel_number"] == panel), {})
                    if panel_record.get("provisional", False):
                        flags.append("provisional_panel")
                    if panel_data.get("page_role") != "story":
                        flags.append("non_story_or_unknown_role")
                    contained = [p for p in valid if p is not proposal and p["score"] >= .75
                                 and area(p["box"]) < area(box) * .75
                                 and intersection(box, p["box"]) / area(p["box"]) >= .9]
                    if len(contained) >= 2:
                        flags.append("possible_multiple_beings")
                    elif any(area(p["box"]) < area(box) * .35 for p in contained):
                        # A small independently detected body inside a much larger
                        # crop can dominate its embedding. Do not guess the actor.
                        flags.append("ambiguous_nested_crop")
                    crop = image.crop(box)
                    if float(np.asarray(crop.convert("L").resize((64, 64))).std()) < 8:
                        flags.append("low_detail")
                    instance_id = stable_id("inst_", manga_dir.name, chapter.name, path.name, digest, box)
                    panel_id = (panel_record.get("panel_id") or f"{chapter.name}/{path.name}#panel_{panel}") if panel else None
                    instances.append({"instance_id": instance_id, "chapter": chapter.name,
                                      "page": str(path.resolve()), "page_hash": digest,
                                      "page_id": panel_data.get("page_id"),
                                      "geometry_revision": panel_data.get("geometry_revision"),
                                      "character_index": index + 1, "box": box,
                                      "panel_id": panel_id, "panel_number": panel,
                                      "story_panel_id": stable_id("story_panel_", digest, panel_record["box"]) if panel else None,
                                      "panel_provisional": panel_record.get("provisional", False),
                                      "reading_position": [chapter_index, page_order[path.name], panel] if panel else None,
                                      "page_role": panel_data.get("page_role", "uncertain"),
                                      "page_role_source": panel_data.get("page_role_source", "review"),
                                      "detector_score": proposal["score"], "sources": sorted(proposal["sources"]),
                                      "quality_flags": flags, "prototype_eligible": not flags,
                                      "embedding_eligible": not (set(flags) - {"non_story_or_unknown_role"})})
                    crop.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    crops.append(crop)
    if "entity" not in seen_sources:
        warnings.append("Existing detector proposals are not validated for all creatures. No species filter is applied, but missed creatures need improved detections.")
    return instances, crops, warnings


class Encoder:
    """Load one local encoder lazily; cache features by exact model and crop ID."""
    def __init__(self, model_root: Path, kind="dino", provider="auto"):
        self.kind, self.provider = kind, provider
        if kind == "ccip":
            root = model_root / "CCIP" / "ccip-caformer_b36-24"
            self.feature_path = root / "model_feat.onnx"
            self.metric_path = root / "model_metrics.onnx"
            paths = [self.feature_path, self.metric_path]
            self.default_threshold = .14
            self.assignment_group_threshold = .14
        elif kind == "dino":
            self.feature_path = model_root / "DINOv3" / "dino_v3.onnx"
            self.metric_path = None
            paths = [self.feature_path, self.feature_path.with_suffix(".onnx.data")]
            # Development comparison on Maou-sama: fewer mixed-character
            # groups than CCIP while retaining simplified/detailed variants.
            # This is a proposal cutoff, not calibrated identity acceptance.
            self.default_threshold = .20
            self.assignment_group_threshold = .08
        else:
            raise ValueError("Encoder must be ccip or dino.")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.fingerprint = stable_id("encoder_", kind, "preprocessing_v1_768", [file_hash(p) for p in paths])
        self.feature = self.metric = None
        self.runtime = "cached"

    def _load(self):
        if self.feature is not None:
            return
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        if self.provider != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.feature = ort.InferenceSession(str(self.feature_path), sess_options=options, providers=providers)
        if self.metric_path:
            self.metric = ort.InferenceSession(str(self.metric_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.runtime = self.feature.get_providers()
        print(f"{self.kind}: {self.runtime}", flush=True)

    def preprocess(self, image):
        if self.kind == "ccip":
            image = image.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
            mean, std = [.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711]
        else:
            image = ImageOps.pad(image.convert("RGB"), (448, 448), color=(128, 128, 128))
            mean, std = [.485, .456, .406], [.229, .224, .225]
        array = (np.asarray(image, dtype=np.float32) / 255 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        return np.ascontiguousarray(array.transpose(2, 0, 1))

    def close(self):
        """Release ONNX sessions before the description model is loaded."""
        self.feature = self.metric = None

    def distances(self, instances, crops, cache_dir: Path, batch_size=4):
        cache = cache_dir / self.fingerprint
        cache.mkdir(parents=True, exist_ok=True)
        vectors = [None] * len(instances)
        missing = []
        for i, instance in enumerate(instances):
            path = cache / f"{instance['instance_id']}.npy"
            if path.is_file():
                try:
                    vector = np.load(path, allow_pickle=False)
                    if vector.ndim != 1 or not np.isfinite(vector).all():
                        raise ValueError("Invalid cache")
                    vectors[i] = vector
                    continue
                except (ValueError, OSError):
                    pass
            missing.append(i)
        if missing:
            self._load()
            shape = self.feature.get_inputs()[0].shape
            actual_batch = shape[0] if isinstance(shape[0], int) else batch_size
            for start in range(0, len(missing), actual_batch):
                chunk = missing[start:start + actual_batch]
                tensors = [self.preprocess(crops[i]) for i in chunk]
                if isinstance(shape[0], int):
                    tensors += [tensors[-1]] * (actual_batch - len(tensors))
                output = self.feature.run(None, {self.feature.get_inputs()[0].name: np.stack(tensors)})[0]
                if output.ndim != 2 or output.shape[0] != len(tensors) or not np.isfinite(output).all():
                    raise ValueError(f"Invalid feature tensor: {output.shape}")
                for i, vector in zip(chunk, output):
                    vectors[i] = vector.astype(np.float32)
                    path = cache / f"{instances[i]['instance_id']}.npy"
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, vectors[i], allow_pickle=False)
                    temporary.replace(path)
                print(f"  Embeddings {min(start + actual_batch, len(missing))}/{len(missing)}", flush=True)
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        features = np.stack(vectors)
        if self.kind == "ccip":
            if self.metric is None:
                import onnxruntime as ort
                self.metric = ort.InferenceSession(str(self.metric_path), providers=["CPUExecutionProvider"])
            distances = np.asarray(self.metric.run(None, {self.metric.get_inputs()[0].name: features})[0], dtype=np.float32)
        else:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            if np.any(norms < 1e-8):
                raise ValueError("Zero embedding returned by DINO.")
            features = features / norms
            distances = 1 - features @ features.T
        distances = np.maximum((distances + distances.T) / 2, 0)
        np.fill_diagonal(distances, 0)
        return distances


def load_calibration(path: Path | None, encoder: Encoder):
    if path is None:
        return None
    return validate_calibration(read_json(path), encoder.fingerprint)


def validate_calibration(data, fingerprint):
    if not isinstance(data, dict) or data.get("encoder_fingerprint") != fingerprint:
        raise ValueError("Calibration must match this exact model and preprocessing fingerprint.")
    if data.get("validated") is not True or not isinstance(data.get("evaluation_reference"), str) or not data["evaluation_reference"]:
        raise ValueError("Calibration requires validated=true and an evaluation_reference.")
    for key in ("max_distance", "min_margin"):
        if type(data.get(key)) not in (float, int) or not np.isfinite(data[key]) or data[key] < 0:
            raise ValueError(f"Calibration requires a finite nonnegative {key}.")
    return data


def prepare_manga(manga_dir: Path, chapters: list[Path], output_dir: Path, *, encoder: Encoder,
                  cache_dir: Path):
    """Run the encoder before Qwen; return reusable CPU-side inputs."""
    start = time.perf_counter()
    instances, crops, warnings = collect_instances(manga_dir, chapters, output_dir)
    # Unusable crops stay in the output/gallery but cannot support any visual
    # decision. Avoid spending encoder time on them.
    eligible = [i for i, item in enumerate(instances) if item.get("embedding_eligible", item["prototype_eligible"])]
    distances = np.full((len(instances), len(instances)), 1e6, dtype=np.float32)
    np.fill_diagonal(distances, 0)
    if eligible:
        distances[np.ix_(eligible, eligible)] = encoder.distances(
            [instances[i] for i in eligible], [crops[i] for i in eligible], cache_dir)
    return {"instances": instances, "crops": crops, "distances": distances, "warnings": warnings,
            "manga": manga_dir.name, "chapters": [p.name for p in sorted(chapters, key=natural_sort_key)],
            "encoder": encoder.kind, "encoder_fingerprint": encoder.fingerprint,
            "provider": encoder.runtime, "group_threshold": encoder.default_threshold,
            "assignment_group_threshold": getattr(encoder, "assignment_group_threshold", .14),
            "elapsed_seconds": time.perf_counter() - start}


def add_style_evidence(prepared: dict, encoder: Encoder, cache_dir: Path) -> dict:
    """Add a second cached distance matrix used only for proposed style links."""
    if encoder.fingerprint == prepared["encoder_fingerprint"]:
        raise ValueError("Style linker must use a different encoder from the primary encoder.")
    instances, crops = prepared["instances"], prepared["crops"]
    eligible = [i for i, item in enumerate(instances)
                if item.get("embedding_eligible", item["prototype_eligible"])]
    distances = np.full((len(instances), len(instances)), 1e6, dtype=np.float32)
    np.fill_diagonal(distances, 0)
    if eligible:
        distances[np.ix_(eligible, eligible)] = encoder.distances(
            [instances[i] for i in eligible], [crops[i] for i in eligible], cache_dir)
    prepared["style_distances"] = distances
    prepared["style_encoder"] = encoder.kind
    prepared["style_encoder_fingerprint"] = encoder.fingerprint
    prepared["style_provider"] = encoder.runtime
    return prepared


def resolve_prepared(prepared: dict, *, reviews=None, calibration=None, transformations=None):
    """Rebuild decisions using current reviews; never load any model here."""
    start = time.perf_counter()
    instances = deepcopy(prepared["instances"])
    distances = prepared["distances"]
    warnings = list(prepared["warnings"])
    if calibration is not None:
        validate_calibration(calibration, prepared["encoder_fingerprint"])
    if transformations:
        if transformations.get("schema_version") != 1:
            raise ValueError("Unsupported transformation schema.")
        for instance in instances:
            data = transformations.get("chapters", {}).get(instance["chapter"], {}).get(Path(instance["page"]).name, {})
            if (instance.get("page_role_source") in ("unknown", "description")
                    and data.get("extraction_status") == "complete"
                    and data.get("source_hash") == instance.get("page_hash")
                    and data.get("geometry_revision") == instance.get("geometry_revision")):
                role = data.get("page_role", "uncertain")
                if role not in ("story", "reference", "extra", "uncertain"):
                    role = "uncertain"
                instance["page_role"] = role
                flags = set(instance["quality_flags"]) - {"non_story_or_unknown_role"}
                if role != "story":
                    flags.add("non_story_or_unknown_role")
                instance["quality_flags"] = sorted(flags)
                instance["prototype_eligible"] = not flags
    result = resolve(instances, distances, reviews=reviews,
                     group_threshold=prepared["group_threshold"], calibration=calibration,
                     assignment_group_threshold=prepared.get("assignment_group_threshold", .14),
                     style_distances=prepared.get("style_distances"))
    result["warnings"] = warnings + result["warnings"]
    assign_group_display_names(result, prepared["manga"])
    if calibration is None:
        result["warnings"].append("No validated calibration: visual groups are proposals. Only reviewed instances receive canonical identities.")
    events = []
    if transformations:
        if transformations.get("schema_version") != 1:
            raise ValueError("Unsupported transformation schema.")
        for chapter, pages in transformations.get("chapters", {}).items():
            for page, data in pages.items():
                if data.get("extraction_status") != "complete":
                    continue
                for event in data.get("events", []):
                    proposal = dict(event)
                    provenance = bool(data.get("source_hash") and data.get("geometry_revision"))
                    matching_instances = [x for x in instances if x["chapter"] == chapter
                                          and Path(x["page"]).name == page]
                    current = provenance and any(x.get("page_hash") == data["source_hash"]
                                                 and x.get("geometry_revision") == data["geometry_revision"]
                                                 for x in matching_instances)
                    proposal.update(chapter=chapter, page=page, status="pending_review",
                                    source_hash=data.get("source_hash"), geometry_revision=data.get("geometry_revision"),
                                    grounding_status="current" if current else "stale_or_unverified")
                    proposal["candidate_instances"] = {
                        side: [x["instance_id"] for x in matching_instances if current
                               and x["panel_number"] == event.get(side + "_panel")]
                        for side in ("source", "target")}
                    proposal["event_id"] = stable_id("event_", chapter, page, data.get("source_hash"),
                                                    data.get("geometry_revision"), event)
                    events.append(proposal)
    result.update(schema_version=2, manga=prepared["manga"], encoder=prepared["encoder"],
                  encoder_fingerprint=prepared["encoder_fingerprint"], provider=prepared["provider"],
                  style_linker={"encoder": prepared.get("style_encoder"),
                                "encoder_fingerprint": prepared.get("style_encoder_fingerprint"),
                                "provider": prepared.get("style_provider"),
                                "policy": "dual_encoder_anchor_v2" if prepared.get("style_distances") is not None else None},
                  acceptance_policy={"rule_version": "resolver_v5",
                                     "proposal_distance": prepared["group_threshold"],
                                     "assignment_group_distance": prepared.get("assignment_group_threshold", .14),
                                     "calibration": calibration},
                  input_chapters=prepared["chapters"], transformation_proposals=events,
                  elapsed_seconds=round(prepared["elapsed_seconds"] + time.perf_counter() - start, 3))
    result["revision"] = stable_id("revision_", "resolver_v5", prepared["encoder_fingerprint"],
                                   prepared["group_threshold"], reviews, calibration,
                                   prepared.get("assignment_group_threshold", .14),
                                   prepared.get("style_encoder_fingerprint"), result["style_links"],
                                   events,
                                   [(x["instance_id"], x["panel_id"], x["quality_flags"],
                                             x["page_role"], x["reading_position"]) for x in instances],
                                   result["characters"], result["identities"], result["visual_groups"])
    return result


def classify_manga(manga_dir: Path, chapters: list[Path], output_dir: Path, *, encoder: Encoder,
                   cache_dir: Path, reviews=None, calibration=None, transformations=None,
                   style_encoder: Encoder | None = None):
    """Return decisions without classification exports; reuse cached embeddings."""
    if calibration is not None:
        validate_calibration(calibration, encoder.fingerprint)
    try:
        prepared = prepare_manga(manga_dir, chapters, output_dir, encoder=encoder, cache_dir=cache_dir)
        if style_encoder is not None:
            add_style_evidence(prepared, style_encoder, cache_dir)
    finally:
        if hasattr(encoder, "close"):
            encoder.close()
        if style_encoder is not None and hasattr(style_encoder, "close"):
            style_encoder.close()
    result = resolve_prepared(prepared, reviews=reviews, calibration=calibration,
                              transformations=transformations)
    return result, prepared["crops"], prepared["distances"]
