"""Normalize entity proposals and run cached, manga-wide visual matching."""
from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config
from description.quality import action_issues, character_description_is_useful, text_issues
from common.utils import natural_sort_key, list_images
from .matching import resolve, stable_id, story_panel
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
                rows = [dict(c, page=p["page"], source_hash=p.get("source_hash"),
                             geometry_revision=p.get("geometry_revision"),
                             grounding_revision=p.get("grounding_revision"))
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
                                                       "source_hash": entry.get("source_hash"),
                                                       "grounding_revision": entry.get("grounding_revision"),
                                                       "grounding_id": entry.get("detection_id"),
                                                       "grounding_label": entry.get("label")})
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
                        if proposal.get("grounding_revision"):
                            duplicate["grounding_revision"] = proposal["grounding_revision"]
                        if isinstance(proposal.get("grounding_id"), str):
                            duplicate["grounding_ids"].add(proposal["grounding_id"])
                        if isinstance(proposal.get("grounding_label"), str):
                            duplicate["grounding_labels"].add(proposal["grounding_label"])
                        continue
                    valid.append({"box": box, "score": proposal["score"], "sources": {proposal["source"]},
                                  "verified_source": proposal["source_hash"] == digest,
                                  "grounding_revision": proposal.get("grounding_revision"),
                                  "grounding_ids": ({proposal["grounding_id"]}
                                                    if isinstance(proposal.get("grounding_id"), str) else set()),
                                  "grounding_labels": ({proposal["grounding_label"]}
                                                       if isinstance(proposal.get("grounding_label"), str) else set())})
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
                    if proposal["score"] < config.CHARACTER_LABEL_CONFIDENCE:
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
                                      "grounding_revision": proposal.get("grounding_revision"),
                                      "character_index": index + 1, "box": box,
                                      "panel_id": panel_id, "panel_number": panel,
                                      "story_panel_id": stable_id("story_panel_", digest, panel_record["box"]) if panel else None,
                                      "panel_provisional": panel_record.get("provisional", False),
                                      "reading_position": [chapter_index, page_order[path.name], panel] if panel else None,
                                      "page_role": panel_data.get("page_role", "uncertain"),
                                      "page_role_source": panel_data.get("page_role_source", "review"),
                                      "detector_score": proposal["score"], "sources": sorted(proposal["sources"]),
                                      "grounding_ids": sorted(proposal["grounding_ids"]),
                                      "grounding_labels": sorted(proposal["grounding_labels"]),
                                      "quality_flags": flags, "prototype_eligible": not flags,
                                      "embedding_eligible": not (set(flags) - {"non_story_or_unknown_role"}),
                                      "narration_eligibility": "uncertain",
                                      "narration_eligibility_source": "unreviewed_body_proposal"})
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


_AUTO_IDENTITY_EVENT_TYPES = {"transformation", "reversion", "disguise_reveal"}


def _build_transformation_proposals(instances: list[dict], transformations: dict | None) -> list[dict]:
    """Ground model events to current instance candidates without accepting them."""
    events = []
    if not transformations:
        return events
    for chapter, pages in transformations.get("chapters", {}).items():
        for page, data in pages.items():
            if data.get("extraction_status") != "complete":
                continue
            matching = [row for row in instances
                        if row["chapter"] == chapter and Path(row["page"]).name == page]
            provenance = bool(data.get("source_hash") and data.get("geometry_revision"))
            current = provenance and any(
                row.get("page_hash") == data["source_hash"]
                and row.get("geometry_revision") == data["geometry_revision"]
                and row.get("grounding_revision") == data.get("grounding_revision")
                for row in matching
            )
            for event in data.get("events", []):
                proposal = dict(event)
                proposal.update(
                    chapter=chapter,
                    page=page,
                    status="pending_review",
                    source_hash=data.get("source_hash"),
                    geometry_revision=data.get("geometry_revision"),
                    grounding_revision=data.get("grounding_revision"),
                    grounding_status="current" if current else "stale_or_unverified",
                )
                exact_grounding = all(
                    isinstance(event.get(side + "_detection_id"), str)
                    for side in ("source", "target")
                )
                if exact_grounding:
                    proposal["candidate_instances"] = {
                        side: [
                            row["instance_id"] for row in matching
                            if current and event[side + "_detection_id"] in row.get("grounding_ids", [])
                            and row["panel_number"] == event.get(side + "_panel")
                        ]
                        for side in ("source", "target")
                    }
                    proposal["grounding_method"] = "labeled_detection"
                else:
                    proposal["candidate_instances"] = {
                        side: [
                            row["instance_id"] for row in matching
                            if current and row["panel_number"] == event.get(side + "_panel")
                        ]
                        for side in ("source", "target")
                    }
                    proposal["grounding_method"] = "legacy_panel_candidates"
                proposal["event_id"] = stable_id(
                    "event_", chapter, page, data.get("source_hash"),
                    data.get("geometry_revision"), data.get("grounding_revision"), event,
                )
                events.append(proposal)
    return events


def _automatic_transformation_assignments(
    result: dict,
    events: list[dict],
    reviews: dict | None,
) -> dict:
    """Accept only uniquely grounded, direct, 100%-certain visible changes."""
    by_id = {row["instance_id"]: row for row in result["characters"]}
    blocked = {
        frozenset((entry.get("left"), entry.get("right")))
        for entry in (reviews or {}).get("different", [])
        if isinstance(entry, dict) and entry.get("left") and entry.get("right")
    }
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    accepted_events = []
    for event in sorted(events, key=lambda row: row["event_id"]):
        reason = None
        candidates = event.get("candidate_instances", {})
        source_ids = candidates.get("source", [])
        target_ids = candidates.get("target", [])
        source_panel = event.get("source_panel")
        target_panel = event.get("target_panel")
        if event.get("type") not in _AUTO_IDENTITY_EVENT_TYPES:
            reason = "event_type_does_not_prove_same_identity"
        elif event.get("grounding_status") != "current":
            reason = "stale_or_unverified_grounding"
        elif event.get("confidence_percent") != 100:
            reason = "model_certainty_below_100"
        elif event.get("evidence_type") != "visible_change":
            reason = "direct_visible_change_required"
        elif (type(source_panel) is not int or type(target_panel) is not int
              or abs(source_panel - target_panel) != 1):
            reason = "adjacent_endpoint_panels_required"
        elif len(source_ids) != 1 or len(target_ids) != 1:
            reason = "exactly_one_instance_per_endpoint_required"
        else:
            source_id, target_id = source_ids[0], target_ids[0]
            source, target = by_id.get(source_id), by_id.get(target_id)
            if source is None or target is None or source_id == target_id:
                reason = "distinct_known_endpoints_required"
            elif not source.get("prototype_eligible") or not target.get("prototype_eligible"):
                reason = "high_quality_endpoints_required"
            elif source.get("page_role") != "story" or target.get("page_role") != "story":
                reason = "story_page_endpoints_required"
            else:
                left_root, right_root = find(source_id), find(target_id)
                members = {
                    identity for identity in parent
                    if find(identity) in (left_root, right_root)
                }
                members.update((source_id, target_id))
                entities = {by_id[identity].get("entity_id") for identity in members
                            if by_id[identity].get("entity_id")}
                if len(entities) > 1:
                    reason = "conflicting_accepted_entities"
                elif any(pair.issubset(members) for pair in blocked):
                    reason = "reviewed_different_identity_constraint"
                else:
                    parent[right_root] = left_root
                    accepted_events.append(event)

        if reason is not None:
            event["automatic_rejection_reason"] = reason
            continue
        event.update(
            status="accepted_automatic",
            acceptance_policy="qwen_direct_visible_change_v1",
            accepted_instance_ids=[source_ids[0], target_ids[0]],
        )

    if not accepted_events:
        return {}

    components: dict[str, set[str]] = {}
    for identity in parent:
        components.setdefault(find(identity), set()).add(identity)
    assignments = {}
    for members in components.values():
        supporting = [event for event in accepted_events
                      if set(event["accepted_instance_ids"]).issubset(members)]
        if not supporting:
            continue
        entities = {by_id[identity].get("entity_id") for identity in members
                    if by_id[identity].get("entity_id")}
        entity_id = (next(iter(entities)) if entities else
                     stable_id("entity_qwen_", sorted(members)))
        event_ids = sorted(event["event_id"] for event in supporting)
        evidence = "Qwen 100% direct visible change: " + ", ".join(event_ids)
        for identity in sorted(members):
            row = by_id[identity]
            assignments[identity] = {
                "entity_id": entity_id,
                "form_id": row.get("form_id") or stable_id(
                    "form_qwen_", row.get("visual_group_id") or identity
                ),
                "evidence": evidence,
                "event_ids": event_ids,
            }
    return assignments


def _grounding_index(characters: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for character in characters:
        for detection_id in character.get("grounding_ids", []):
            index.setdefault(detection_id, []).append(character)
    return index


def _current_description_page(data: dict, matching: list[dict]) -> bool:
    return bool(
        data.get("extraction_status") in {"complete", "complete_with_uncertainty"}
        and data.get("source_hash")
        and data.get("geometry_revision")
        and any(
            row.get("page_hash") == data["source_hash"]
            and row.get("geometry_revision") == data["geometry_revision"]
            and row.get("grounding_revision") == data.get("grounding_revision")
            for row in matching
        )
    )


def _apply_grounded_descriptions(result: dict, descriptions: dict | None) -> None:
    """Attach Qwen appearance phrases through exact labeled detections."""
    if not descriptions:
        return
    characters = result.get("characters", [])
    by_grounding = _grounding_index(characters)
    for chapter, pages in descriptions.items():
        if not isinstance(pages, dict):
            continue
        for page, data in pages.items():
            if not isinstance(data, dict):
                continue
            matching = [row for row in characters
                        if row.get("chapter") == chapter and Path(row.get("page", "")).name == page]
            if not _current_description_page(data, matching):
                continue
            panel_context = {
                panel.get("panel_number"): panel.get("description")
                for panel in data.get("panels", []) if isinstance(panel, dict)
            }
            for row in matching:
                context = panel_context.get(row.get("panel_number"))
                if isinstance(context, str) and context.strip():
                    row["panel_description_context"] = context
            for descriptor in data.get("characters", []):
                detection_id = descriptor.get("detection_id")
                candidates = [row for row in by_grounding.get(detection_id, []) if row in matching]
                visual = descriptor.get("visual_description")
                if len(candidates) != 1 or not isinstance(visual, str) or not visual.strip():
                    continue
                candidates[0].update(
                    visual_description=visual,
                    visual_description_status="qwen_labeled_observation",
                    visual_description_detection_id=detection_id,
                )
                representation = descriptor.get("representation_type")
                if representation in {"live_character", "toy", "statue", "image", "uncertain"}:
                    candidates[0].update(
                        representation_type=representation,
                        representation_status=descriptor.get(
                            "representation_status", "model_observation",
                        ),
                    )
                if candidates[0].get("narration_eligibility") in (None, "uncertain"):
                    candidates[0].update(
                        narration_eligibility="eligible_automatic",
                        narration_eligibility_source="body_observation_format_v2",
                    )

def _apply_eligibility_reviews(result: dict, reviews: dict | None) -> None:
    """Apply exact, source-bound narration decisions without deleting instances."""
    by_id = {row["instance_id"]: row for row in result.get("characters", [])}
    for correction in (reviews or {}).get("appearance_corrections", []):
        if (not isinstance(correction, dict)
                or correction.get("status") != "accepted_reviewed"):
            raise ValueError("Appearance correction must be accepted_reviewed.")
        row = by_id.get(correction.get("instance_id"))
        text = correction.get("text")
        if (row is None or not _review_source_matches(correction, row)
                or not isinstance(text, str)
                or not character_description_is_useful(text)):
            raise ValueError("Appearance correction has stale or invalid evidence.")
        row.update(
            visual_description=text.strip(),
            visual_description_status="reviewed_observation",
        )
    seen = set()
    for decision in (reviews or {}).get("eligibility", []):
        if not isinstance(decision, dict):
            raise ValueError("Eligibility review must be an object.")
        instance_id = decision.get("instance_id")
        row = by_id.get(instance_id)
        value = decision.get("decision")
        if instance_id in seen or row is None:
            raise ValueError("Eligibility review references a missing or duplicate instance.")
        seen.add(instance_id)
        if value not in {"eligible", "uncertain", "rejected_non_character"}:
            raise ValueError("Eligibility review has an invalid decision.")
        if any(decision.get(key) != row.get(row_key) for key, row_key in (
            ("source_hash", "page_hash"),
            ("geometry_revision", "geometry_revision"),
            ("grounding_revision", "grounding_revision"),
        )):
            raise ValueError(f"Stale eligibility review for {instance_id}.")
        row.update(
            narration_eligibility=("reviewed_eligible" if value == "eligible" else value),
            narration_eligibility_source="reviewed_narration_eligibility_v1",
            narration_eligibility_reason=decision.get("reason"),
        )


def _apply_stable_descriptions(result: dict) -> None:
    """Share descriptions only across accepted entity/form membership."""
    characters = result.get("characters", [])
    confirmed = {}
    for row in characters:
        if (row.get("entity_id") and row.get("form_id")
                and row.get("identity_status") != "unknown"):
            confirmed.setdefault((row["entity_id"], row["form_id"]), []).append(row)
    for rows in confirmed.values():
        candidates = [row for row in rows if row.get("visual_description")]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (
            row.get("visual_description_status") != "reviewed_observation",
            row.get("reading_position") or [10**9], row["instance_id"],
        ))
        stable = candidates[0]["visual_description"]
        source = candidates[0].get("visual_description_status")
        for row in rows:
            row.update(stable_description=stable, stable_description_source=source)
    for row in characters:
        if not row.get("stable_description") and row.get("visual_description"):
            row.update(
                stable_description=row["visual_description"],
                stable_description_source=row.get("visual_description_status"),
            )


_REPRESENTATION_PATTERNS = {
    "toy": re.compile(
        r"\b(?:toy|figurine|action figure|miniature|doll|plush|capsule figure|"
        r"small figure of|model figure)\b", re.IGNORECASE,
    ),
    "statue": re.compile(r"\b(?:statue|sculpture|mannequin)\b", re.IGNORECASE),
    "image": re.compile(
        r"\b(?:photograph|photo of|portrait of|poster|printed image|on (?:a )?screen|"
        r"on (?:the )?packag(?:e|ing))\b", re.IGNORECASE,
    ),
}


def _classify_representation_types(result: dict) -> None:
    """Separate visible people from toys and other depictions before identity promotion.

    This deliberately abstains when no grounded character description exists. A
    representation may resemble a character, but resemblance is not evidence
    that the represented object is the live narrative actor.
    """
    for row in result.get("characters", []):
        description = row.get("visual_description") or ""
        lexical = next(
            (kind for kind, pattern in _REPRESENTATION_PATTERNS.items()
             if pattern.search(description)),
            None,
        )
        observed = row.get("representation_type")
        context = row.get("panel_description_context") or ""
        contextual_toy = bool(
            re.search(r"\b(?:toy|figurine|doll|capsule figure)s?\b", context, re.IGNORECASE)
            and re.search(r"\b(?:figure|armou?red|miniature|small)\b", description, re.IGNORECASE)
        )
        representation = (
            lexical or ("toy" if contextual_toy else None)
            or (observed if observed in {
                "live_character", "toy", "statue", "image", "uncertain",
            } else None)
            or ("live_character" if description.strip() else "uncertain")
        )
        row["representation_type"] = representation
        row["representation_policy"] = (
            "grounded_appearance_representation_v2"
            if observed in {"live_character", "toy", "statue", "image", "uncertain"}
            else "grounded_appearance_representation_v1_fallback"
        )


def _box_iou(left, right) -> float:
    if not (isinstance(left, list) and isinstance(right, list)
            and len(left) == len(right) == 4):
        return 0.0
    overlap = intersection(left, right)
    union = area(left) + area(right) - overlap
    return overlap / union if union > 0 else 0.0


def _appearance_traits(text: str | None) -> dict[str, str]:
    """Extract only explicit contradictions; absence of a trait proves nothing."""
    words = set(re.findall(r"[a-z]+", str(text or "").lower()))
    traits = {}
    if words & {"girl", "woman", "female"}:
        traits["presentation"] = "female"
    elif words & {"boy", "man", "male"}:
        traits["presentation"] = "male"
    colours = {
        "dark": {"dark", "black"}, "light": {"light", "white", "blonde", "blond"},
        "red": {"red"}, "blue": {"blue"}, "green": {"green"},
    }
    mentioned = [name for name, values in colours.items() if words & values]
    if len(mentioned) == 1 and "hair" in words:
        traits["hair_colour"] = mentioned[0]
    return traits


def _automatic_identity_conflict(left: dict, right: dict) -> bool:
    if left.get("representation_type") != right.get("representation_type"):
        return True
    # Manga panels can depict the same character more than once (for example a
    # full body beside an inset close-up), so co-presence alone is not a safe
    # cannot-link. Only explicit grounded trait contradictions are used here.
    a = _appearance_traits(left.get("visual_description"))
    b = _appearance_traits(right.get("visual_description"))
    return any(key in a and key in b and a[key] != b[key] for key in set(a) | set(b))


def _promote_automatic_identities(result: dict, distances: np.ndarray, *,
                                  proposal_threshold: float,
                                  strict_threshold: float,
                                  style_distances: np.ndarray | None = None) -> None:
    """Confirm recurrent, contradiction-free cores inside visual proposals.

    A display proposal is not accepted wholesale. Toys and uncertain instances
    are removed first, then a two-neighbour core is built from independently
    agreeing primary/style edges. This prevents one weak bridge from turning a
    chain of lookalikes into a canonical identity.
    """
    rows = result.get("characters", [])
    if not rows:
        return
    eligible = {
        index for index, row in enumerate(rows)
        if row.get("entity_id") is None
        and row.get("prototype_eligible")
        and row.get("narration_eligibility") in {"eligible_automatic", "reviewed_eligible"}
        and row.get("representation_type") == "live_character"
        and row.get("visual_description")
    }
    index_by_id = {row["instance_id"]: index for index, row in enumerate(rows)}
    accepted = []
    for proposal in result.get("visual_groups", []):
        proposal_indices = [
            index_by_id[instance_id] for instance_id in proposal.get("instance_ids", [])
            if instance_id in index_by_id
        ]
        # Reviewed/transformation identities take precedence. Do not create a
        # second automatic entity beside accepted membership in one proposal.
        if any(rows[index].get("entity_id") is not None for index in proposal_indices):
            continue
        described = [index for index in proposal_indices
                     if rows[index].get("narration_eligibility") in {
                         "eligible_automatic", "reviewed_eligible",
                     }
                     and rows[index].get("visual_description")]
        members = [index for index in described if index in eligible]
        if not described or len(members) / len(described) < .7:
            continue
        edge_threshold = proposal_threshold if style_distances is not None else strict_threshold
        adjacency = {index: set() for index in members}
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                if (_automatic_identity_conflict(rows[left], rows[right])
                        or float(distances[left, right]) > edge_threshold):
                    continue
                if (style_distances is not None
                        and float(style_distances[left, right]) > .18):
                    continue
                adjacency[left].add(right)
                adjacency[right].add(left)
        # Iterative two-neighbour pruning gives every accepted occurrence more
        # than one independent visual bridge and removes dangling satellites.
        core = set(members)
        while True:
            remove = {
                index for index in core
                if len({story_panel(rows[other]) for other in adjacency[index] & core
                        if story_panel(rows[other]) != story_panel(rows[index])}) < 2
            }
            if not remove:
                break
            core -= remove
        components = []
        unseen = set(core)
        while unseen:
            seed = min(unseen, key=lambda index: rows[index]["instance_id"])
            component, stack = set(), [seed]
            while stack:
                index = stack.pop()
                if index in component:
                    continue
                component.add(index)
                stack.extend((adjacency[index] & core) - component)
            unseen -= component
            components.append(sorted(component, key=lambda index: rows[index]["instance_id"]))
        for members in components:
            if any(_automatic_identity_conflict(rows[left], rows[right])
                   for offset, left in enumerate(members) for right in members[offset + 1:]):
                continue
            panels = {story_panel(rows[index]) for index in members if story_panel(rows[index])}
            if len(members) < 3 or len(panels) < 3:
                continue
            internal = [float(distances[i, j]) for offset, i in enumerate(members)
                        for j in members[offset + 1:] if j in adjacency[i]]
            if not internal:
                continue
            trait_panels = {}
            for index in members:
                for key, value in _appearance_traits(rows[index].get("visual_description")).items():
                    trait_panels.setdefault((key, value), set()).add(story_panel(rows[index]))
            corroborated_traits = sorted(
                f"{key}:{value}" for (key, value), supporting_panels in trait_panels.items()
                if len(supporting_panels - {None}) >= 2
            )
            if not corroborated_traits:
                continue
            style_support = None
            if style_distances is not None:
                style_values = [float(style_distances[i, j]) for offset, i in enumerate(members)
                                for j in members[offset + 1:] if j in adjacency[i]]
                if not style_values:
                    continue
                style_support = round(float(np.quantile(style_values, .75)), 6)
            entity_id = stable_id("entity_auto_", sorted(rows[i]["instance_id"] for i in members))
            form_id = stable_id("form_auto_", entity_id, "default")
            evidence = {
                "policy": "multi_signal_identity_v1",
                "primary_edge_max_distance": round(max(internal), 6),
                "style_distance_q75": style_support,
                "independent_panel_count": len(panels),
                "representation_type": "live_character",
                "corroborated_appearance_traits": corroborated_traits,
                "source_visual_group_id": proposal.get("group_id"),
            }
            for index in members:
                rows[index].update(
                    entity_id=entity_id, form_id=form_id,
                    identity_status="accepted_automatic",
                    assignment_evidence=[evidence],
                    decision_ids=[stable_id("decision_", entity_id, rows[index]["instance_id"], evidence)],
                    appearance_count=len(panels),
                )
            accepted.append({
                "entity_id": entity_id, "name": None,
                "appearance_count": len(panels), "recurrence": "recurring",
                "name_record": None, "naming_eligible": len(panels) >= 4,
                "form_ids": [form_id],
                "instance_ids": [rows[i]["instance_id"] for i in members],
                "acceptance_policy": "multi_signal_identity_v1",
                "acceptance_evidence": evidence,
            })
    result["identities"].extend(accepted)


def _build_story_events(result: dict, descriptions: dict | None) -> list[dict]:
    """Resolve labeled visible actions to exact current ReID instances."""
    if not descriptions:
        return []
    characters = result.get("characters", [])
    by_grounding = _grounding_index(characters)
    events = []
    for chapter, pages in descriptions.items():
        if not isinstance(pages, dict):
            continue
        for page, data in pages.items():
            if not isinstance(data, dict):
                continue
            matching = [row for row in characters
                        if row.get("chapter") == chapter and Path(row.get("page", "")).name == page]
            current = _current_description_page(data, matching)
            for panel in data.get("panels", []):
                panel_number = panel.get("panel_number")
                for action in panel.get("actions", []):
                    actor_detection = action.get("actor_detection_id")
                    target_detection = action.get("target_detection_id")
                    actor_candidates = [
                        row for row in by_grounding.get(actor_detection, [])
                        if row in matching and row.get("panel_number") == panel_number
                    ] if current and actor_detection else []
                    target_candidates = [
                        row for row in by_grounding.get(target_detection, [])
                        if row in matching and row.get("panel_number") == panel_number
                    ] if current and target_detection else []
                    event = {
                        "event_id": stable_id(
                            "story_event_", chapter, page, data.get("source_hash"),
                            data.get("geometry_revision"), data.get("grounding_revision"),
                            panel_number, action,
                        ),
                        "type": "visible_action",
                        "chapter": chapter,
                        "page": page,
                        "panel_number": panel_number,
                        "panel_id": actor_candidates[0].get("panel_id") if len(actor_candidates) == 1 else None,
                        "text": action.get("text"),
                        "actor_detection_id": actor_detection,
                        "target_detection_id": target_detection,
                        "status": "pending_review",
                        "grounding_status": "current" if current else "stale_or_unverified",
                    }
                    reason = None
                    if not current:
                        reason = "stale_or_unverified_grounding"
                    elif not actor_detection:
                        reason = action.get("grounding_status") or "actor_detection_missing"
                    elif len(actor_candidates) != 1:
                        reason = "exact_actor_instance_required"
                    elif target_detection and len(target_candidates) != 1:
                        reason = "exact_target_instance_required"
                    elif actor_candidates[0].get("narration_eligibility") not in {
                        "eligible_automatic", "reviewed_eligible",
                    }:
                        reason = "actor_not_narration_eligible"
                    elif target_candidates and target_candidates[0].get(
                        "narration_eligibility"
                    ) not in {"eligible_automatic", "reviewed_eligible"}:
                        reason = "target_not_narration_eligible"
                    elif action_issues(action.get("text", "")):
                        reason = "invalid_action_phrase"
                    elif (target_candidates
                          and actor_candidates[0]["instance_id"] != target_candidates[0]["instance_id"]
                          and actor_candidates[0].get("entity_id") is not None
                          and actor_candidates[0].get("entity_id") == target_candidates[0].get("entity_id")):
                        reason = "same_identity_occurrences_require_review"
                    else:
                        event.update(
                            status="accepted_automatic",
                            acceptance_policy="grounded_visible_action_format_v3",
                            actor_instance_id=actor_candidates[0]["instance_id"],
                            target_instance_id=(
                                target_candidates[0]["instance_id"]
                                if target_candidates else None
                            ),
                            actor_description=actor_candidates[0].get(
                                "stable_description"
                            ) or actor_candidates[0].get("visual_description"),
                            target_description=(
                                target_candidates[0].get("stable_description")
                                or target_candidates[0].get("visual_description")
                                if target_candidates else None
                            ),
                        )
                    if reason:
                        event["automatic_rejection_reason"] = reason
                    events.append(event)
    return events


def _review_source_matches(review: dict, row: dict, prefix: str = "") -> bool:
    fields = (
        ((prefix + "page_hash", "page_hash"),
         (prefix + "geometry_revision", "geometry_revision"),
         (prefix + "grounding_revision", "grounding_revision"))
        if prefix else
        (("source_hash", "page_hash"),
         ("geometry_revision", "geometry_revision"),
         ("grounding_revision", "grounding_revision"))
    )
    return all(review.get(key) == row.get(row_key) for key, row_key in fields)


def _apply_reviewed_events(result: dict, reviews: dict | None, events: list[dict],
                           transformations: list[dict]) -> None:
    """Import explicit current action/transformation endpoints from one sidecar."""
    reviews = reviews or {}
    by_id = {row["instance_id"]: row for row in result.get("characters", [])}
    for review in reviews.get("reviewed_actions", []):
        if not isinstance(review, dict) or review.get("status") != "accepted_reviewed":
            raise ValueError("Reviewed action must be an accepted_reviewed object.")
        actor = by_id.get(review.get("actor_instance_id"))
        target = by_id.get(review.get("target_instance_id")) if review.get("target_instance_id") else None
        text = review.get("text")
        if (actor is None or not _review_source_matches(review, actor)
                or actor.get("panel_id") != review.get("panel_id")
                or not isinstance(text, str) or action_issues(text)):
            raise ValueError("Reviewed action has stale or invalid actor evidence.")
        if target is not None and (target.get("panel_id") != actor.get("panel_id")
                                   or not _review_source_matches(review, target)):
            raise ValueError("Reviewed action has stale target evidence.")
        if actor.get("narration_eligibility") not in {"eligible_automatic", "reviewed_eligible"}:
            raise ValueError("Reviewed action actor is not narration-eligible.")
        if target is not None and target.get("narration_eligibility") not in {
            "eligible_automatic", "reviewed_eligible",
        }:
            raise ValueError("Reviewed action target is not narration-eligible.")
        events.append({
            "event_id": review.get("fact_id") or stable_id("story_event_", review),
            "type": "visible_action",
            "chapter": actor.get("chapter"), "page": Path(actor.get("page", "")).name,
            "panel_number": actor.get("panel_number"), "panel_id": actor.get("panel_id"),
            "text": text.strip(), "status": "accepted_reviewed",
            "acceptance_policy": "reviewed_visible_action_v1",
            "actor_instance_id": actor["instance_id"],
            "target_instance_id": target["instance_id"] if target else None,
            "review_reason": review.get("reason"),
        })

    for review in reviews.get("reviewed_transformations", []):
        if not isinstance(review, dict) or review.get("status") != "accepted_reviewed":
            raise ValueError("Reviewed transformation must be accepted_reviewed.")
        source = by_id.get(review.get("source_instance_id"))
        target = by_id.get(review.get("target_instance_id"))
        evidence = review.get("evidence")
        if (review.get("type", "transformation") not in {
                "transformation", "reversion", "disguise_reveal", "possession",
                "summoning", "splitting", "fusion",
            }
                or source is None or target is None or source is target
                or not _review_source_matches(review, source, "source_")
                or not _review_source_matches(review, target, "target_")
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Reviewed transformation has stale or invalid endpoints.")
        transformations.append({
            "event_id": review.get("fact_id") or stable_id("transformation_", review),
            "type": review.get("type", "transformation"),
            "status": "accepted_reviewed",
            "acceptance_policy": "reviewed_transformation_v1",
            "accepted_instance_ids": [source["instance_id"], target["instance_id"]],
            "source_panel_id": source.get("panel_id"),
            "target_panel_id": target.get("panel_id"),
            "source_page_hash": source.get("page_hash"),
            "target_page_hash": target.get("page_hash"),
            "source_geometry_revision": source.get("geometry_revision"),
            "target_geometry_revision": target.get("geometry_revision"),
            "source_grounding_revision": source.get("grounding_revision"),
            "target_grounding_revision": target.get("grounding_revision"),
            "evidence": evidence.strip(),
        })


def resolve_prepared(prepared: dict, *, reviews=None, calibration=None, transformations=None,
                     descriptions=None):
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
        page_name = Path(instance["page"]).name
        description_page = (descriptions or {}).get(instance["chapter"], {}).get(page_name, {})
        transformation_page = (
            (transformations or {}).get("chapters", {})
            .get(instance["chapter"], {}).get(page_name, {})
        )
        data = (
            description_page
            if description_page.get("extraction_status") in {
                "complete", "complete_with_uncertainty",
            }
            else transformation_page
        )
        if (instance.get("page_role_source") in ("unknown", "description")
                and data.get("source_hash") == instance.get("page_hash")
                and data.get("geometry_revision") == instance.get("geometry_revision")
                and data.get("page_role") in ("story", "reference", "extra", "uncertain")):
            role = data["page_role"]
            instance["page_role"] = role
            flags = set(instance["quality_flags"]) - {"non_story_or_unknown_role"}
            if role != "story":
                flags.add("non_story_or_unknown_role")
            instance["quality_flags"] = sorted(flags)
            instance["prototype_eligible"] = not flags
    resolver_kwargs = {
        "reviews": reviews,
        "group_threshold": prepared["group_threshold"],
        "calibration": calibration,
        "assignment_group_threshold": prepared.get("assignment_group_threshold", .14),
        "style_distances": prepared.get("style_distances"),
    }
    result = resolve(instances, distances, **resolver_kwargs)
    events = _build_transformation_proposals(instances, transformations)
    automatic_assignments = _automatic_transformation_assignments(result, events, reviews)
    if automatic_assignments:
        result = resolve(
            instances,
            distances,
            automatic_assignments=automatic_assignments,
            **resolver_kwargs,
        )
    result["warnings"] = warnings + result["warnings"]
    assign_group_display_names(result, prepared["manga"])
    _apply_grounded_descriptions(result, descriptions)
    _apply_eligibility_reviews(result, reviews)
    _classify_representation_types(result)
    _promote_automatic_identities(
        result,
        distances,
        proposal_threshold=prepared.get("group_threshold", .20),
        strict_threshold=prepared.get("assignment_group_threshold", .14),
        style_distances=prepared.get("style_distances"),
    )
    _apply_stable_descriptions(result)
    story_events = _build_story_events(result, descriptions)
    _apply_reviewed_events(result, reviews, story_events, events)
    text_proposals = []
    by_detection = {}
    for character in result.get("characters", []):
        for detection_id in character.get("grounding_ids", []) or []:
            by_detection.setdefault(detection_id, []).append(character)
    for chapter, pages in (descriptions or {}).items():
        if not isinstance(pages, dict):
            continue
        for page_name, page in pages.items():
            if not isinstance(page, dict):
                continue
            page_complete = page.get("extraction_status") in {
                "complete", "complete_with_uncertainty",
            }
            for proposal in page.get("text_proposals", []):
                if (not isinstance(proposal, dict)
                        or proposal.get("status") != "pending_review"
                        or not isinstance(proposal.get("panel_id"), str)):
                    continue
                exported = deepcopy(proposal)
                exported.update(
                    chapter=chapter,
                    page=page_name,
                    source_hash=page.get("source_hash"),
                    geometry_revision=page.get("geometry_revision"),
                    grounding_revision=page.get("grounding_revision"),
                )
                candidates = [
                    row for row in by_detection.get(
                        proposal.get("speaker_detection_id"), [],
                    )
                    if row.get("panel_id") == proposal.get("panel_id")
                ]
                exported["candidate_speaker_instance_id"] = (
                    candidates[0]["instance_id"] if len(candidates) == 1 else None
                )
                reason = None
                if not page_complete:
                    reason = "incomplete_page_extraction"
                elif proposal.get("kind") not in {
                    "dialogue", "thought", "narration", "sign", "sound_effect", "publication",
                }:
                    reason = "invalid_text_kind"
                elif text_issues(proposal.get("text", "")):
                    reason = "invalid_or_incomplete_transcription"
                elif (proposal.get("speaker_detection_id") is not None
                      and len(candidates) != 1):
                    reason = "exact_speaker_instance_required"
                elif (candidates and candidates[0].get("narration_eligibility") not in {
                    "eligible_automatic", "reviewed_eligible",
                }):
                    reason = "speaker_not_narration_eligible"
                else:
                    exported.update(
                        status="accepted_automatic",
                        acceptance_policy="grounded_text_transcription_format_v2",
                        speaker_instance_id=exported["candidate_speaker_instance_id"],
                        truth_status="transcribed_statement_only",
                    )
                if reason:
                    exported["automatic_rejection_reason"] = reason
                text_proposals.append(exported)
    if calibration is None:
        result["warnings"].append(
            "No validated calibration: broad visual groups remain proposals. Only recurrent "
            "multi-signal live-character cores may be confirmed automatically; other occurrences "
            "still require review or a strictly accepted Qwen transformation."
        )
    result.update(schema_version=3, manga=prepared["manga"], encoder=prepared["encoder"],
                  encoder_fingerprint=prepared["encoder_fingerprint"], provider=prepared["provider"],
                  style_linker={"encoder": prepared.get("style_encoder"),
                                "encoder_fingerprint": prepared.get("style_encoder_fingerprint"),
                                "provider": prepared.get("style_provider"),
                                "policy": "dual_encoder_anchor_v2" if prepared.get("style_distances") is not None else None},
                    acceptance_policy={"rule_version": "resolver_v11_story_text_scope",
                                     "proposal_distance": prepared["group_threshold"],
                                     "assignment_group_distance": prepared.get("assignment_group_threshold", .14),
                                     "calibration": calibration,
                                      "automatic_transformation": "qwen_direct_visible_change_v1",
                                        "grounded_story_event": "grounded_visible_action_format_v3",
                                       "reviewed_story_event": "reviewed_visible_action_v1",
                                       "automatic_text": "grounded_text_transcription_format_v2",
                                       "automatic_identity": "multi_signal_identity_v1"},
                  input_chapters=prepared["chapters"], transformation_proposals=events,
                   story_events=story_events, text_proposals=text_proposals,
                  description_fingerprint=stable_id("descriptions_", descriptions),
                  reviewed_evidence={
                      key: deepcopy((reviews or {}).get(key, []))
                      for key in (
                          "panel_descriptions", "reviewed_text_facts",
                          "reviewed_narration", "unresolved_critical_items",
                      )
                  },
                  elapsed_seconds=round(prepared["elapsed_seconds"] + time.perf_counter() - start, 3))
    result["revision"] = stable_id("revision_", "resolver_v11_story_text_scope", prepared["encoder_fingerprint"],
                                   prepared["group_threshold"], reviews, calibration,
                                   prepared.get("assignment_group_threshold", .14),
                                   prepared.get("style_encoder_fingerprint"), result["style_links"],
                                    events,
                                    story_events,
                                    text_proposals,
                                    result["description_fingerprint"],
                                    [(x["instance_id"], x["panel_id"], x["quality_flags"],
                                             x["page_role"], x["reading_position"]) for x in instances],
                                   result["characters"], result["identities"], result["visual_groups"])
    return result


def classify_manga(manga_dir: Path, chapters: list[Path], output_dir: Path, *, encoder: Encoder,
                   cache_dir: Path, reviews=None, calibration=None, transformations=None,
                   descriptions=None,
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
                              transformations=transformations, descriptions=descriptions)
    return result, prepared["crops"], prepared["distances"]
