"""Pure matching rules. Proposals cannot silently turn into named identities."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math

import numpy as np


def stable_id(prefix: str, *parts: object) -> str:
    value = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + sha256(value.encode("utf-8")).hexdigest()[:24]


def story_panel(instance: dict):
    """Imported copies cannot become independent identity evidence."""
    return instance.get("story_panel_id", instance.get("panel_id"))


def conservative_groups(instances: list[dict], distances: np.ndarray,
                        threshold: float, cannot_link: set[frozenset] | None = None) -> list[list[int]]:
    """Complete-link clustering with hard pair exclusions and stable tie-breaking.

    Low-quality instances remain singletons. No same-panel exclusion is invented.
    These are visual proposals, never automatically confirmed identities.
    """
    n = len(instances)
    distances = np.asarray(distances)
    if distances.shape != (n, n) or not np.isfinite(distances).all():
        raise ValueError("Distance matrix must be finite and square, matching the instances.")
    if len({x["instance_id"] for x in instances}) != n:
        raise ValueError("Instance IDs must be unique.")
    if not np.isfinite(threshold) or threshold < 0 or np.any(distances < 0):
        raise ValueError("Distances and proposal threshold must be nonnegative.")
    if not np.allclose(distances, distances.T, atol=1e-5):
        raise ValueError("Distance matrix must be symmetric.")
    blocked = cannot_link or set()
    active = {i: [i] for i in range(n)}
    owner = list(range(n))
    candidates = sorted(
        (float(distances[i, j]), *sorted((instances[i]["instance_id"], instances[j]["instance_id"])), i, j)
        for i in range(n) for j in range(i + 1, n)
        if instances[i].get("embedding_eligible", instances[i]["prototype_eligible"])
        and instances[j].get("embedding_eligible", instances[j]["prototype_eligible"])
        and distances[i, j] <= threshold
    )
    for _, _, _, i, j in candidates:
        a, b = owner[i], owner[j]
        if a == b:
            continue
        left, right = active[a], active[b]
        if np.max(distances[np.ix_(left, right)]) > threshold:
            continue
        if any(frozenset((instances[x]["instance_id"], instances[y]["instance_id"])) in blocked
               for x in left for y in right):
            continue
        active[a] = left + right
        for member in right:
            owner[member] = a
        del active[b]
    return sorted((sorted(group, key=lambda i: instances[i]["instance_id"])
                   for group in active.values()), key=lambda g: instances[g[0]]["instance_id"])


def _two_panel_bridge_score(instances: list[dict], distances: np.ndarray,
                            left: list[int], right: list[int]) -> float | None:
    """Best two cross-group matches backed by distinct panels on both sides."""
    pairs = sorted((float(distances[i, j]), i, j) for i in left for j in right)
    for first, (_, left_a, right_a) in enumerate(pairs):
        for score, left_b, right_b in pairs[first + 1:]:
            if (left_a != left_b and right_a != right_b
                    and story_panel(instances[left_a]) != story_panel(instances[left_b])
                    and story_panel(instances[right_a]) != story_panel(instances[right_b])):
                return score
    return None


def link_style_satellites(instances: list[dict], groups: list[list[int]],
                          primary_distances: np.ndarray, style_distances: np.ndarray,
                          cannot_link: set[frozenset] | None = None, *,
                          satellite_max_size: int = 20, core_min_size: int = 4,
                          primary_max_distance: float = .30, style_max_distance: float = .18,
                          primary_min_margin: float = .02, style_min_margin: float = .02,
                          refinement_min_panels: int = 4):
    """Link simple/detail subgroups only when two encoders independently agree.

    The complete-link groups remain the trusted visual cores. A singleton never
    links, and the secondary encoder cannot create a group by itself. These are
    review proposals only and never identity assignments.
    """
    n = len(instances)
    style_distances = np.asarray(style_distances)
    if style_distances.shape != (n, n) or not np.isfinite(style_distances).all():
        raise ValueError("Style distance matrix must match instances and be finite.")
    if np.any(style_distances < 0) or not np.allclose(style_distances, style_distances.T, atol=1e-5):
        raise ValueError("Style distance matrix must be symmetric and nonnegative.")
    blocked = cannot_link or set()
    def blocked_between(source, target):
        return any(frozenset((instances[x]["instance_id"], instances[y]["instance_id"])) in blocked
                   for x in source for y in target)

    def choose_target(source_index, target_indices, family_members=None, limits=None):
        source = groups[source_index]
        p_max, s_max, p_margin, s_margin = limits or (
            primary_max_distance, style_max_distance, primary_min_margin, style_min_margin)
        candidates = []
        for target_index in target_indices:
            target = groups[target_index]
            if (target_index == source_index or len(target) < core_min_size
                    or len(target) <= len(source)):
                continue
            constraint_target = family_members(target_index) if family_members else target
            is_blocked = blocked_between(source, constraint_target)
            primary = _two_panel_bridge_score(instances, primary_distances, source, target)
            style = _two_panel_bridge_score(instances, style_distances, source, target)
            if primary is not None and style is not None:
                candidates.append((primary, style, target_index, is_blocked))
        if len(candidates) < 2:
            return None
        by_primary = sorted(candidates, key=lambda item: (item[0], item[2]))
        by_style = sorted(candidates, key=lambda item: (item[1], item[2]))
        best_primary, best_style = by_primary[0], by_style[0]
        if (best_primary[2] != best_style[2]
                or best_primary[3] or best_style[3]
                or best_primary[0] > p_max
                or best_style[1] > s_max
                or by_primary[1][0] - best_primary[0] < p_margin
                or by_style[1][1] - best_style[1] < s_margin):
            return None
        return {
            "target": best_primary[2],
            "primary_distance": round(best_primary[0], 6),
            "style_distance": round(best_style[1], 6),
            "primary_margin": round(by_primary[1][0] - best_primary[0], 6),
            "style_margin": round(by_style[1][1] - best_style[1], 6),
        }

    choices = {}
    for source_index, source in enumerate(groups):
        if 2 <= len(source) <= satellite_max_size:
            evidence = choose_target(source_index, range(len(groups)))
            if evidence is not None:
                choices[source_index] = evidence

    # A satellite cannot simultaneously act as a bridge to form a merge chain.
    choices = {source: evidence for source, evidence in choices.items()
               if evidence["target"] not in choices}
    parent = list(range(len(groups)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    links, linked_sources = [], set()

    def attach(source, evidence, stage):
        target = evidence["target"]
        parent[find(source)] = find(target)
        linked_sources.add(source)
        links.append({
            "source_instance_ids": [instances[i]["instance_id"] for i in groups[source]],
            "target_instance_ids": [instances[i]["instance_id"] for i in groups[target]],
            **{key: value for key, value in evidence.items() if key != "target"},
            "status": "proposed_style_link",
            "stage": stage,
        })

    for source, evidence in sorted(choices.items()):
        attach(source, evidence, "direct_dual_encoder")

    # A confidently linked subgroup is no longer a competing character. Recheck
    # remaining recurring subgroups against original anchor cores only. Linked
    # satellite images never become evidence, preventing transitive drift.
    while True:
        anchors = [index for index in range(len(groups)) if index not in linked_sources]

        def family_members(target_index):
            root = find(target_index)
            return [member for index, group in enumerate(groups) if find(index) == root for member in group]

        refinements, refinement_stages = {}, {}
        for source_index, source in enumerate(groups):
            panels = {story_panel(instances[i]) for i in source if story_panel(instances[i])}
            if source_index in linked_sources or len(source) > satellite_max_size:
                continue
            if len(source) >= core_min_size and len(panels) >= refinement_min_panels:
                limits = None
                stage = "anchor_refinement"
            elif len(source) == 2 and len(panels) == 2:
                limits = (.16, .12, .08, .03)
                stage = "strict_small_anchor_refinement"
            else:
                continue
            evidence = choose_target(source_index, anchors, family_members, limits)
            if evidence is not None:
                refinements[source_index] = evidence
                refinement_stages[source_index] = stage
        refinements = {source: evidence for source, evidence in refinements.items()
                       if evidence["target"] not in refinements}
        if not refinements:
            break
        for source, evidence in sorted(refinements.items()):
            attach(source, evidence, refinement_stages[source])
    families = defaultdict(list)
    for group_index, members in enumerate(groups):
        families[find(group_index)].extend(members)
    merged = sorted((sorted(members, key=lambda i: instances[i]["instance_id"])
                     for members in families.values()), key=lambda g: instances[g[0]]["instance_id"])
    return merged, links


def validate_reviews(reviews: dict, instances: list[dict]) -> tuple[dict, set[frozenset], list[str]]:
    """Exact instance IDs include image+box fingerprints; changed crops go stale."""
    if not isinstance(reviews, dict) or reviews.get("schema_version", 1) != 1:
        raise ValueError("Reviews must be a schema_version 1 object.")
    known = {x["instance_id"]: x for x in instances}
    assignments, blocked, warnings = {}, set(), []
    for key in ("assignments", "different", "names"):
        if not isinstance(reviews.get(key, []), list):
            raise ValueError(f"reviews.{key} must be a list.")
    for entry in reviews.get("assignments", []):
        if not isinstance(entry, dict):
            raise ValueError("Each review assignment must be an object.")
        if entry.get("accepted") is not True:
            continue
        required = ("instance_id", "entity_id", "form_id", "evidence")
        if any(not isinstance(entry.get(k), str) or not entry[k].strip() for k in required):
            raise ValueError(f"Accepted assignment requires {required}.")
        identity = entry["instance_id"]
        if identity not in known:
            warnings.append(f"Stale review ignored: {identity}")
            continue
        if identity in assignments and any(assignments[identity][k] != entry[k]
                                           for k in ("entity_id", "form_id")):
            raise ValueError(f"Conflicting reviews for {identity}.")
        assignments[identity] = dict(entry)
        reveal = entry.get("identity_reveal_instance_id")
        if reveal is not None and (not isinstance(reveal, str) or not reveal.strip()):
            raise ValueError("identity_reveal_instance_id must be a nonempty instance ID.")
        if reveal is not None and reveal not in known:
            warnings.append(f"Stale identity reveal; naming disabled: {identity}")
    form_entities = {}
    for entry in assignments.values():
        form = entry["form_id"]
        if form in form_entities and form_entities[form] != entry["entity_id"]:
            raise ValueError(f"Form {form} assigned to different entities.")
        form_entities[form] = entry["entity_id"]
    for pair in reviews.get("different", []):
        if not isinstance(pair, dict) or not all(isinstance(pair.get(k), str) and pair[k]
                                               for k in ("left", "right", "evidence")):
            raise ValueError("Different-identity constraint needs left, right, evidence.")
        if pair["left"] == pair["right"]:
            raise ValueError("An instance cannot be different from itself.")
        if pair["left"] not in known or pair["right"] not in known:
            warnings.append("Stale different-identity constraint ignored.")
            continue
        left, right = assignments.get(pair["left"]), assignments.get(pair["right"])
        if left and right and left["entity_id"] == right["entity_id"]:
            raise ValueError("Accepted assignments conflict with a different-identity constraint.")
        blocked.add(frozenset((pair["left"], pair["right"])))
    # Reviewed forms remain separate visual groups even if the entity is shared.
    keys = sorted(assignments)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if (assignments[left]["entity_id"], assignments[left]["form_id"]) != (
                    assignments[right]["entity_id"], assignments[right]["form_id"]):
                blocked.add(frozenset((left, right)))
    return assignments, blocked, warnings


def validate_automatic_assignments(assignments: dict | None, instances: list[dict]) -> dict:
    """Validate internal, evidence-gated transformation assignments."""
    if assignments is None:
        return {}
    if not isinstance(assignments, dict):
        raise ValueError("Automatic assignments must be keyed by instance ID.")
    known = {row["instance_id"] for row in instances}
    validated = {}
    for instance_id, entry in assignments.items():
        if instance_id not in known or not isinstance(entry, dict):
            raise ValueError("Automatic assignment references an unknown instance.")
        if any(not isinstance(entry.get(key), str) or not entry[key].strip()
               for key in ("entity_id", "form_id", "evidence")):
            raise ValueError("Automatic assignment needs entity, form, and evidence.")
        event_ids = entry.get("event_ids")
        if (not isinstance(event_ids, list) or not event_ids
                or any(not isinstance(value, str) or not value for value in event_ids)):
            raise ValueError("Automatic assignment needs supporting event IDs.")
        validated[instance_id] = dict(entry, event_ids=sorted(set(event_ids)))
    return validated


def resolve(instances: list[dict], distances: np.ndarray, *, reviews: dict | None = None,
            group_threshold: float = 0.20, calibration: dict | None = None,
            assignment_group_threshold: float = 0.14,
            style_distances: np.ndarray | None = None,
            automatic_assignments: dict | None = None,
            minimum_group_frequency_ratio: float = .05) -> dict:
    """Return independent instance decisions; names never transfer from proposals.

    Without a validated calibration, only explicit reviewed assignments become
    accepted. Visual grouping still runs and generates useful review galleries.
    Replaying from inputs+reviews automatically removes revoked decisions.
    """
    reviews = reviews or {}
    assignments, blocked, warnings = validate_reviews(reviews, instances)
    automatic = validate_automatic_assignments(automatic_assignments, instances)
    core_groups = conservative_groups(instances, distances, group_threshold, blocked)
    groups, style_links = ((core_groups, []) if style_distances is None else
                           link_style_satellites(instances, core_groups, distances,
                                                 style_distances, blocked))
    # Display grouping may be more tolerant of style changes. Keep the original
    # competitor boundaries for identity acceptance independent of that setting.
    acceptance_groups = (core_groups if group_threshold == assignment_group_threshold else
                         conservative_groups(instances, distances, assignment_group_threshold, blocked))
    acceptance_group_for = {member: i for i, group in enumerate(acceptance_groups) for member in group}
    proposals = []
    group_for = {}
    for members in groups:
        ids = [instances[i]["instance_id"] for i in members]
        occurrence_count = len({story_panel(instances[i]) for i in members if story_panel(instances[i])})
        group_id = stable_id("group_", ids)
        proposals.append({"group_id": group_id, "instance_ids": ids, "status": "proposed",
                          "size": len(ids), "occurrence_count": occurrence_count,
                          "style_links": [link for link in style_links
                                          if set(link["source_instance_ids"]).issubset(ids)]})
        group_for.update({i: group_id for i in members})
    gallery = defaultdict(list)
    for i, instance in enumerate(instances):
        reviewed = assignments.get(instance["instance_id"])
        if reviewed and instance["prototype_eligible"]:
            gallery[(reviewed["entity_id"], reviewed["form_id"])].append(i)
    records = []
    for i, instance in enumerate(instances):
        row = dict(instance)
        reviewed = assignments.get(instance["instance_id"])
        transformed = automatic.get(instance["instance_id"])
        row.update(entity_id=None, form_id=None, name=None, identity_status="unknown",
                   visual_group_id=group_for[i], assignment_evidence=[], candidates=[],
                   naming_allowed=False, narration_allowed=False)
        if reviewed:
            row.update(entity_id=reviewed["entity_id"], form_id=reviewed["form_id"],
                       identity_status="reviewed", assignment_evidence=[reviewed["evidence"]],
                       decision_ids=[stable_id("decision_", "assignments", reviewed)])
        elif transformed and instance["prototype_eligible"]:
            row.update(
                entity_id=transformed["entity_id"],
                form_id=transformed["form_id"],
                identity_status="accepted_qwen_transformation",
                assignment_evidence=[transformed["evidence"]],
                decision_ids=[
                    stable_id("decision_", "qwen_transformation", event_id)
                    for event_id in transformed["event_ids"]
                ],
            )
        elif instance["prototype_eligible"]:
            ranked = []
            for (entity, form), refs in gallery.items():
                # Multiple images of the same panel are not independent support.
                panels = {story_panel(instances[r]) for r in refs} - {None, story_panel(instance)}
                refs = [r for r in refs if story_panel(instances[r]) in panels]
                if len(panels) < 2:
                    continue
                if any(frozenset((instance["instance_id"], instances[r]["instance_id"])) in blocked
                       for r in refs):
                    continue
                ranked.append((float(np.max(distances[i, refs])), entity, form, refs))
            ranked.sort(key=lambda item: item[:3])
            row["candidates"] = [{"entity_id": e, "form_id": f, "max_distance": round(d, 6)}
                                 for d, e, f, _ in ranked[:3]]
            if ranked:
                best, entity, form, refs = ranked[0]
                competing = [x[0] for x in ranked[1:]]
                # Also consider unreviewed lookalikes outside the chosen visual
                # group, so a single named gallery is not a closed-world classifier.
                # Even ONE reference for another reviewed entity is a competitor,
                # including references inside the query's visual proposal group.
                other = [j for j in range(len(instances)) if j != i
                         and instances[j]["prototype_eligible"]
                         and assignments.get(instances[j]["instance_id"], {}).get("entity_id") != entity
                         and (instances[j]["instance_id"] in assignments
                              or acceptance_group_for[j] != acceptance_group_for[i])]
                if other:
                    competing.append(float(np.min(distances[i, other])))
                margin = min(competing) - best if competing else None
                row["match_margin"] = margin
                if (calibration and best <= calibration["max_distance"]
                        and (margin is None or (margin > 0 and margin >= calibration["min_margin"]))):
                    row.update(entity_id=entity, form_id=form, identity_status="accepted_visual",
                               assignment_evidence=[instances[r]["instance_id"] for r in refs],
                               decision_ids=[stable_id("decision_", "assignments", assignments[instances[r]["instance_id"]])
                                             for r in refs])
        records.append(row)

    # A newly accepted pair must not violate an explicit separation anywhere.
    by_id = {r["instance_id"]: r for r in records}
    for pair in blocked:
        a, b = [by_id[key] for key in pair]
        if a["entity_id"] and a["entity_id"] == b["entity_id"]:
            # Separate reviewed forms of the same entity are intentional.
            if all(x["identity_status"] == "reviewed" for x in (a, b)):
                continue
            for record in (a, b):
                if record["identity_status"] in (
                    "accepted_visual", "accepted_qwen_transformation"
                ):
                    record.update(entity_id=None, form_id=None, identity_status="unknown",
                                  assignment_evidence=[], decision_ids=[])
                    warnings.append(f"Rejected constraint conflict: {record['instance_id']}")

    name_records = {}
    for entry in reviews.get("names", []):
        if not isinstance(entry, dict):
            raise ValueError("Name record must be an object.")
        if entry.get("accepted") is not True:
            continue
        if any(not isinstance(entry.get(k), str) or not entry[k].strip()
               for k in ("entity_id", "name", "evidence", "source_instance_id")):
            raise ValueError("Accepted name requires entity_id, name, evidence, source_instance_id.")
        source = by_id.get(entry["source_instance_id"])
        if not source or source["entity_id"] != entry["entity_id"]:
            warnings.append(f"Stale or ungrounded name ignored: {entry['name']}")
            continue
        if entry["entity_id"] in name_records and name_records[entry["entity_id"]]["name"] != entry["name"]:
            raise ValueError("Conflicting canonical names; keep aliases outside the canonical name list.")
        name_records[entry["entity_id"]] = entry

    entities = []
    for entity in sorted({r["entity_id"] for r in records if r["entity_id"]}):
        members = [r for r in records if r["entity_id"] == entity]
        panels = {story_panel(r) for r in members if story_panel(r) and r.get("page_role") == "story"
                  and not r.get("panel_provisional", False)}
        naming = name_records.get(entity)
        eligible = len(panels) >= 4 or bool(naming and naming.get("promote") is True)
        name = naming["name"] if naming and eligible else None
        entities.append({"entity_id": entity, "name": name, "appearance_count": len(panels),
                         "recurrence": "recurring" if len(panels) >= 4 else "minor",
                         "name_record": dict(naming) if naming else None,
                         "naming_eligible": eligible, "form_ids": sorted({r["form_id"] for r in members}),
                         "instance_ids": [r["instance_id"] for r in members]})
        for row in members:
            row["appearance_count"] = len(panels)
            row["name"] = name
            # A name is exportable only after its reviewed reveal point.
            source = by_id.get(naming["source_instance_id"]) if naming else None
            reveal = source.get("reading_position") if source else None
            position = row.get("reading_position")
            row["naming_allowed"] = bool(name and reveal is not None and position is not None and position >= reveal)
            # A form-specific identity reveal can be later than the name itself.
            # Apply every current reviewed reveal for the form, including to
            # automatic attachments, without leaking that knowledge backwards.
            form_reviews = [a for a in assignments.values()
                            if a["entity_id"] == entity and a["form_id"] == row["form_id"]]
            for decision in form_reviews:
                identity_reveal = decision.get("identity_reveal_instance_id")
                if identity_reveal is not None:
                    reveal_position = by_id.get(identity_reveal, {}).get("reading_position")
                    if reveal_position is None or position is None or position < reveal_position:
                        row["naming_allowed"] = False
            if not row["naming_allowed"]:
                row["name"] = None
            # ReID cannot prove which actor performed a described action.
            row["narration_allowed"] = False
    # Rare proposals are retained as unclassified observations. Count distinct
    # panels rather than detector boxes and scale the cutoff to the largest group.
    if not 0 <= minimum_group_frequency_ratio <= 1:
        raise ValueError("Minimum group frequency ratio must be between 0 and 1.")
    multi = [group for group in proposals if group["size"] > 1]
    maximum_occurrences = max((group["occurrence_count"] for group in multi), default=0)
    minimum_occurrences = (math.ceil(maximum_occurrences * minimum_group_frequency_ratio)
                           if maximum_occurrences else 0)
    proposals = [group for group in multi if group["occurrence_count"] >= minimum_occurrences]
    retained = {group["group_id"] for group in proposals}
    retained_instances = {instance_id for group in proposals for instance_id in group["instance_ids"]}
    style_links = [link for link in style_links
                   if set(link["source_instance_ids"]).issubset(retained_instances)
                   and set(link["target_instance_ids"]).issubset(retained_instances)]
    ungrouped = []
    for row in records:
        if row["visual_group_id"] not in retained:
            row["visual_group_id"] = None
            ungrouped.append(row["instance_id"])
    return {"characters": records, "identities": entities, "visual_groups": proposals,
            "ungrouped_instance_ids": ungrouped,
            "style_links": style_links,
            "visual_group_filter": {"minimum_frequency_ratio": minimum_group_frequency_ratio,
                                    "maximum_occurrences": maximum_occurrences,
                                    "minimum_occurrences": minimum_occurrences},
            "warnings": warnings, "calibrated": calibration is not None}
