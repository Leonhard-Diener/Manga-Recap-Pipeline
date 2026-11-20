"""Stable gender-neutral narration aliases, independent of canonical names."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_fallback_names() -> list[str]:
    """Load and validate the bundled common Japanese surname pool."""
    path = Path(__file__).with_name("manga_names.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data.get("names", []) if isinstance(data, dict) else []
    if data.get("schema_version") != 2 or data.get("kind") != "common_japanese_surnames":
        raise ValueError(f"Unsupported fallback-name pool in {path}.")
    if len(names) != 200 or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"Expected exactly 200 non-empty fallback names in {path}.")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(f"Fallback names must be unique in {path}.")
    return names


def _ranked_names(names: list[str], manga: str, subject_id: str) -> list[str]:
    """Return a deterministic pseudorandom order specific to one subject."""
    def score(name: str) -> bytes:
        value = f"{manga}\0{subject_id}\0{name}".encode("utf-8")
        return hashlib.sha256(value).digest()

    return sorted(names, key=score)


def assign_group_display_names(result: dict, manga: str) -> None:
    """Name only retained ReID groups, sharing aliases across accepted forms."""
    groups = sorted(result.get("visual_groups", []), key=lambda row: row["group_id"])
    characters = {
        row["instance_id"]: row
        for row in result.get("characters", [])
    }

    # A reviewed/accepted entity can contain several deliberately separate
    # visual groups (for example human and dragon forms). Those groups must use
    # one narration name. Unresolved groups remain independent and never gain
    # an identity merely because a transformation was proposed.
    group_subjects: dict[str, tuple[str, str]] = {}
    for group in groups:
        entity_ids = {
            characters[instance_id].get("entity_id")
            for instance_id in group.get("instance_ids", [])
            if instance_id in characters
        }
        # One accepted occurrence must not rename an otherwise unresolved group.
        if len(entity_ids) == 1 and None not in entity_ids:
            group_subjects[group["group_id"]] = ("entity", entity_ids.pop())
        else:
            group_subjects[group["group_id"]] = ("visual_group", group["group_id"])

    subjects = sorted(set(group_subjects.values()))
    names = load_fallback_names()
    if len(subjects) > len(names):
        raise ValueError(
            f"{len(subjects)} alias subjects exceed the {len(names)}-name fallback pool."
        )

    used: set[str] = set()
    subject_names: dict[tuple[str, str], str] = {}
    for subject in subjects:
        selected = next(
            name for name in _ranked_names(names, manga, subject[1])
            if name.casefold() not in used
        )
        used.add(selected.casefold())
        subject_names[subject] = selected

    for group in groups:
        group_id = group["group_id"]
        subject = group_subjects[group_id]
        selected = subject_names[subject]
        group.update(
            display_name=selected,
            display_name_source="fallback_pool",
            display_name_scope=subject[0],
        )

    for character in result.get("characters", []):
        subject = group_subjects.get(character.get("visual_group_id"))
        if subject is None:
            for key in ("display_name", "display_name_allowed", "display_name_source",
                        "display_name_scope", "name_evidence_ids"):
                character.pop(key, None)
            continue
        character.update(
            display_name=subject_names[subject],
            display_name_allowed=True,
            display_name_source="fallback_pool",
            display_name_scope=subject[0],
            name_evidence_ids=[],
        )

    for identity in result.get("identities", []):
        subject = ("entity", identity.get("entity_id"))
        selected = subject_names.get(subject)
        if selected is not None:
            identity.update(
                display_name=selected,
                display_name_source="fallback_pool",
                display_name_scope="entity",
            )

    result["display_naming_policy"] = {
        "rule_version": "retained_reid_gender_neutral_aliases_v6",
        "fallback_pool_size": len(names),
        "pool_kind": "common_japanese_surnames",
        "ungrouped_characters_named": False,
        "accepted_entity_forms_share_name": True,
    }
