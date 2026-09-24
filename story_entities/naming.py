"""Stable gender-neutral display names for retained visual groups."""

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


def _ranked_names(names: list[str], manga: str, group_id: str) -> list[str]:
    """Return a deterministic pseudorandom order specific to one visual group."""
    def score(name: str) -> bytes:
        value = f"{manga}\0{group_id}\0{name}".encode("utf-8")
        return hashlib.sha256(value).digest()

    return sorted(names, key=score)


def assign_group_display_names(result: dict, manga: str) -> None:
    """Name retained groups only; ungrouped observations remain unnamed."""
    groups = sorted(result.get("visual_groups", []), key=lambda row: row["group_id"])
    names = load_fallback_names()
    if len(groups) > len(names):
        raise ValueError(
            f"{len(groups)} retained visual groups exceed the {len(names)}-name fallback pool."
        )

    used: set[str] = set()
    assigned: dict[str, str] = {}
    for group in groups:
        group_id = group["group_id"]
        selected = next(
            name for name in _ranked_names(names, manga, group_id)
            if name.casefold() not in used
        )
        used.add(selected.casefold())
        assigned[group_id] = selected
        group.update(
            display_name=selected,
            display_name_source="fallback_pool",
        )

    for character in result.get("characters", []):
        selected = assigned.get(character.get("visual_group_id"))
        if selected is None:
            character.pop("display_name", None)
            character.pop("display_name_allowed", None)
            character.pop("display_name_source", None)
            character.pop("name_evidence_ids", None)
            continue
        character.update(
            display_name=selected,
            display_name_allowed=True,
            display_name_source="fallback_pool",
            name_evidence_ids=[],
        )

    result["display_naming_policy"] = {
        "rule_version": "stable_gender_neutral_surname_pool_v1",
        "fallback_pool_size": len(names),
        "pool_kind": "common_japanese_surnames",
        "ungrouped_characters_named": False,
    }
