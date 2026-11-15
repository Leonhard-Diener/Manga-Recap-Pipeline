"""Validate page descriptions and proposed character events without model imports."""

from __future__ import annotations

import re
from description.format import parse_json

from description.quality import (
    action_issues,
    annotate_panel_quality,
    character_description_is_useful,
)


EVENT_TYPES = {
    "transformation", "reversion", "disguise_reveal", "possession",
    "summoning", "splitting", "fusion",
}

def parse_page_response(response: str, *, truncated: bool = False, expected_panels=None,
                        character_labels=None, require_characters: bool = True,
                        require_events: bool = True) -> dict:
    """Reject incomplete/ungrounded event records; never treat failure as no events.

    Panel references are checked against the returned descriptions. This checks
    structure, not whether the model interpreted the image correctly. All events
    therefore remain pending and cannot establish a character identity.
    """
    result = {
        "panels": [], "characters": [], "events": [],
        "page_role": "uncertain", "status": "complete", "errors": [],
        "grounding_errors": [], "quality_errors": {}, "rejected_records": [],
    }
    if truncated:
        result.update(status="truncated", errors=["Generation reached its token limit."])
        return result

    text = response.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    try:
        payload = parse_json(text)
    except (ValueError, TypeError):
        result.update(status="invalid_output", errors=["Response is not a complete JSON object."])
        return result
    if not isinstance(payload, dict):
        result.update(status="invalid_output", errors=["Expected a JSON object."])
        return result

    role = payload.get("page_role", payload.get("r", "uncertain"))
    if role not in ("story", "reference", "extra", "uncertain"):
        result["errors"].append("Invalid page_role.")
    else:
        result["page_role"] = role

    label_records = character_labels if isinstance(character_labels, list) else None
    label_map = {
        row.get("label"): row
        for row in (label_records or [])
        if isinstance(row, dict) and isinstance(row.get("label"), str)
        and isinstance(row.get("detection_id"), str)
    }
    if label_records is not None and require_characters:
        described = payload.get("characters", payload.get("c"))
        seen_labels = set()
        if isinstance(described, dict):
            described = [
                {
                    "label": label,
                    "visual_description": (
                        visual.get("d") if isinstance(visual, dict) else visual
                    ),
                    "representation_type": (
                        visual.get("r") if isinstance(visual, dict) else "uncertain"
                    ),
                }
                for label, visual in described.items()
            ]
        elif not isinstance(described, list):
            result["grounding_errors"].append("Missing or invalid character descriptor list.")
            described = []
        for item in described:
            if not isinstance(item, dict):
                result["grounding_errors"].append("Character descriptor must be an object.")
                continue
            label = item.get("label", item.get("i"))
            visual = item.get("visual_description", item.get("v"))
            representation = item.get("representation_type", item.get("r", "uncertain"))
            if (label not in label_map or label in seen_labels
                    or not isinstance(visual, str) or (visual != "" and not visual.strip())
                    or representation not in {
                        "live_character", "toy", "statue", "image", "uncertain",
                    }):
                result["grounding_errors"].append("Character descriptor has an unknown/duplicate label or no description.")
                continue
            seen_labels.add(label)
            mapping = label_map[label]
            cleaned_visual = visual
            if cleaned_visual != "" and not character_description_is_useful(cleaned_visual):
                result["grounding_errors"].append(
                    f"Character {label} has no useful visible descriptor."
                )
                result["rejected_records"].append({
                    "kind": "character_description", "label": label,
                    "reasons": ["unclear_descriptor"], "value": cleaned_visual,
                })
                continue
            result["characters"].append({
                "label": label,
                "detection_id": mapping["detection_id"],
                "panel_number": mapping.get("panel_number"),
                "visual_description": cleaned_visual,
                "representation_type": representation,
                "representation_status": "model_observation",
                "status": "pending_reid",
                "appearance_status": "usable" if cleaned_visual else "unknown",
                "provenance": "qwen_focused_or_grounded_observation",
            })
        if seen_labels != set(label_map):
            result["grounding_errors"].append("Character descriptor coverage does not match the supplied labels.")

    panels = payload.get("panels", payload.get("p"))
    if not isinstance(panels, list) or not panels:
        result.update(status="invalid_output", errors=["Expected a non-empty panels list."])
        return result
    panel_numbers = set()
    for panel in panels:
        if not isinstance(panel, dict):
            result["errors"].append("Panel must be an object.")
            continue
        number = panel.get("panel_number", panel.get("n"))
        description = panel.get("description", panel.get("d"))
        if (type(number) is not int or number <= 0 or number in panel_numbers
                or not isinstance(description, str) or not description.strip()):
            result["errors"].append("Panel needs a unique positive number and a description.")
            continue
        panel_numbers.add(number)
        panel_result = {
            "panel_number": number,
            "description": description,
            "status": "usable",
            "provenance": "qwen_visual_observation",
        }
        if label_records is not None:
            panel_result["actions"] = []
            actions = panel.get("actions", panel.get("a"))
            if not isinstance(actions, list):
                result["grounding_errors"].append(
                    f"Panel {number} needs an explicit actions list."
                )
                actions = []
            for action_index, action in enumerate(actions):
                if not isinstance(action, dict):
                    result["grounding_errors"].append(
                        f"Panel {number} action {action_index + 1} must be an object."
                    )
                    continue
                actor = action.get("actor", action.get("s"))
                target = action.get("target", action.get("t"))
                text = action.get("text", action.get("x"))
                referenced = [label for label in (actor, target) if label is not None]
                if any(not isinstance(label, str) or label not in label_map for label in referenced):
                    result["grounding_errors"].append(
                        f"Panel {number} action {action_index + 1} references an unknown character label."
                    )
                    continue
                if any(label_map[label].get("panel_number") != number for label in referenced):
                    result["grounding_errors"].append(
                        f"Panel {number} action {action_index + 1} references a label outside that panel."
                    )
                    continue
                if not isinstance(text, str) or not text.strip():
                    result["grounding_errors"].append(
                        f"Panel {number} action {action_index + 1} has no action text."
                    )
                    continue
                semantic_issues = action_issues(
                    text, actor=actor, target=target,
                )
                if semantic_issues:
                    result["grounding_errors"].append(
                        f"Panel {number} action {action_index + 1} is unsafe: "
                        + ", ".join(semantic_issues) + "."
                    )
                    result["rejected_records"].append({
                        "kind": "action", "panel_number": number,
                        "reasons": semantic_issues, "value": text,
                    })
                    continue
                if target is not None and target == actor:
                    result["grounding_errors"].append(
                        f"Panel {number} action {action_index + 1} targets its own actor."
                    )
                    result["rejected_records"].append({
                        "kind": "action", "panel_number": number,
                        "reasons": ["self_target"], "value": text,
                    })
                    continue
                panel_result["actions"].append({
                    "actor_label": actor,
                    "actor_detection_id": label_map[actor]["detection_id"] if actor else None,
                    "target_label": target,
                    "target_detection_id": label_map[target]["detection_id"] if target else None,
                    "text": text,
                    "status": "pending_review" if actor else "ungrounded",
                    "grounding_status": "current" if actor else "actor_detection_missing",
                    "provenance": "qwen_grounded_action_proposal",
                })
                if actor is None:
                    result["grounding_errors"].append(
                        f"Panel {number} action {action_index + 1} is visible but has no actor detection."
                    )
        result["panels"].append(panel_result)

    if expected_panels is not None and panel_numbers != set(expected_panels):
        result["errors"].append("Panel coverage does not match the supplied page manifest.")
        result["panels"] = []

    quality_errors, rejected = annotate_panel_quality(result["panels"])
    result["quality_errors"] = quality_errors
    result["rejected_records"].extend(rejected)
    for panel in result["panels"]:
        if panel["panel_number"] in quality_errors:
            panel["status"] = "ambiguous"

    events = payload.get("events", payload.get("e"))
    if not require_events and events is None:
        events = []
    if not isinstance(events, list):
        result["errors"].append("Missing or invalid events list; an explicit [] is required.")
        events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            result["errors"].append(f"Event {index + 1} must be an object.")
            continue
        event_type = event.get("type", event.get("t"))
        if not isinstance(event_type, str) or event_type not in EVENT_TYPES:
            result["errors"].append(f"Event {index + 1} has an unsupported type.")
            continue
        references = [event.get("source_panel", event.get("sp")),
                      event.get("target_panel", event.get("tp"))]
        if any(type(number) is not int or number not in panel_numbers for number in references):
            result["errors"].append(f"Event {index + 1} references an undescribed panel.")
            continue
        field_values = {
            "source_description": event.get("source_description", event.get("sd")),
            "target_description": event.get("target_description", event.get("td")),
            "evidence": event.get("evidence", event.get("x")),
        }
        if any(not isinstance(value, str) or not value.strip()
               for value in field_values.values()):
            result["errors"].append(f"Event {index + 1} needs source, target, and evidence text.")
            continue
        evidence_type = event.get("evidence_type", event.get("xt"))
        if evidence_type not in ("visible_change", "explicit_text", "ambiguous_sequence"):
            result["errors"].append(f"Event {index + 1} has an unsupported evidence type.")
            continue
        confidence = event.get("confidence_percent", event.get("c"))
        if type(confidence) is not int or not 0 <= confidence <= 100:
            result["errors"].append(
                f"Event {index + 1} needs an integer confidence_percent from 0 to 100."
            )
            continue
        event_result = {
            "type": event_type,
            "source_panel": references[0],
            "target_panel": references[1],
            **field_values,
            "evidence_type": evidence_type,
            "confidence_percent": confidence,
            "status": "pending_review",
        }
        if label_records is not None:
            source_label = event.get("source_label", event.get("si"))
            target_label = event.get("target_label", event.get("ti"))
            if source_label not in label_map or target_label not in label_map:
                result["errors"].append(f"Event {index + 1} references an unknown character label.")
                continue
            if (label_map[source_label].get("panel_number") != references[0]
                    or label_map[target_label].get("panel_number") != references[1]):
                result["errors"].append(f"Event {index + 1} character labels do not match its panels.")
                continue
            event_result.update(
                source_label=source_label,
                source_detection_id=label_map[source_label]["detection_id"],
                target_label=target_label,
                target_detection_id=label_map[target_label]["detection_id"],
            )
        result["events"].append(event_result)

    if result["errors"]:
        result["status"] = "invalid_output"
        # Retain usable descriptions, but do not release events from a response
        # with ambiguous panel mapping or a partially invalid event list.
        result["events"] = []
        result["page_role"] = "uncertain"
    result["panels"].sort(key=lambda panel: panel["panel_number"])
    return result
