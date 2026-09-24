"""Validate page descriptions and proposed character events without model imports."""

from __future__ import annotations

import json
import re


EVENT_TYPES = {
    "transformation", "reversion", "disguise_reveal", "possession",
    "summoning", "splitting", "fusion",
}

NAME_EVIDENCE_TYPES = {
    "self_introduction", "introduced_by_other", "profile_label", "direct_address",
}


def parse_page_response(response: str, *, truncated: bool = False, expected_panels=None) -> dict:
    """Reject incomplete/ungrounded event records; never treat failure as no events.

    Panel references are checked against the returned descriptions. This checks
    structure, not whether the model interpreted the image correctly. All events
    therefore remain pending and cannot establish a character identity.
    """
    result = {
        "panels": [], "events": [], "name_mentions": [],
        "page_role": "uncertain", "status": "complete", "errors": [],
        "name_status": "complete", "name_errors": [],
    }
    if truncated:
        result.update(status="truncated", errors=["Generation reached its token limit."])
        return result

    text = response.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    try:
        payload = json.loads(text)
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
        result["panels"].append({"panel_number": number, "description": description.strip()})

    if expected_panels is not None and panel_numbers != set(expected_panels):
        result["errors"].append("Panel coverage does not match the supplied page manifest.")
        result["panels"] = []

    events = payload.get("events", payload.get("e"))
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
        result["events"].append({
            "type": event_type,
            "source_panel": references[0],
            "target_panel": references[1],
            **{key: value.strip() for key, value in field_values.items()},
            "evidence_type": evidence_type,
            "status": "pending_review",
        })

    # Optional OCR evidence fails independently. A malformed name must not
    # invalidate correct descriptions or transformation proposals.
    mentions = payload.get("name_mentions", payload.get("m", []))
    if not isinstance(mentions, list):
        result["name_status"] = "invalid_output"
        result["name_errors"].append("Invalid name_mentions list.")
        mentions = []
    for index, mention in enumerate(mentions):
        if not isinstance(mention, dict):
            result["name_errors"].append(f"Name mention {index + 1} must be an object.")
            continue
        name = mention.get("name", mention.get("n"))
        evidence = mention.get("evidence", mention.get("x"))
        number = mention.get("panel_number", mention.get("p"))
        box = mention.get("character_box", mention.get("b"))
        evidence_type = mention.get("evidence_type", mention.get("xt"))
        valid_box = (
            isinstance(box, list) and len(box) == 4
            and all(type(value) in (int, float) and 0 <= value <= 1000 for value in box)
            and box[2] > box[0] and box[3] > box[1]
        )
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            result["name_errors"].append(f"Name mention {index + 1} has an invalid name.")
            continue
        if type(number) is not int or number not in panel_numbers:
            result["name_errors"].append(f"Name mention {index + 1} references an undescribed panel.")
            continue
        if not valid_box:
            result["name_errors"].append(f"Name mention {index + 1} has an invalid character box.")
            continue
        if not isinstance(evidence, str) or not evidence.strip():
            result["name_errors"].append(f"Name mention {index + 1} needs exact text evidence.")
            continue
        if evidence_type not in NAME_EVIDENCE_TYPES:
            result["name_errors"].append(f"Name mention {index + 1} has an unsupported evidence type.")
            continue
        result["name_mentions"].append({
            "name": name.strip(),
            "panel_number": number,
            "character_box": [float(value) for value in box],
            "evidence": evidence.strip(),
            "evidence_type": evidence_type,
            "status": "automatic_candidate",
        })
    if result["name_errors"]:
        result["name_status"] = "partial" if result["name_mentions"] else "invalid_output"

    if result["errors"]:
        result["status"] = "invalid_output"
        # Retain usable descriptions, but do not release events from a response
        # with ambiguous panel mapping or a partially invalid event list.
        result["events"] = []
        result["page_role"] = "uncertain"
    result["panels"].sort(key=lambda panel: panel["panel_number"])
    return result
