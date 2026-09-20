"""Validate page descriptions and proposed character events without model imports."""

from __future__ import annotations

import json
import re


EVENT_TYPES = {
    "transformation", "reversion", "disguise_reveal", "possession",
    "summoning", "splitting", "fusion",
}


def parse_page_response(response: str, *, truncated: bool = False, expected_panels=None) -> dict:
    """Reject incomplete/ungrounded event records; never treat failure as no events.

    Panel references are checked against the returned descriptions. This checks
    structure, not whether the model interpreted the image correctly. All events
    therefore remain pending and cannot establish a character identity.
    """
    result = {"panels": [], "events": [], "page_role": "uncertain", "status": "complete", "errors": []}
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

    role = payload.get("page_role", "uncertain")
    if role not in ("story", "reference", "extra", "uncertain"):
        result["errors"].append("Invalid page_role.")
    else:
        result["page_role"] = role

    panels = payload.get("panels")
    if not isinstance(panels, list) or not panels:
        result.update(status="invalid_output", errors=["Expected a non-empty panels list."])
        return result
    panel_numbers = set()
    for panel in panels:
        if not isinstance(panel, dict):
            result["errors"].append("Panel must be an object.")
            continue
        number = panel.get("panel_number")
        description = panel.get("description")
        if (type(number) is not int or number <= 0 or number in panel_numbers
                or not isinstance(description, str) or not description.strip()):
            result["errors"].append("Panel needs a unique positive number and a description.")
            continue
        panel_numbers.add(number)
        result["panels"].append({"panel_number": number, "description": description.strip()})

    if expected_panels is not None and panel_numbers != set(expected_panels):
        result["errors"].append("Panel coverage does not match the supplied page manifest.")
        result["panels"] = []

    events = payload.get("events")
    if not isinstance(events, list):
        result["errors"].append("Missing or invalid events list; an explicit [] is required.")
        events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            result["errors"].append(f"Event {index + 1} must be an object.")
            continue
        if not isinstance(event.get("type"), str) or event["type"] not in EVENT_TYPES:
            result["errors"].append(f"Event {index + 1} has an unsupported type.")
            continue
        references = [event.get("source_panel"), event.get("target_panel")]
        if any(type(number) is not int or number not in panel_numbers for number in references):
            result["errors"].append(f"Event {index + 1} references an undescribed panel.")
            continue
        fields = ("source_description", "target_description", "evidence")
        if any(not isinstance(event.get(key), str) or not event[key].strip() for key in fields):
            result["errors"].append(f"Event {index + 1} needs source, target, and evidence text.")
            continue
        if event.get("evidence_type") not in ("visible_change", "explicit_text", "ambiguous_sequence"):
            result["errors"].append(f"Event {index + 1} has an unsupported evidence type.")
            continue
        result["events"].append({
            "type": event["type"],
            "source_panel": references[0],
            "target_panel": references[1],
            **{key: event[key].strip() for key in fields},
            "evidence_type": event["evidence_type"],
            "status": "pending_review",
        })

    if result["errors"]:
        result["status"] = "invalid_output"
        # Retain usable descriptions, but do not release events from a response
        # with ambiguous panel mapping or a partially invalid event list.
        result["events"] = []
        result["page_role"] = "uncertain"
    result["panels"].sort(key=lambda panel: panel["panel_number"])
    return result
