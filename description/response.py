"""Validate page descriptions and proposed character events without model imports."""

from __future__ import annotations

import json
import re


EVENT_TYPES = {
    "transformation", "reversion", "disguise_reveal", "possession",
    "summoning", "splitting", "fusion",
}

def _finish_description(value: str) -> str:
    """Normalize labels and safely close a sentence clipped by the schema cap."""
    text = re.sub(r"^\s*Panel\s+\d+\s*:\s*", "", value, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Dialogue/title transcription adds cost but no useful visual summary.
    # Remove both complete and cap-truncated quoted spans while preserving
    # apostrophes inside ordinary words such as "girl's".
    text = re.sub(r'(?<!\w)"[^"]*(?:"|$)', "", text)
    text = re.sub(r"(?<!\w)'[^']*(?:'|$)", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    text = re.sub(
        r",?\s*(?:featuring|showing)?\s*(?:written\s+)?"
        r"(?:text|titles?|sound effects?|speech bubbles?|name tags?).*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" ,;:-")
    text = re.sub(
        r",?\s*(?:shouting|yelling|saying|reading)$", "", text,
        flags=re.IGNORECASE,
    ).strip(" ,;:-")

    # When the model writes to the schema's string limit, drop a short dangling
    # clause instead of exporting a cut-off word or phrase.
    if len(text) >= 92:
        candidates = []
        for separator in (",", ";", " while ", " with ", " featuring ", " showing "):
            position = text.rfind(separator)
            if position > 0:
                head = text[:position].rstrip()
                if len(head.split()) >= 5:
                    candidates.append((position, text[:position].rstrip()))
        if candidates:
            text = max(candidates)[1]
        elif len(text.split()) > 5:
            text = text.rsplit(" ", 1)[0]
    if not text:
        return "Visible manga scene."
    if text[-1] in ".!?":
        return text
    return text.rstrip(" ,;:-'\"") + "."


def parse_page_response(response: str, *, truncated: bool = False, expected_panels=None) -> dict:
    """Reject incomplete/ungrounded event records; never treat failure as no events.

    Panel references are checked against the returned descriptions. This checks
    structure, not whether the model interpreted the image correctly. All events
    therefore remain pending and cannot establish a character identity.
    """
    result = {
        "panels": [], "events": [],
        "page_role": "uncertain", "status": "complete", "errors": [],
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
        result["panels"].append({
            "panel_number": number,
            "description": _finish_description(description),
        })

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
        confidence = event.get("confidence_percent", event.get("c"))
        if type(confidence) is not int or not 0 <= confidence <= 100:
            result["errors"].append(
                f"Event {index + 1} needs an integer confidence_percent from 0 to 100."
            )
            continue
        result["events"].append({
            "type": event_type,
            "source_panel": references[0],
            "target_panel": references[1],
            **{key: value.strip() for key, value in field_values.items()},
            "evidence_type": evidence_type,
            "confidence_percent": confidence,
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
