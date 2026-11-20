"""Evidence-grounded, resumable text-only manga recap writer."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time

import config
from common.json_io import write_json
from common.progress import show_progress
from common.utils import natural_sort_key
from description.quality import text_issues
from story_entities.manifest import file_hash
from story_entities.matching import stable_id


PROMPT_VERSION = "grounded_story_writer_v37_quoted_name_boundaries"
STORY_SCHEMA_VERSION = 1
SPEECH_VERBS = (
    r"(?:says?|said|speaks?|shouts?|yells?|warns?|asks?|repl(?:y|ies)|responds?|remarks?|"
    r"adds?|exclaims?|mutters?|whispers?|notes?|declares?|swears?|vows?|promises?|"
    r"announces?|explains?|insists?|admits?|agrees?|answers?|demands?|calls?|cries|"
    r"screams?|tells?|retorts?|protests?|reassures?|repeats?|recounts?|continues?|pleads?|begs?)\b"
)
ANONYMOUS_SPEECH = (
    r"(?:someone(?:\s+else)?|somebody|a\s+voice|another\s+voice)"
    r"(?:\s+(?:[\w-]+ly|nearby|off[- ]?screen|elsewhere|then|still|also))*\s+" + SPEECH_VERBS
)
SOURCE_COMMENTARY = re.compile(
    r"\b(?:unattributed|narration captions?|narrative captions?)\b"
    r"|\b(?:a|an|the)\s+(?:shot|panel|image|picture|close[- ]up)\s+(?:shows?|displays?|depicts?|reveals?|captures?)\b"
    r"|\b(?:a|an|the)\s+(?:shot|close[- ]up)\s+of\b"
    r"|\b(?:a|another|the)\s+view\s+of\b"
    r"|\b(?:the\s+)?scene\s+(?:shows?|displays?|depicts?|reveals?|shifts?|changes?|cuts?)\b"
    r"|\b(?:anonymous|unknown|unidentified)\s+(?:voice|speaker|warning|exclamation)\b"
    r"|\b(?:titles?|signs?|captions?|lettering|speech bubbles?)\s+(?:(?:above|below|nearby|overhead|on[- ]screen)\s+)*(?:says?|reads?|states?|declares?|proclaims?|hails?|identifies?|describes?|explains?)\b"
    r"|\b(?:text|dialogue|narration)\s+(?:reads?|states?|declares?|identifies?|describes?|explains?|includes?)\b"
    r"|\b(?:a|the)\s+sign\s+(?:reads?|states?|declares?)\b"
    r"|\b(?:exclamation|warning|dialogue|text)\s+(?:appears?|is shown|is displayed)\b"
    r"|\bsound effects?\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """\
Write a continuous spoken story using only the supplied accepted facts, in order.
Tell the story in natural third-person present tense, not a list of visible things.
Lead with the situation and stakes when supplied, then follow action, reaction,
and consequence. Connect related beats with ordinary transitions like "but",
"then", and "instead" only when their relationship is supported. Keep meaningful
dialogue and reveals; do not substitute a description of the speaker's appearance.
Do not mention every person or every visual change. Write complete sentences,
vary their length, and make the segments sound like one continuous paragraph.
Describe what changes or matters, not what the drawing contains.
Write one or two connected, complete sentences about the main action, reaction,
or speech. Use roughly 95% he/she or permitted-name references when grounded.
Establish a subject, then prefer pronouns. A brief reference like "the blonde girl
screams" is enough when needed to disambiguate. Do not inventory hair, clothing,
body parts, poses, backgrounds, or every drawn figure. Omit incidental details.
If visual_hold_allowed is true and there is no new meaningful story beat, return
text "" and narration_policy "visual_hold_v1", with empty used_instance_ids and
name_uses. That panel remains visible without a forced caption. Never use a hold
to omit supplied speech, narration, an accepted action, or a transformation.
Otherwise use a short true sentence, not padding or an invented event.
When visual_hold_required is true, return the silent hold, not a visual caption.
Keep each segment under 50 words. Preserve meaningful warnings, threats, setups,
contrasts, reveals, punchlines, and brief interjections. Join related facts without
inventing causes, motives, dialogue, relationships, identity, or transformations.

Evidence and references:
- story_context supplies narration. Start with its key story point, retell it
  directly, and preserve meaningful introductions/background. Do not announce captions.
  Convert first-person narrator context to third person without guessing who "I"
  refers to. If identity is not grounded, preserve only the supported neutral content.
- Descriptions support visible observations. Bind accepted_actions and
  accepted_transformations only to their supplied endpoints. Pending facts are absent
  on purpose. A transcribed claim is speech/thought, not necessarily a true event.
- Use only the exact occurrence's allowed_name. These are neutral aliases, not manga
  names. Do not adopt names from dialogue or introduce "a boy named". Use an alias
  once per segment; withheld aliases must not be reconstructed. allowed_reference is
  a hint, not mandatory wording. He/she needs clear evidence, never an alias or title.
  A demon king need not be male. If gender is unclear, use a short neutral reference.
- Equal non-null entity_id establishes continuity; visual_group_id alone does not.
  Repeated views with the same alias are not extra people. Never invent "another [alias]".
  reference_context is continuity help, not new events or permission to repeat prose.
- Remove camera framing, drawing-scale wording, effect commentary, meaningless glyphs,
  and publication material. Do not invent an in-world cause for drawn effects.
  Distant people are not miniature people. If evidence_density is uncertain, say only
  that details are unclear. Narrate only this panel, never earlier or later events.

Speech:
- A non-null speaker_instance_id assigns that exact line to that occurrence.
  Use its speaker_reference, preferably he/she when clear, or its permitted name.
  Do not replace a known speaker with someone. Prefer concise indirect speech.
- anonymous_dialogue has NO known speaker: prefer a standalone exact quotation
  in double quotes, without "someone says/asks" or any guessed character attribution.
  A quotation can stand alone, e.g. "Pardon?" Keep brief interjections verbatim.
  Never guess a visible character, even if only one is drawn or a nearby line has a
  known speaker. Do not infer a shout from plain speech or an audible voice from a thought.
  Omit unknown-speaker thoughts. Do not move speech between panels or give one speaker
  the entire exchange. Do not merge an unknown line into "he adds/she notes".
  First-person words inside source quotations do not change narrator perspective.

Return one compact JSON object with segments in exactly the requested panel order.
Cover every panel once. Each segment contains:
- primary_panel_id: exact current ID
- supporting_panel_ids: supplied IDs including the primary ID, never future evidence
- text: the narration, with no audience address or source/presentation commentary
- used_instance_ids: exact supplied occurrences actually referred to
- name_uses: [] or {name, instance_id} entries only for permitted names actually spoken
For an allowed silent panel only, also include narration_policy: "visual_hold_v1".
No empty names/IDs, invented IDs, or extra fields.
"""


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_hash(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _words(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:[’'-][\w]+)*", text, flags=re.UNICODE)


def _safe_reference(description: str | None) -> str | None:
    if not isinstance(description, str) or not description.strip():
        return None
    value = re.sub(r"\s+", " ", description).strip().rstrip(".!?")
    if re.match(r"^(?:the|a|an)\b", value, flags=re.IGNORECASE):
        return value
    return "the " + value


def _character_fact(row: dict) -> dict:
    allowed_name = (
        row.get("display_name")
        if row.get("display_name_allowed") else None
    )
    description = row.get("stable_description") or row.get("visual_description")
    reference = allowed_name or _safe_reference(description) or "an unidentified character"
    return {
        "instance_id": row["instance_id"],
        "entity_id": row.get("entity_id"),
        "form_id": row.get("form_id"),
        "visual_group_id": row.get("visual_group_id"),
        "identity_status": row.get("identity_status", "unknown"),
        "allowed_name": allowed_name,
        "allowed_reference": reference,
        "visual_description": description,
        "narration_eligibility": row.get("narration_eligibility", "eligible"),
    }


def _is_narration_eligible(row: dict) -> bool:
    """Keep rejected/ambiguous body proposals out of narrated evidence."""
    return row.get("narration_eligibility", "eligible") in {
        "eligible", "eligible_automatic", "reviewed_eligible",
    }


def prepare_story(manga_dir: Path, chapters: list[Path], output_root: Path) -> dict:
    """Load, join and validate every upstream story-writer input."""
    manga = manga_dir.name
    description_path = output_root / f"{manga}_descriptions.json"
    classification_path = output_root / manga / "character_classification.json"
    description_data = _read_json(description_path)
    classification = _read_json(classification_path)
    if "story_events" not in classification:
        raise ValueError(
            "Character classification predates grounded story events; rerun "
            "test_panels.py, test_description.py, and test_character_reid.py."
        )
    descriptions = description_data.get("descriptions")
    if not isinstance(descriptions, dict):
        raise ValueError("Description export has no manga-wide descriptions object.")
    expected_description_fingerprint = stable_id("descriptions_", descriptions)
    if classification.get("description_fingerprint") != expected_description_fingerprint:
        raise ValueError(
            "Character classification is missing or was built from older descriptions; rerun "
            "test_character_reid.py."
        )
    page_filter = description_data.get("page_filter", {}).get("chapters")
    if not isinstance(page_filter, dict):
        raise ValueError(
            "Description export predates story-page filtering; rerun test_description.py."
        )

    input_files = {
        str(description_path.resolve()): file_hash(description_path),
        str(classification_path.resolve()): file_hash(classification_path),
    }
    character_rows = classification.get("characters", [])
    if not isinstance(character_rows, list):
        raise ValueError("Character classification has no characters list.")
    if any(not isinstance(row, dict) or not isinstance(row.get("instance_id"), str)
           for row in character_rows):
        raise ValueError("Every classified character must have a stable instance ID.")
    characters_by_id = {
        row.get("instance_id"): row for row in character_rows
        if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
    }
    if len(characters_by_id) != len(character_rows):
        raise ValueError("Character classification contains duplicate instance IDs.")
    reviewed_evidence = classification.get("reviewed_evidence", {})
    if not isinstance(reviewed_evidence, dict):
        raise ValueError("Character classification has invalid reviewed evidence.")
    unresolved = reviewed_evidence.get("unresolved_critical_items", [])
    preparation_warnings = [
        "Ignored unresolved optional evidence: "
        + str(row.get("reason") or row.get("fact_id") or row)
        for row in unresolved
        if isinstance(row, dict) and row.get("status", "unresolved") == "unresolved"
    ]

    def reviewed_by_panel(key: str) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        rows = reviewed_evidence.get(key, [])
        if not isinstance(rows, list):
            raise ValueError(f"Reviewed evidence section {key!r} must be a list.")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("panel_id"), str):
                raise ValueError(f"Reviewed evidence section {key!r} has an invalid record.")
            grouped.setdefault(row["panel_id"], []).append(row)
        return grouped

    reviewed_descriptions = reviewed_by_panel("panel_descriptions")
    reviewed_text_facts = reviewed_by_panel("reviewed_text_facts")
    reviewed_narration = reviewed_by_panel("reviewed_narration")
    automatic_text_by_panel: dict[str, list[dict]] = {}
    for row in classification.get("text_proposals", []):
        if not isinstance(row, dict) or row.get("status") != "accepted_automatic":
            continue
        if (row.get("acceptance_policy") not in {
                    "grounded_text_transcription_v1", "grounded_text_transcription_format_v2",
                }
                or not isinstance(row.get("panel_id"), str)
                or not isinstance(row.get("proposal_id"), str)):
            raise ValueError("Automatically accepted text has invalid acceptance evidence.")
        automatic_text_by_panel.setdefault(row["panel_id"], []).append(row)
    characters_by_panel: dict[str, list[dict]] = {}
    characters_by_grounding: dict[str, list[dict]] = {}
    for row in character_rows:
        panel_id = row.get("panel_id") if isinstance(row, dict) else None
        if isinstance(panel_id, str):
            characters_by_panel.setdefault(panel_id, []).append(row)
        grounding_ids = row.get("grounding_ids", [])
        if not isinstance(grounding_ids, list):
            raise ValueError("Character classification contains invalid grounding IDs.")
        for detection_id in grounding_ids:
            if isinstance(detection_id, str):
                characters_by_grounding.setdefault(detection_id, []).append(row)

    eligible_characters = {
        instance_id: row for instance_id, row in characters_by_id.items()
        if _is_narration_eligible(row)
    }

    accepted_actions: dict[str, list[dict]] = {}
    for event in classification.get("story_events", []):
        if (not isinstance(event, dict)
                or event.get("status") not in {"accepted_automatic", "accepted_reviewed"}):
            continue
        expected_policies = (
            {"reviewed_visible_action_v1"}
            if event.get("status") == "accepted_reviewed"
            else {"grounded_visible_action_v1", "grounded_visible_action_format_v2", "grounded_visible_action_format_v3"}
        )
        panel_id = event.get("panel_id")
        actor_id = event.get("actor_instance_id")
        target_id = event.get("target_instance_id")
        if (panel_id is None or actor_id not in eligible_characters
                or event.get("acceptance_policy") not in expected_policies
                or not isinstance(event.get("text"), str)
                or not event["text"].strip()):
            raise ValueError("Accepted story event has stale panel/actor evidence.")
        if target_id is not None and target_id not in eligible_characters:
            raise ValueError("Accepted story event has a stale target instance.")
        if characters_by_id[actor_id].get("panel_id") != panel_id:
            raise ValueError("Accepted story event actor is not in its supporting panel.")
        if target_id is not None and characters_by_id[target_id].get("panel_id") != panel_id:
            raise ValueError("Accepted story event target is not in its supporting panel.")
        accepted_actions.setdefault(panel_id, []).append({
            "event_id": event.get("event_id"),
            "text": event.get("text"),
            "actor_instance_id": actor_id,
            "actor_reference": _character_fact(characters_by_id[actor_id])["allowed_reference"],
            "target_instance_id": target_id,
            "target_reference": (_character_fact(characters_by_id[target_id])["allowed_reference"]
                                 if target_id else None),
            "acceptance_policy": event.get("acceptance_policy"),
        })

    accepted_transformations: dict[str, list[dict]] = {}
    for event in classification.get("transformation_proposals", []):
        if (not isinstance(event, dict)
                or event.get("status") not in {"accepted_automatic", "accepted_reviewed"}):
            continue
        reviewed = event.get("status") == "accepted_reviewed"
        if ((not reviewed and event.get("confidence_percent") != 100)
                or (not reviewed and event.get("acceptance_policy")
                    != "qwen_direct_visible_change_v1")
                or (reviewed and event.get("acceptance_policy")
                    != "reviewed_transformation_v1")
                or not isinstance(event.get("evidence"), str)
                or not event["evidence"].strip()):
            raise ValueError("Accepted transformation lacks strict acceptance evidence.")
        endpoints = event.get("accepted_instance_ids", [])
        if (not isinstance(endpoints, list) or len(endpoints) != 2
                or any(instance_id not in eligible_characters for instance_id in endpoints)):
            raise ValueError("Accepted transformation has stale endpoint instances.")
        source, target = [characters_by_id[instance_id] for instance_id in endpoints]
        target_panel_id = target.get("panel_id")
        if not isinstance(target_panel_id, str):
            raise ValueError("Accepted transformation has no target panel ID.")
        accepted_transformations.setdefault(target_panel_id, []).append({
            "event_id": event.get("event_id"),
            "type": event.get("type"),
            "source_instance_id": source["instance_id"],
            "source_panel_id": source.get("panel_id"),
            "target_instance_id": target["instance_id"],
            "target_panel_id": target_panel_id,
            "source_reference": _character_fact(source)["allowed_reference"],
            "target_reference": _character_fact(target)["allowed_reference"],
            "evidence": event.get("evidence"),
            "confidence_percent": event.get("confidence_percent"),
            "acceptance_policy": event.get("acceptance_policy"),
        })

    panels = []
    seen_panel_ids = set()
    global_order = 0
    for chapter in sorted(chapters, key=natural_sort_key):
        chapter_output = output_root / manga / chapter.name
        manifest_path = chapter_output / "panel_manifest.json"
        manifest = _read_json(manifest_path)
        input_files[str(manifest_path.resolve())] = file_hash(manifest_path)
        chapter_descriptions = descriptions.get(chapter.name, {})
        if not isinstance(chapter_descriptions, dict):
            raise ValueError(f"Invalid description chapter: {chapter.name}")
        chapter_filter = page_filter.get(chapter.name)
        if not isinstance(chapter_filter, dict):
            raise ValueError(f"Missing story-page filter chapter: {chapter.name}")
        crop_index = 0
        for page in sorted(manifest.get("pages", []), key=lambda row: row.get("page_order", 0)):
            page_name = Path(page.get("page", "")).name
            filter_record = chapter_filter.get(page_name)
            if not isinstance(filter_record, dict):
                raise ValueError(
                    f"Missing story-page decision: {chapter.name}/{page_name}. "
                    "Resume main.py with --start-at description to complete page analysis first."
                )
            if (filter_record.get("source_hash") != page.get("source_hash")
                    or filter_record.get("geometry_revision") != page.get("geometry_revision")
                    or filter_record.get("grounding_revision") != page.get("grounding_revision")):
                raise ValueError(f"Stale story-page decision: {chapter.name}/{page_name}")
            if filter_record.get("extraction_status") not in {
                "complete", "complete_with_uncertainty",
            }:
                raise ValueError(
                    f"Page analysis is incomplete for {chapter.name}/{page_name}; "
                    "rerun test_description.py before writing the story."
                )
            included_in_story = filter_record.get("included_in_story") is True
            page_description = chapter_descriptions.get(page_name)
            panel_descriptions = {}
            if included_in_story:
                if not isinstance(page_description, dict):
                    raise ValueError(f"Missing included story page: {chapter.name}/{page_name}")
                if (page_description.get("extraction_status") not in {
                        "complete", "complete_with_uncertainty",
                    }
                        or page_description.get("page_role") != "story"):
                    raise ValueError(f"Included description is not a complete story page: {chapter.name}/{page_name}")
                if (page_description.get("source_hash") != page.get("source_hash")
                        or page_description.get("geometry_revision") != page.get("geometry_revision")
                        or page_description.get("grounding_revision") != page.get("grounding_revision")):
                    raise ValueError(f"Stale description/manifest evidence: {chapter.name}/{page_name}")
                panel_descriptions = {
                    item.get("panel_number"): item
                    for item in page_description.get("panels", [])
                    if isinstance(item, dict)
                }
                manifest_panels = {
                    item.get("panel_number"): item.get("panel_id")
                    for item in page.get("panels", [])
                    if isinstance(item, dict)
                }
                character_labels = page.get("character_labels")
                if not isinstance(character_labels, list):
                    raise ValueError(
                        f"Panel manifest predates grounded character labels: "
                        f"{chapter.name}/{page_name}"
                    )
                for label in character_labels:
                    if not isinstance(label, dict) or label.get("panel_number") is None:
                        continue
                    detection_id = label.get("detection_id")
                    expected_panel_id = manifest_panels.get(label.get("panel_number"))
                    matching = [
                        row for row in characters_by_grounding.get(detection_id, [])
                        if row.get("panel_id") == expected_panel_id
                    ]
                    if not isinstance(detection_id, str) or len(matching) != 1:
                        preparation_warnings.append(
                            "Ignored character label without one current stable instance: "
                            f"{chapter.name}/{page_name}/{label.get('label')}"
                        )
            elif page_description is not None:
                raise ValueError(
                    f"Excluded page is unexpectedly present in story descriptions: "
                    f"{chapter.name}/{page_name}"
                )
            source_path = Path(page["page"])
            if included_in_story:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                if file_hash(source_path) != page.get("source_hash"):
                    raise ValueError(f"Source page changed: {source_path}")
            for panel in sorted(page.get("panels", []), key=lambda row: row.get("panel_number", 0)):
                crop_index += 1
                if not included_in_story:
                    continue
                number = panel.get("panel_number")
                described = panel_descriptions.get(number)
                if not described or not isinstance(described.get("description"), str):
                    raise ValueError(f"Missing panel description: {chapter.name}/{page_name}#{number}")
                issues = text_issues(described["description"])
                if issues:
                    raise ValueError(f"Invalid panel description format: {chapter.name}/{page_name}#{number}")
                if "actions" not in described:
                    preparation_warnings.append(
                        f"Panel has no structured action field: "
                        f"{chapter.name}/{page_name}#{number}"
                    )
                    described = dict(described)
                    described["actions"] = []
                panel_id = panel.get("panel_id")
                if not isinstance(panel_id, str) or panel_id in seen_panel_ids:
                    raise ValueError(f"Missing or duplicate panel ID: {chapter.name}/{page_name}#{number}")
                seen_panel_ids.add(panel_id)
                evidence_rows = (
                    reviewed_descriptions.get(panel_id, [])
                    + reviewed_text_facts.get(panel_id, [])
                    + reviewed_narration.get(panel_id, [])
                )
                for evidence in evidence_rows:
                    if (evidence.get("status") != "accepted_reviewed"
                            or evidence.get("source_hash") != page.get("source_hash")
                            or evidence.get("geometry_revision") != page.get("geometry_revision")
                            or evidence.get("grounding_revision") != page.get("grounding_revision")):
                        raise ValueError(f"Stale reviewed evidence for panel {panel_id}.")
                corrections = reviewed_descriptions.get(panel_id, [])
                if len(corrections) > 1:
                    raise ValueError(f"Multiple reviewed descriptions for panel {panel_id}.")
                if corrections:
                    corrected = corrections[0].get("text")
                    if not isinstance(corrected, str) or text_issues(corrected):
                        raise ValueError(f"Invalid reviewed description for panel {panel_id}.")
                    described = dict(described)
                    described.update(description=corrected.strip(), status="reviewed")
                text_facts = []
                for fact in automatic_text_by_panel.get(panel_id, []):
                    if (fact.get("chapter") != chapter.name
                            or fact.get("page") != page_name
                            or fact.get("source_hash") != page.get("source_hash")
                            or fact.get("geometry_revision") != page.get("geometry_revision")
                            or fact.get("grounding_revision") != page.get("grounding_revision")
                            or fact.get("kind") not in {
                                "narration", "dialogue", "thought", "sign", "sound_effect", "publication",
                            }
                            or not isinstance(fact.get("text"), str)
                            or text_issues(fact["text"])):
                        raise ValueError(
                            f"Automatically accepted text is stale or invalid for {panel_id}."
                        )
                    speaker_id = fact.get("speaker_instance_id")
                    if (speaker_id is not None
                            and (speaker_id not in eligible_characters
                                 or eligible_characters[speaker_id].get("panel_id") != panel_id)):
                        raise ValueError(
                            f"Automatically accepted text has an unavailable speaker on {panel_id}."
                        )
                    text_facts.append({
                        "fact_id": fact["proposal_id"], "kind": fact["kind"],
                        "text": fact["text"],
                        "speaker_instance_id": speaker_id,
                        "truth_status": "transcribed_statement_only",
                        "acceptance_policy": fact["acceptance_policy"],
                    })
                for fact in reviewed_text_facts.get(panel_id, []):
                    if (not isinstance(fact.get("fact_id"), str)
                            or fact.get("kind") not in {
                                "narration", "dialogue", "thought", "sign", "sound_effect", "publication",
                            }
                            or not isinstance(fact.get("text"), str)
                            or not fact["text"].strip()):
                        raise ValueError(f"Invalid reviewed text fact for panel {panel_id}.")
                    speaker_id = fact.get("speaker_instance_id")
                    if (speaker_id is not None
                            and (speaker_id not in eligible_characters
                                 or eligible_characters[speaker_id].get("panel_id") != panel_id)):
                        raise ValueError(
                            f"Reviewed text fact has an unavailable speaker on panel {panel_id}."
                        )
                    text_facts.append({
                        "fact_id": fact["fact_id"], "kind": fact["kind"],
                        "text": fact["text"].strip(),
                        "speaker_instance_id": speaker_id,
                        "truth_status": fact.get("truth_status", "statement_only"),
                        "acceptance_policy": "reviewed_text_fact_v1",
                    })
                narration_rows = reviewed_narration.get(panel_id, [])
                if len(narration_rows) > 1:
                    raise ValueError(f"Multiple reviewed narrations for panel {panel_id}.")
                crop_path = chapter_output / f"panel_{crop_index:04d}.jpg"
                if not crop_path.is_file():
                    raise FileNotFoundError(crop_path)
                global_order += 1
                raw_panel_characters = characters_by_panel.get(panel_id, [])
                if any(
                    row.get("chapter") != chapter.name
                    or Path(row.get("page", "")).name != page_name
                    or row.get("page_hash") != page.get("source_hash")
                    or row.get("geometry_revision") != page.get("geometry_revision")
                    or row.get("grounding_revision") != page.get("grounding_revision")
                    for row in raw_panel_characters
                ):
                    raise ValueError(
                        f"Character classification is stale for panel {panel_id}."
                    )
                panel_characters = sorted(
                    (_character_fact(row) for row in raw_panel_characters
                     if _is_narration_eligible(row)),
                    key=lambda row: row["instance_id"],
                )
                panels.append({
                    "order": global_order,
                    "chapter": chapter.name,
                    "page": page_name,
                    "panel_number": number,
                    "panel_id": panel_id,
                    "source_page": str(source_path.resolve()),
                    "panel_image": str(crop_path.resolve()),
                    "description": described["description"],
                    "description_status": described.get("status", "usable"),
                    "characters": panel_characters,
                    "accepted_actions": sorted(
                        accepted_actions.get(panel_id, []),
                        key=lambda row: row.get("event_id") or "",
                    ),
                    "accepted_transformations": sorted(
                        accepted_transformations.get(panel_id, []),
                        key=lambda row: row.get("event_id") or "",
                    ),
                    "accepted_text_facts": text_facts,
                    "reviewed_narration": (
                        narration_rows[0].get("text") if narration_rows else None
                    ),
                })
                panels[-1]["evidence_density"] = (
                    "uncertain" if described.get("status") == "ambiguous" else
                    "sparse" if not panel_characters
                    and not panels[-1]["accepted_actions"]
                    and not panels[-1]["accepted_transformations"]
                    and len(_words(described["description"])) <= 8
                    else "ordinary"
                )

    stale_action_panels = set(accepted_actions) - seen_panel_ids
    stale_transformation_panels = set(accepted_transformations) - seen_panel_ids
    stale_text_panels = set(automatic_text_by_panel) - seen_panel_ids
    stale_transformation_panels.update(
        event.get("source_panel_id")
        for events in accepted_transformations.values() for event in events
        if event.get("source_panel_id") not in seen_panel_ids
    )
    if stale_action_panels or stale_transformation_panels or stale_text_panels:
        raise ValueError(
            "Accepted events or text reference pages excluded from the current story; "
            "rerun test_character_reid.py."
        )
    if not panels:
        raise ValueError("No complete story panels are available.")

    guide = {}
    for panel in panels:
        for character in panel["characters"]:
            subject = (character.get("entity_id"), character.get("form_id"),
                       character.get("visual_group_id"), character["allowed_reference"])
            guide.setdefault(subject, {
                "allowed_reference": character["allowed_reference"],
                "allowed_name": character["allowed_name"],
                "visual_description": character["visual_description"],
                "entity_id": character["entity_id"],
                "form_id": character["form_id"],
                "visual_group_id": character["visual_group_id"],
            })

    settings = {
        "narration_words_per_minute": config.STORY_NARRATION_WPM,
        "target_seconds_per_panel": config.STORY_SECONDS_PER_PANEL,
        "single_request_input_tokens": config.STORY_SINGLE_REQUEST_TOKENS,
        "chunk_target_input_tokens": config.STORY_CHUNK_TARGET_TOKENS,
        "chunk_hard_input_tokens": config.STORY_CHUNK_HARD_TOKENS,
        "chunk_max_panels": config.STORY_CHUNK_MAX_PANELS,
        "context_token_budget": config.STORY_CONTEXT_TOKEN_BUDGET,
        "output_headroom_tokens": config.STORY_OUTPUT_HEADROOM_TOKENS,
        "context_words": config.STORY_CONTEXT_WORDS,
        "do_sample": False,
        "thinking": False,
        "backend": "llama.cpp_native_schema_v1",
        "model_path": str(config.STORY_MODEL_PATH),
    }
    for label, path in (("model", config.STORY_MODEL_PATH), ("runtime", config.STORY_LLAMA_SERVER)):
        path = Path(path)
        settings[label + "_artifact"] = (
            {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            if path.is_file() else {"path": str(path.resolve()), "missing": True}
        )
    fingerprint_payload = {
        "inputs": input_files,
        "identity_revision": classification.get("revision"),
        "canonical_names": sorted({row["name"] for row in classification.get("identities", []) if row.get("name")}),
        "prompt_version": PROMPT_VERSION,
        "settings": settings,
    }
    return {
        "manga": manga,
        "panels": panels,
        "character_guide": sorted(guide.values(), key=lambda row: (
            row.get("allowed_name") or "", row["allowed_reference"],
            row.get("form_id") or "", row.get("visual_group_id") or "",
        )),
        "identity_revision": classification.get("revision"),
        "input_files": input_files,
        "canonical_names": fingerprint_payload["canonical_names"],
        "settings": settings,
        "input_fingerprint": _stable_hash(fingerprint_payload),
        "model_version": Path(config.STORY_MODEL_PATH).name,
        "warnings": preparation_warnings,
    }


def _narrative_observation(text: str) -> str:
    """Remove explicit drawing conventions only from temporary writer input."""
    text = re.sub(r"\b(?:small|smaller|tiny|miniature)\s+version\s+of\s+(?:a|the)\s+(?=character\b)",
                  "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:small|smaller|tiny)\s+(?=group\s+of\s+(?:people|characters|figures|students)\b)",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:another\s+(?:full[- ]body\s+)?view\s+of\s+the\s+same\s+character)[^.!?]*[.!?]?",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:amidst|amid|among|with|and|surrounded by)\s+"
        r"(?:(?:electrical|lightning|impact|visual|drawing|motion|speed)\s+(?:and\s+)?)+effects?"
        r"(?:\s+around\s+(?:them|him|her))?",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:small|smaller|tiny|miniature)\s+(?=(?:people|characters?|figures?|students?|boys?|girls?|men|women)\b)",
        "", text, flags=re.IGNORECASE,
    )
    return " ".join(text.split())


def _brief_appearance(description: str | None) -> str:
    description = description or ""
    if re.search(r"\b(?:group|people|characters|figures|students|boys|girls|men|women)\b", description, re.IGNORECASE):
        return "the group"
    role = re.search(r"\b(?:girl|woman|boy|man|maid|creature|child|student|guard|knight|soldier|person|character)\b",
                     description, re.IGNORECASE)
    noun = role.group().lower() if role else "character"
    trait = ""
    hair = re.search(r"\b(dark|black|white|light)(?:[- ]haired|\s+(?:\w+\s+){0,2}hair)\b", description, re.IGNORECASE)
    if re.search(r"\bblond(?:e)?\b", description, re.IGNORECASE):
        trait = "blonde "
    elif hair:
        trait = hair[1].lower() + "-haired "
    elif re.search(r"\b(?:horns|horned)\b", description, re.IGNORECASE):
        trait = "horned "
    return "the " + trait + noun


def _brief_character_reference(character: dict, panel: dict) -> str:
    reference = character.get("allowed_reference")
    if reference in {"he", "she"}:
        return reference
    description = character.get("visual_description") or ""
    if _brief_appearance(description) == "the group":
        return "the group"
    female = bool(re.search(r"\b(?:female|woman|girl|she|her)\b", description, re.IGNORECASE))
    male = bool(re.search(r"\b(?:male|man|boy|he|him|his)\b", description, re.IGNORECASE))
    if not female and not male and len(panel["characters"]) == 1:
        female = bool(re.search(r"\b(?:female|woman|girl|she|her)\b", panel.get("description", ""), re.IGNORECASE))
        male = bool(re.search(r"\b(?:male|man|boy|he|him|his)\b", panel.get("description", ""), re.IGNORECASE))
    return "she" if female and not male else "he" if male and not female else _brief_appearance(description)


def _panel_prompt_record(panel: dict, canonical_names: list[str] | None = None) -> dict:
    facts = []
    characters = {c["instance_id"]: c for c in panel["characters"]}
    def event_references(event):
        event = dict(event)
        for endpoint in ("actor", "target", "source"):
            character = characters.get(event.get(endpoint + "_instance_id"))
            if character:
                event[endpoint + "_reference"] = character.get("allowed_name") or _brief_character_reference(character, panel)
        return event
    for original in panel.get("accepted_text_facts", []):
        fact = dict(original)
        speaker = characters.get(fact.get("speaker_instance_id"))
        if speaker and fact["kind"] in {"dialogue", "thought"}:
            fact["speaker_reference"] = _brief_character_reference(speaker, panel)
        elif fact["kind"] == "dialogue" and fact.get("speaker_instance_id") is None:
            fact["speaker_reference"] = None
        for name in canonical_names or []:
            introduction = re.fullmatch(rf"\s*I\s+AM\s+{re.escape(name)}\s*,?\s*(.*?)\s*",
                                        fact["text"], flags=re.IGNORECASE)
            if not introduction or fact["kind"] != "dialogue":
                continue
            alias = characters.get(fact.get("speaker_instance_id"), {}).get("allowed_name")
            if alias:
                fact["text"] = re.sub(rf"\b{re.escape(name)}\b", lambda _: alias, fact["text"], flags=re.IGNORECASE)
            elif introduction[1].strip(" .!?,"):
                fact["text"] = "I am " + introduction[1].lstrip(", ")
            else:
                fact["text"] = "The speaker introduces themself."
        facts.append(fact)
    return {
        "chapter": panel["chapter"],
        "page": panel["page"],
        "panel_number": panel["panel_number"],
        "panel_id": panel["panel_id"],
        "story_context": [fact["text"] for fact in facts if fact["kind"] == "narration"],
        "description": _narrative_observation(panel["description"]),
        "characters": [dict(character,
                            visual_description=_brief_appearance(character.get("visual_description")),
                            allowed_reference=character.get("allowed_name") or _brief_character_reference(character, panel),
                            **{key: stable_id("reference_", key, character[key]) if character.get(key) else None
              for key in ("entity_id", "form_id")}) for character in panel["characters"]],
        "accepted_actions": [event_references(event) for event in panel["accepted_actions"]],
        "accepted_transformations": [event_references(event) for event in panel["accepted_transformations"]],
        "accepted_text_facts": [fact for fact in facts
                                 if fact["kind"] not in {"publication", "sound_effect", "narration"}
                                 and not (fact["kind"] in {"dialogue", "thought"}
                                          and fact.get("speaker_instance_id") is None)],
        "anonymous_dialogue": [fact for fact in facts
                                 if fact["kind"] == "dialogue"
                                 and fact.get("speaker_instance_id") is None],
        "evidence_density": panel.get("evidence_density", "ordinary"),
        "visual_hold_allowed": _visual_hold_allowed(panel),
        "visual_hold_required": _static_visual_hold(panel),
    }


def _visual_hold_allowed(panel: dict) -> bool:
    """No accepted story event or meaningful text may disappear into a visual hold."""
    return (panel.get("evidence_density", "ordinary") != "uncertain"
            and not panel.get("reviewed_narration")
            and not panel.get("accepted_actions") and not panel.get("accepted_transformations")
            and not any(fact.get("kind") not in {"publication", "sound_effect"}
                        for fact in panel.get("accepted_text_facts", [])))


def _static_visual_hold(panel: dict) -> bool:
    """Only explicit static fragments, never a repair for missing action/speech evidence."""
    # shortcut: three observed static forms only; replace with reviewed scene beats for broader selection.
    return _visual_hold_allowed(panel) and bool(re.fullmatch(
        r"(?:lower body and legs of a person standing|"
        r"a character standing in a doorway(?: in the background)?|"
        r"a close[- ]up(?: view)? of (?:blond(?:e)?|dark|light|black|white)(?:[- ]haired)? hair)\.?",
        panel.get("description", "").strip(), flags=re.IGNORECASE,
    ))


def _reference_panels(panels: list[dict], previous_segments: list[dict] | None = None,
                      *, suppress_names: bool = False) -> list[dict]:
    """Withhold recently spoken aliases from both the prompt and output schema."""
    recent_names = {use["name"] for segment in (previous_segments or [])[-2:]
                    for use in segment.get("name_uses", [])}
    replacements = {}
    for panel in panels:
        for character in panel["characters"]:
            name = character.get("allowed_name")
            if not name or not (suppress_names or name in recent_names):
                continue
            replacements[name] = _brief_character_reference(character, panel)

    def rewrite(value):
        if isinstance(value, dict):
            return {key: (None if key == "allowed_name" and item in replacements else rewrite(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            for name, reference in replacements.items():
                value = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", lambda _: reference, value, flags=re.IGNORECASE)
        return value

    return rewrite(panels)


def build_story_prompt(
    panels: list[dict], *, outline: str = "", character_guide: list[dict] | None = None,
    completed_panels: list[dict], previous_narration: str, tokenizer,
    previous_segments: list[dict] | None = None,
    suppress_names: bool = False,
    canonical_names: list[str] | None = None,
) -> str:
    panels = _reference_panels(panels, previous_segments, suppress_names=suppress_names)
    covered = [panel["panel_id"] for panel in completed_panels]
    current_aliases = {c.get("allowed_name") for panel in panels for c in panel["characters"]} - {None}
    earlier_characters = {
        c["instance_id"]: c for panel in completed_panels for c in panel["characters"]
    }
    last_subjects = {
        earlier_characters[instance_id].get("allowed_name")
        for instance_id in ((previous_segments or [{}])[-1].get("used_instance_ids", []))
        if instance_id in earlier_characters
    }
    context = {
        "covered_panels": {"count": len(covered), "rule": "Already committed; do not generate them again."},
        "reference_context": {
            "previously_used_aliases": sorted({
                use["name"] for segment in previous_segments or []
                for use in segment.get("name_uses", []) if use["name"] in current_aliases
            }),
            "last_unambiguous_subject": (next(iter(last_subjects))
                                         if len(last_subjects) == 1 and last_subjects <= current_aliases else None),
        },
        # Previous prose was copied into unrelated panels; accepted IDs carry continuity instead.
        "next_panels": [_panel_prompt_record(panel, canonical_names) for panel in panels],
    }
    contract = {
        "segments": [{
            "primary_panel_id": "exact supplied panel_id",
            "supporting_panel_ids": ["one or more supplied panel IDs"],
            "text": "third-person present-tense narration",
            "used_instance_ids": ["exact instance IDs actually referenced"],
            "name_uses": [],
        }]
    }
    return (
        SYSTEM_PROMPT
        + "\nRequired output shape:\n"
        + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
        + "\nGrounded input:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\nGenerate exactly " + str(len(panels)) + " segments, ONLY for these primary_panel_ids in this order: "
        + json.dumps([panel["panel_id"] for panel in panels])
    )


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Story response root must be an object.")
    return value


def validate_story_response(
    raw: str,
    expected_panels: list[dict],
    *,
    known_names: set[str] | None = None,
    supporting_panels: list[dict] | None = None,
    previous_segments: list[dict] | None = None,
) -> tuple[list[dict], list[str]]:
    """Validate panel coverage, references, names and basic narration constraints."""
    errors = []
    previous_texts = {}
    for row in previous_segments or []:
        previous_texts.setdefault(tuple(_words(row["text"].casefold())), []).append(row.get("primary_panel_id"))
    try:
        payload = _extract_json(raw)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return [], [f"Invalid JSON: {error}"]
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return [], ["Missing segments list."]
    expected_ids = [panel["panel_id"] for panel in expected_panels]
    returned_ids = [segment.get("primary_panel_id") if isinstance(segment, dict) else None
                    for segment in segments]
    if returned_ids != expected_ids:
        errors.append("Segments must cover every requested panel exactly once and in order.")
    support_rows = supporting_panels if supporting_panels is not None else expected_panels
    panel_by_id = {
        panel["panel_id"]: panel
        for panel in support_rows
    }
    panel_positions = {
        panel["panel_id"]: panel.get("order", index)
        for index, panel in enumerate(support_rows)
    }
    all_instance_facts = {
        character["instance_id"]: character
        for panel in panel_by_id.values() for character in panel["characters"]
    }
    all_allowed_names = {
        fact["allowed_name"] for fact in all_instance_facts.values()
        if fact.get("allowed_name")
    }
    all_allowed_names.update(known_names or set())
    normalized = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(f"Segment {index + 1} must be an object.")
            continue
        primary = segment.get("primary_panel_id")
        text = segment.get("text")
        supporting = segment.get("supporting_panel_ids")
        used_instances = segment.get("used_instance_ids")
        name_uses = segment.get("name_uses")
        if segment.get("narration_policy") is not None:
            from story_writer.display import is_visual_hold
            if (is_visual_hold(segment) and primary in panel_by_id
                    and _visual_hold_allowed(panel_by_id[primary]) and supporting == [primary]):
                normalized.append({key: segment[key] for key in (
                    "primary_panel_id", "supporting_panel_ids", "text", "used_instance_ids", "name_uses", "narration_policy")})
                continue
            errors.append(f"Segment {index + 1} has an invalid visual hold; accepted story facts must be narrated.")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"Segment {index + 1} has no narration text.")
            continue
        accepted_fallback = False
        if segment.get("fallback_policy") is not None:
            try:
                expected_fallback = _restricted_fallback(panel_by_id[primary])
                accepted_fallback = all(segment.get(key) == value for key, value in expected_fallback.items())
            except (KeyError, ValueError):
                pass
            if not accepted_fallback:
                errors.append(f"Segment {index + 1} does not match its accepted fallback evidence.")
        count = len(_words(text))
        current_panel = panel_by_id.get(primary, {})
        if _static_visual_hold(current_panel):
            errors.append(f"Segment {index + 1} is a static detail panel; use its required visual hold instead of a caption.")
        dialogue = [fact for fact in current_panel.get("accepted_text_facts", []) if fact["kind"] == "dialogue"]
        unknown_lines = {tuple(_words(fact["text"].casefold().replace("’", "'"))) for fact in dialogue
                         if fact.get("speaker_instance_id") is None and _words(fact["text"])}
        standalone_quotes = {tuple(_words(match[1].casefold().replace("’", "'"))) for match in re.finditer(
            r'(?:^|(?<=[.!?]["”])\s+|(?<=[.!?])\s+)["“]([^"”]+)["”](?=$|\s+[A-Z“"])', text,
        )}
        quoted_unknown = bool(unknown_lines) and unknown_lines <= standalone_quotes
        anonymous_speech = re.search(
            r"\b" + ANONYMOUS_SPEECH,
            text, re.IGNORECASE,
        )
        if (any(fact.get("speaker_instance_id") is None and _words(fact["text"]) for fact in dialogue)
                and not anonymous_speech and not quoted_unknown):
            errors.append(
                f"Segment {index + 1} does not keep unknown dialogue explicitly anonymous. "
                "Keep that unknown line as a standalone exact quotation in double quotes without a speaker tag; "
                "never merge it into a known speaker's speech."
            )
        if dialogue and all(fact.get("speaker_instance_id") is not None for fact in dialogue):
            anonymous_speech = re.search(
                r"(?:^|[.!?]\s+)" + ANONYMOUS_SPEECH,
                text, re.IGNORECASE,
            )
            if anonymous_speech:
                errors.append(
                    f"Segment {index + 1} anonymizes known speech. All dialogue here has a supplied speaker; "
                    "use the corresponding he/she or neutral speaker reference, not someone else."
                )
        for fact in current_panel.get("accepted_text_facts", []):
            if fact["kind"] != "dialogue" or fact.get("speaker_instance_id") is not None:
                continue
            interjection = re.fullmatch(r"\s*([A-Za-z]+)\s*[!?]+\s*", fact["text"])
            if interjection and not re.search(rf"\b{re.escape(interjection[1])}\b", text, re.IGNORECASE):
                errors.append(
                    f"Segment {index + 1} omits brief anonymous dialogue {fact['text']!r}. "
                    "Keep it as a short anonymous quotation in this panel; "
                    "do not move it or assign a visible speaker."
                )
        matching_panels = previous_texts.get(tuple(_words(text.casefold())), [])
        description = " ".join(current_panel.get("description", "").split()).casefold()
        same_description = description and any(
            description == " ".join(panel_by_id.get(panel_id, {}).get("description", "").split()).casefold()
            for panel_id in matching_panels
        )
        if count >= 12 and matching_panels and not same_description:
            errors.append(
                f"Segment {index + 1} repeats previously completed narration verbatim. "
                "Narrate only this panel's supplied facts, not the previous-narration context."
            )
        presentation = None
        for candidate in SOURCE_COMMENTARY.finditer(text):
            physical_sign = re.fullmatch(r"(?:(?:a|the)\s+)?sign\s+reads?", candidate.group(), re.IGNORECASE)
            grounded_sign = physical_sign and any(
                fact["kind"] == "sign" and fact["text"].strip()
                and re.match(r"\s*[:,'\"“‘]*\s*" + re.escape(fact["text"].strip()) + r"(?!\w)",
                             text[candidate.end():], flags=re.IGNORECASE)
                for fact in current_panel.get("accepted_text_facts", [])
            )
            if not grounded_sign:
                presentation = candidate
                break
        if presentation:
            errors.append(
                f"Segment {index + 1}: {presentation.group()!r} reports source presentation. "
                "Remove the framing or attribution label and retain the subject and content directly; "
                "keep unknown speech as an unassigned quotation. Do not invent or omit story facts."
            )
        if re.search(r"\b(?:close[- ]up view|in profile|(?:legs|body parts) (?:are )?visible|caption box (?:says|declares)|story reveals)\b", text, re.IGNORECASE):
            errors.append(f"Segment {index + 1} reports camera framing or source presentation; narrate the story beat, or use an allowed visual hold.")
        if re.search(r"\b(?:camera (?:focuses|shows)|(?:speaks|stands) against a .* background)\b", text, re.IGNORECASE):
            errors.append(f"Segment {index + 1} reports camera framing or source presentation; keep the event, not its background.")
        from story_writer.evidence import outside_quotations
        narrator_text, quote_errors = outside_quotations(text)
        errors.extend(f"Segment {index + 1}: {error}" for error in quote_errors)
        if re.search(r"\b(?:I|we|our|ours)\b", narrator_text, re.IGNORECASE):
            errors.append(f"Segment {index + 1} uses first-person narration outside quotations; retell narrator context in third person without guessing an identity.")
        style_error = re.search(
            r"\b(?:small|tiny|miniature)\s+(?:people|characters?|figures?|students?|boys?|girls?|men|women)\b"
            r"|\b(?:boy|girl|man|woman|person|character|student)\s+(?:named|called)\b"
            r"|\b(?:lightning|impact|electrical|visual|drawing|motion|speed)\s+effects?\b",
            text, re.IGNORECASE,
        )
        if style_error:
            errors.append(
                f"Segment {index + 1}: {style_error.group()!r} uses drawing-scale, effects, or name-introduction wording. "
                "Use the alias directly, omit drawing effects, and call distant people people in the background."
            )
        for name in {c.get("allowed_name") for c in current_panel.get("characters", [])} - {None}:
            occurrences = re.findall(rf"(?<!\w){re.escape(name)}(?!\w)", narrator_text, re.IGNORECASE)
            recent_names = {use["name"] for segment in ((previous_segments or []) + normalized)[-2:]
                            for use in segment.get("name_uses", [])}
            if len(occurrences) > 1 or (occurrences and name in recent_names):
                errors.append(
                    f"Segment {index + 1} violates name-frequency limits: one use per alias per segment, "
                    "then two segments without that alias. Use clear pronouns or supplied visual references."
                )
            # A busy exchange may need repeated names; style alone must not deadlock generation.
            if re.search(rf"\banother\s+{re.escape(name)}\b", text, re.IGNORECASE):
                errors.append(
                    f"Segment {index + 1} repeats or duplicates alias {name!r}. "
                    "Describe the character once; prefer clear pronouns after establishing the subject."
                )
        short_unknown_quote = quoted_unknown and re.fullmatch(r'["“][^"”]+["”]', text.strip())
        if not accepted_fallback and ((count < 3 and not short_unknown_quote) or count > 150):
            errors.append(f"Segment {index + 1} has {count} words; allowed range is 3-150 words.")
        if (not isinstance(supporting, list) or primary not in supporting
                or len(supporting) != len(set(supporting))
                or any(panel_id not in panel_by_id for panel_id in supporting)):
            errors.append(f"Segment {index + 1} has invalid supporting panel IDs.")
            supporting = [primary] if primary in panel_by_id else []
        primary_position = panel_positions.get(primary, -1)
        if any(panel_positions.get(panel_id, primary_position + 1) > primary_position
               for panel_id in supporting):
            errors.append(f"Segment {index + 1} cites evidence from a future panel.")
            supporting = [
                panel_id for panel_id in supporting
                if panel_positions.get(panel_id, primary_position + 1) <= primary_position
            ]
        available_instances = {
            character["instance_id"]
            for panel_id in supporting for character in panel_by_id[panel_id]["characters"]
        }
        if (not isinstance(used_instances, list)
                or len(used_instances) != len(set(used_instances))
                or any(instance_id not in available_instances for instance_id in used_instances)):
            errors.append(f"Segment {index + 1} has invalid used_instance_ids.")
            used_instances = []
        declared_names = []
        if not isinstance(name_uses, list):
            errors.append(f"Segment {index + 1} has no valid name_uses list.")
            name_uses = []
        for use in name_uses:
            if not isinstance(use, dict):
                errors.append(f"Segment {index + 1} has a malformed name use.")
                continue
            name, instance_id = use.get("name"), use.get("instance_id")
            fact = all_instance_facts.get(instance_id)
            if (instance_id not in used_instances or not isinstance(name, str)
                    or fact is None or fact.get("allowed_name") != name):
                errors.append(f"Segment {index + 1} uses an unapproved name.")
                continue
            if not re.search(rf"(?<!\w){re.escape(name)}(?!\w)", narrator_text, flags=re.IGNORECASE):
                errors.append(f"Segment {index + 1} declares a name that is not in its text.")
                continue
            declared_names.append(name)
        source_speech = {tuple(_words(f["text"].casefold().replace("’", "'")))
                         for f in current_panel.get("accepted_text_facts", [])
                         if f["kind"] in {"dialogue", "thought"}}
        quotations = [next(part for part in quote.groups() if part is not None)
                      for quote in re.finditer(r'"([^"]+)"|“([^”]+)”|(?<!\w)\'(.*?)\'(?!\w)', text, re.DOTALL)]
        for name in all_allowed_names:
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", narrator_text, flags=re.IGNORECASE):
                if not any(name.casefold() == declared.casefold() for declared in declared_names):
                    errors.append(f"Segment {index + 1} does not ground the name {name!r}.")
            else:
                for quoted in quotations:
                    if (re.search(rf"(?<!\w){re.escape(name)}(?!\w)", quoted, re.IGNORECASE)
                            and tuple(_words(quoted.casefold().replace("’", "'"))) not in source_speech):
                        errors.append(f"Segment {index + 1} uses name {name!r} in a quotation not supported by exact source speech.")
        normalized.append({
            "primary_panel_id": primary,
            "supporting_panel_ids": supporting,
            "text": text,
            "used_instance_ids": used_instances,
            "name_uses": name_uses,
        })
    return normalized, errors


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _story_response_schema(panels: list[dict], previous_segments: list[dict] | None = None,
                           *, suppress_names: bool = False) -> dict:
    """Constrain syntax, output count, and exact ID vocabulary, not narration."""
    def references(values):
        values = sorted(set(values))
        return {
            "type": "array", "maxItems": len(values),
            # LMFE treats an empty enum as unrestricted, even for maxItems=0.
            "items": {"type": "string", "enum": values or [""]},
        }

    panels = _reference_panels(panels, previous_segments, suppress_names=suppress_names)
    variants = []
    for panel in panels:
        characters = panel["characters"]
        names = [
            {"type": "object", "properties": {
                "name": {"type": "string", "enum": [character["allowed_name"]]},
                "instance_id": {"type": "string", "enum": [character["instance_id"]]},
            }, "required": ["name", "instance_id"], "additionalProperties": False}
            for character in characters if character.get("allowed_name")
        ]
        segment = {
            "type": "object", "properties": {
                "primary_panel_id": {"type": "string", "enum": [panel["panel_id"]]},
                "supporting_panel_ids": {"type": "array", "minItems": 1, "maxItems": 1,
                                         "items": {"type": "string", "enum": [panel["panel_id"]]}},
                "text": {"type": "string", "minLength": 1, "maxLength": 800},
                "used_instance_ids": references(character["instance_id"] for character in characters),
                "name_uses": {"type": "array", "maxItems": len(names),
                              "items": {"anyOf": names} if names else {"type": "object"}},
            },
            "required": ["primary_panel_id", "supporting_panel_ids", "text", "used_instance_ids", "name_uses"],
            "additionalProperties": False,
        }
        variants.append(segment)
        if _static_visual_hold(panel):
            variants.pop()
        if _visual_hold_allowed(panel):
            variants.append({
                "type": "object", "properties": {
                    "primary_panel_id": {"type": "string", "enum": [panel["panel_id"]]},
                    "supporting_panel_ids": segment["properties"]["supporting_panel_ids"],
                    "text": {"type": "string", "enum": [""]},
                    "used_instance_ids": {"type": "array", "maxItems": 0, "items": {"type": "string"}},
                    "name_uses": {"type": "array", "maxItems": 0, "items": {"type": "object"}},
                    "narration_policy": {"type": "string", "enum": ["visual_hold_v1"]},
                }, "required": ["primary_panel_id", "supporting_panel_ids", "text", "used_instance_ids", "name_uses", "narration_policy"],
                "additionalProperties": False,
            })
    return {
        "type": "object", "properties": {
            "segments": {"type": "array", "items": {"anyOf": variants},
                         "minItems": len(panels), "maxItems": len(panels)},
        }, "required": ["segments"], "additionalProperties": False,
    }


def _output_token_budget(tokenizer, panels: list[dict]) -> int:
    """Tokenize a representative result and reserve prose plus JSON headroom."""
    representative = {"segments": []}
    for panel in panels:
        characters = panel.get("characters", [])
        used = [row["instance_id"] for row in characters]
        names = [
            {"name": row["allowed_name"], "instance_id": row["instance_id"]}
            for row in characters if row.get("allowed_name")
        ]
        representative["segments"].append({
            "primary_panel_id": panel["panel_id"],
            "supporting_panel_ids": [panel["panel_id"]],
            "text": " ".join(["narration"] * 45),
            "used_instance_ids": used,
            "name_uses": names,
        })
    measured = _token_count(
        tokenizer,
        json.dumps(representative, ensure_ascii=False, separators=(",", ":")),
    )
    return min(
        config.STORY_MAX_OUTPUT_TOKENS,
        max(256, measured + config.STORY_OUTPUT_HEADROOM_TOKENS),
    )


def _request_fits(input_tokens: int, output_tokens: int, *, single=False) -> bool:
    input_limit = (
        config.STORY_SINGLE_REQUEST_TOKENS if single
        else config.STORY_CHUNK_HARD_TOKENS
    )
    return (
        input_tokens <= input_limit
        and input_tokens + output_tokens <= config.STORY_CONTEXT_TOKEN_BUDGET
    )


def _previous_words(segments: list[dict], count: int) -> str:
    text = " ".join(segment["text"] for segment in segments).strip()
    matches = list(re.finditer(r"[\w]+(?:[’'-][\w]+)*", text, flags=re.UNICODE))
    if len(matches) <= count:
        return text
    return text[matches[-count].start():]


def _initial_chunks(prepared: dict, tokenizer, completed_panels: list[dict]) -> list[list[dict]]:
    remaining = prepared["panels"][len(completed_panels):]
    if not remaining:
        return []
    full_prompt = build_story_prompt(
        remaining,
        completed_panels=completed_panels, previous_narration="", tokenizer=tokenizer,
    )
    full_input = _token_count(tokenizer, full_prompt)
    if (not completed_panels and len(remaining) <= config.STORY_CHUNK_MAX_PANELS
            and len({(panel["chapter"], panel["page"]) for panel in remaining}) == 1
            and _request_fits(full_input, _output_token_budget(tokenizer, remaining), single=True)):
        return [remaining]

    chunks, current = [], []
    for panel in remaining:
        candidate = current + [panel]
        prompt = build_story_prompt(
            candidate,
            completed_panels=completed_panels,
            previous_narration="",
            tokenizer=tokenizer,
        )
        input_tokens = _token_count(tokenizer, prompt)
        output_tokens = _output_token_budget(tokenizer, candidate)
        if current and (
            len(candidate) > config.STORY_CHUNK_MAX_PANELS
            or (panel["chapter"], panel["page"]) != (current[0]["chapter"], current[0]["page"])
            or input_tokens > config.STORY_CHUNK_TARGET_TOKENS
            or not _request_fits(input_tokens, output_tokens)
        ):
            chunks.append(current)
            current = [panel]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _restricted_fallback(panel: dict) -> dict:
    """Build one-panel narration from an already accepted visual fact only."""
    reviewed = panel.get("reviewed_narration")
    text = re.sub(r"\s+", " ", reviewed or panel["description"]).strip()
    if not reviewed and text_issues(text):
        raise ValueError(
            f"Panel {panel['panel_id']} has no usable accepted fact for restricted fallback."
        )
    return {
        "primary_panel_id": panel["panel_id"],
        "supporting_panel_ids": [panel["panel_id"]],
        "text": text,
        "used_instance_ids": [],
        "name_uses": [],
        "fallback_policy": (
            "reviewed_narration_v1" if reviewed else "accepted_panel_fact_v1"
        ),
    }


def load_story_model():
    """Start one owned llama.cpp server; no PyTorch or vision model is loaded."""
    from story_writer.llama_cpp import LlamaCppWriter
    model = LlamaCppWriter()
    return model, model


def generate_response(processor, model, prompt: str, *, max_new_tokens: int,
                      response_schema: dict | None = None) -> tuple[str, dict]:
    """Generate with llama.cpp's native JSON grammar, not Python token callbacks."""
    return model.generate(prompt, max_new_tokens=max_new_tokens, response_schema=response_schema)


def _archive_stale_run(path: Path, manga: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = [
        path,
        path.parent / f"{manga}_story.txt",
        path.parent / f"{manga}_story.json",
        path.parent / f"{manga}_panel_timeline.json",
    ]
    for current in paths:
        if current.is_file():
            archive = current.with_name(
                f"{current.stem}.stale_{stamp}{current.suffix}"
            )
            current.replace(archive)


def _load_progress(path: Path, prepared: dict, *, restart: bool) -> dict:
    if restart:
        _archive_stale_run(path, prepared["manga"])
    if path.is_file():
        progress = _read_json(path)
        if progress.get("input_fingerprint") != prepared["input_fingerprint"]:
            _archive_stale_run(path, prepared["manga"])
        else:
            expected = [panel["panel_id"] for panel in prepared["panels"]]
            covered = progress.get("completed_panel_ids", [])
            segments = progress.get("segments", [])
            if (not isinstance(covered, list) or not isinstance(segments, list)
                    or covered != expected[:len(covered)]
                    or [row.get("primary_panel_id") for row in segments
                        if isinstance(row, dict)] != covered
                    or len(segments) != len(covered)):
                raise ValueError("Story progress panel order is invalid; use --restart.")
            _, errors = validate_story_response(
                json.dumps({"segments": segments}, ensure_ascii=False),
                prepared["panels"][:len(covered)],
                known_names=set(prepared.get("canonical_names", [])) | {
                    row["allowed_name"] for row in prepared["character_guide"]
                    if row.get("allowed_name")
                },
            ) if covered else ([], [])
            if errors:
                raise ValueError(
                    "Saved story progress failed validation; use --restart. "
                    + "; ".join(errors)
                )
            if progress.get("complete") and covered != expected:
                raise ValueError("Story progress is marked complete before all panels; use --restart.")
            return progress
    elif any((path.parent / filename).is_file() for filename in (
        f"{prepared['manga']}_story.txt",
        f"{prepared['manga']}_story.json",
        f"{prepared['manga']}_panel_timeline.json",
    )):
        _archive_stale_run(path, prepared["manga"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + prepared["input_fingerprint"][:12]
    return {
        "schema_version": 2,
        "manga": prepared["manga"],
        "input_fingerprint": prepared["input_fingerprint"],
        "identity_revision": prepared["identity_revision"],
        "prompt_version": PROMPT_VERSION,
        "run_id": run_id,
        "completed_panel_ids": [],
        "segments": [],
        "chunks": [],
        "complete": False,
    }


def _boundary_edit(segments: list[dict]) -> list[dict]:
    """Keep model sentences unchanged, including repeats across chunk boundaries."""
    return [dict(segment) for segment in segments]


def _timeline(segments: list[dict], panel_lookup: dict[str, dict], wpm: int) -> list[dict]:
    from story_writer.display import VISUAL_HOLD_SECONDS, is_visual_hold, narration_blocks
    timeline = []
    word_cursor = 0
    seconds_cursor = 0.0
    words_per_second = wpm / 60
    scene_durations = {}
    for block in narration_blocks(segments):
        if segments[block[-1]].get("scene_panel_ids") is not None:
            duration = len(_words(segments[block[-1]]["text"])) / words_per_second
            scene_durations.update({index: duration / len(block) for index in block})
    for index, segment in enumerate(segments):
        count = len(_words(segment["text"]))
        duration = VISUAL_HOLD_SECONDS if is_visual_hold(segment) else count / words_per_second if words_per_second else 0
        duration = scene_durations.get(index, duration)
        panel = panel_lookup[segment["primary_panel_id"]]
        timeline.append({
            "panel_id": segment["primary_panel_id"],
            "chapter": panel["chapter"],
            "panel_image": panel["panel_image"],
            "start_word": word_cursor,
            "end_word": word_cursor + count,
            "word_count": count,
            "estimated_start_seconds": round(seconds_cursor, 3),
            "estimated_end_seconds": round(seconds_cursor + duration, 3),
        })
        word_cursor += count
        seconds_cursor += duration
    return timeline


def _publish(prepared: dict, progress: dict, output_root: Path) -> dict:
    from story_writer.display import DISPLAY_POLICY, build_display_groups

    panel_lookup = {panel["panel_id"]: panel for panel in prepared["panels"]}
    segments = _boundary_edit(progress["segments"])
    published_segments = []
    for segment in segments:
        panel = panel_lookup[segment["primary_panel_id"]]
        row = dict(segment)
        row.update(
            chapter=panel["chapter"],
            page=panel["page"],
            panel_number=panel["panel_number"],
            panel_image=panel["panel_image"],
            word_count=len(_words(segment["text"])),
        )
        published_segments.append(row)
    narration = " ".join(segment["text"] for segment in published_segments if segment["text"]).strip()
    metadata = {
        "input_files": prepared["input_files"],
        "input_fingerprint": prepared["input_fingerprint"],
        "identity_revision": prepared["identity_revision"],
        "prompt_version": PROMPT_VERSION,
        "model_version": prepared["model_version"],
        "narration_words_per_minute": config.STORY_NARRATION_WPM,
        "target_seconds_per_panel": config.STORY_SECONDS_PER_PANEL,
        "generation_settings": prepared["settings"],
        "run_id": progress["run_id"],
        "boundary_edit_policy": "verbatim_model_text_v2",
        "narration_mode": prepared.get("narration_mode", "panel"),
        "resumed_from_prompt_version": progress.get("resumed_from_prompt_version"),
    }
    if "quality_checks" in progress:
        metadata["quality_checks"] = progress["quality_checks"]
    story = {
        "schema_version": STORY_SCHEMA_VERSION,
        "manga": prepared["manga"],
        "narration": narration,
        "segments": published_segments,
        "metadata": metadata,
    }
    timeline = {
        "schema_version": 1,
        "manga": prepared["manga"],
        "timeline": _timeline(published_segments, panel_lookup, config.STORY_NARRATION_WPM),
        "metadata": metadata,
    }
    timeline["display_groups"] = build_display_groups(published_segments, timeline["timeline"])
    timeline["display_policy"] = DISPLAY_POLICY
    text_path = output_root / f"{prepared['manga']}_story.txt"
    story_path = output_root / f"{prepared['manga']}_story.json"
    timeline_path = output_root / f"{prepared['manga']}_panel_timeline.json"
    _atomic_text(text_path, narration + "\n")
    write_json(story_path, story)
    write_json(timeline_path, timeline)
    progress.update(complete=True, output_files={
        "story_text": str(text_path.resolve()),
        "story_json": str(story_path.resolve()),
        "panel_timeline": str(timeline_path.resolve()),
    })
    return story


def generate_story(prepared: dict, processor, model, output_root: Path, *, restart=False) -> dict:
    """Generate or resume all chunks and atomically publish the final story."""
    progress_path = output_root / f"{prepared['manga']}_story_progress.json"
    progress = _load_progress(progress_path, prepared, restart=restart)
    if progress.get("complete"):
        story = _publish(prepared, progress, output_root)
        write_json(progress_path, progress)
        return story

    panel_lookup = {panel["panel_id"]: panel for panel in prepared["panels"]}
    completed_panels = [panel_lookup[panel_id] for panel_id in progress["completed_panel_ids"]]
    total_panels = len(prepared["panels"])
    show_progress("Story panels", len(completed_panels), total_panels)
    print(
        f"Story progress: {len(completed_panels)}/{total_panels} panels "
        f"({len(completed_panels) / total_panels:.0%})",
        flush=True,
    )
    chunks = _initial_chunks(prepared, processor.tokenizer, completed_panels)
    debug_dir = output_root / prepared["manga"] / "story_runs" / progress["run_id"]
    debug_dir.mkdir(parents=True, exist_ok=True)
    chunk_number = len(progress["chunks"])
    retry_feedback = {}
    panel_failures = {}
    name_suppressed_chunks = set()
    effect_cleaned_chunks = set()
    anonymous_repair_chunks = set()

    while chunks:
        chunk = chunks.pop(0)
        previous = _previous_words(progress["segments"], config.STORY_CONTEXT_WORDS)
        prompt = build_story_prompt(
            chunk,
            completed_panels=completed_panels,
            previous_narration=previous,
            tokenizer=processor.tokenizer,
            previous_segments=progress["segments"],
            canonical_names=prepared.get("canonical_names"),
        )
        input_tokens = _token_count(processor.tokenizer, prompt)
        whole_story_request = (
            not completed_panels
            and not progress["segments"]
            and len(chunk) == len(prepared["panels"])
            and _request_fits(
                input_tokens,
                _output_token_budget(processor.tokenizer, chunk),
                single=True,
            )
        )
        output_budget = _output_token_budget(processor.tokenizer, chunk)
        if len(chunk) > config.STORY_CHUNK_MAX_PANELS:
            chunks[0:0] = [chunk[:config.STORY_CHUNK_MAX_PANELS], chunk[config.STORY_CHUNK_MAX_PANELS:]]
            continue
        if (not whole_story_request and not _request_fits(input_tokens, output_budget)
                and len(chunk) > 1):
            midpoint = len(chunk) // 2
            chunks.insert(0, chunk[midpoint:])
            chunks.insert(0, chunk[:midpoint])
            continue
        if not whole_story_request and not _request_fits(input_tokens, output_budget):
            raise ValueError(
                "One panel exceeds the combined story input/output context budget."
            )

        chunk_number += 1
        max_new_tokens = output_budget
        accepted_segments = None
        attempt_records = []
        validation_errors = []
        response_schema = _story_response_schema(chunk, progress["segments"])
        raw = ""
        chunk_key = tuple(panel["panel_id"] for panel in chunk)
        previous_failure = retry_feedback.pop(chunk_key, None)
        for attempt in (1, 2):
            print(
                f"  Generating chunk {chunk_number}, attempt {attempt}/2: "
                f"{len(chunk)} panels; {len(completed_panels)}/{total_panels} complete",
                flush=True,
            )
            attempt_prompt = prompt
            failure_errors = validation_errors if attempt == 2 else (previous_failure or (None, []))[1]
            if any("name-frequency" in error or "declares a name that is not in its text" in error
                   for error in failure_errors):
                name_suppressed_chunks.add(chunk_key)
            if any("sound effect" in error and "reports source presentation" in error for error in failure_errors):
                effect_cleaned_chunks.add(chunk_key)
            if any("unknown dialogue explicitly anonymous" in error or "omits brief anonymous dialogue" in error
                   for error in failure_errors):
                anonymous_repair_chunks.add(chunk_key)
            if (chunk_key in name_suppressed_chunks or chunk_key in effect_cleaned_chunks
                    or chunk_key in anonymous_repair_chunks):
                # Do not reintroduce withheld names through the failed response or its schema.
                retry_chunk = chunk
                if chunk_key in effect_cleaned_chunks:
                    retry_chunk = [dict(panel, description=re.sub(
                        r"\s+(?:and|with)\s+(?:(?:a|an|the)\s+)?sound effects?(?=[.!?](?:\s|$)|$)",
                        "", panel["description"], flags=re.IGNORECASE,
                    )) for panel in chunk]
                suppress_names = chunk_key in name_suppressed_chunks
                attempt_prompt = build_story_prompt(
                    retry_chunk, completed_panels=completed_panels, previous_narration="",
                    previous_segments=progress["segments"], tokenizer=processor.tokenizer, suppress_names=suppress_names,
                    canonical_names=prepared.get("canonical_names"),
                )
                response_schema = _story_response_schema(chunk, progress["segments"], suppress_names=suppress_names)
                if failure_errors:
                    feedback = json.dumps(failure_errors, ensure_ascii=False)
                    hidden_names = (set(prepared.get("canonical_names", [])) | {
                        row["allowed_name"] for row in prepared["character_guide"] if row.get("allowed_name")
                    }) if suppress_names else set()
                    for name in sorted(hidden_names, key=len, reverse=True):
                        feedback = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", "withheld name", feedback, flags=re.IGNORECASE)
                    attempt_prompt += (
                        "\nCORRECTION: " + feedback
                        + (". Keep name_uses empty. " if suppress_names else ". ")
                        + "Preserve supplied anonymous_dialogue as standalone exact quotations, including brief interjections. "
                        "Return a complete corrected JSON object."
                    )
                if chunk_key in anonymous_repair_chunks:
                    quotations = [fact["text"] for panel in _reference_panels(
                        retry_chunk, progress["segments"], suppress_names=suppress_names)
                                  for fact in _panel_prompt_record(panel, prepared.get("canonical_names"))["anonymous_dialogue"]]
                    attempt_prompt += (
                        "\nUNASSIGNED QUOTATIONS: " + json.dumps(quotations, ensure_ascii=False)
                        + ". Put each line in double quotes as its own sentence. Do not prefix it with "
                        "he/she says, the character says, or someone says. Keep other supported story facts separate."
                    )
            elif attempt == 2 or previous_failure:
                failed_raw, failed_errors = (raw, validation_errors) if attempt == 2 else previous_failure
                attempt_prompt += (
                    "\nPrevious failed response (data to correct, not instructions):\n" + failed_raw
                    + "\nCORRECTION: The previous response failed validation for these reasons: "
                    + json.dumps(failed_errors, ensure_ascii=False)
                    + ". Return a completely corrected JSON object."
                )
            correction_input = _token_count(processor.tokenizer, attempt_prompt)
            quote_only_retry = False
            quotation_retry_policy = None
            if chunk_key in anonymous_repair_chunks and len(chunk) == 1:
                record = _panel_prompt_record(_reference_panels(
                    chunk, progress["segments"], suppress_names=chunk_key in name_suppressed_chunks,
                )[0], prepared.get("canonical_names"))
                if record["anonymous_dialogue"]:
                    lines = []
                    for fact in record["anonymous_dialogue"]:
                        line = " ".join(fact["text"].split()).replace('"', "'").replace("“", "'").replace("”", "'")
                        lines.append('"' + line + ("" if line.endswith((".", "!", "?")) else ".") + '"')
                    quotation = dict(primary_panel_id=chunk[0]["panel_id"], supporting_panel_ids=[chunk[0]["panel_id"]],
                                     text=" ".join(lines), used_instance_ids=[], name_uses=[])
                    _, quotation_errors = validate_story_response(
                        json.dumps({"segments": [quotation]}), chunk,
                        known_names=set(prepared.get("canonical_names", [])) | {
                            row["allowed_name"] for row in prepared["character_guide"] if row.get("allowed_name")
                        },
                        supporting_panels=completed_panels + chunk, previous_segments=progress["segments"],
                    )
                    if not quotation_errors:
                        properties = response_schema["properties"]["segments"]["items"]["anyOf"][0]["properties"]
                        quote_only_retry = not any(record[key] for key in (
                            "story_context", "accepted_text_facts", "accepted_actions", "accepted_transformations"))
                        if quote_only_retry:
                            properties["text"] = {"type": "string", "enum": [quotation["text"]]}
                            properties["used_instance_ids"]["maxItems"] = 0
                            properties["name_uses"]["maxItems"] = 0
                            quotation_retry_policy = "source_quotation_only_v1"
                        else:
                            # Native pattern decoding mishandles ASCII quotes inside JSON strings.
                            prefix = " ".join("“" + line[1:-1] + "”" for line in lines)
                            properties["text"] = {"type": "string", "maxLength": 800,
                                                  "pattern": "^" + re.escape(prefix).replace(r"\ ", " ") + r'( [A-Z][^"\\\n\r]*)?$'}
                            quotation_retry_policy = "source_quotation_prefix_v1"
                            attempt_prompt += "\nBegin with the unassigned quotations, then continue with the supported actions and known speech."
                            correction_input = _token_count(processor.tokenizer, attempt_prompt)
            if not _request_fits(correction_input, max_new_tokens, single=whole_story_request):
                validation_errors = ["Correction prompt exceeds the combined context budget."]
                break
            try:
                raw, generation = generate_response(
                    processor, model, attempt_prompt, max_new_tokens=max_new_tokens,
                    response_schema=response_schema,
                )
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                attempt_records.append({
                    "attempt": attempt,
                    "generation": {"error": "out_of_memory"},
                    "validation_errors": ["Generation exceeded available memory."],
                })
                validation_errors = ["Generation exceeded available memory."]
                break
            raw_path = debug_dir / f"chunk_{chunk_number:04d}_attempt_{attempt}.txt"
            _atomic_text(raw_path, raw)
            segments, validation_errors = validate_story_response(
                raw,
                chunk,
                known_names=set(prepared.get("canonical_names", [])) | {
                    row["allowed_name"] for row in prepared["character_guide"]
                    if row.get("allowed_name")
                },
                supporting_panels=completed_panels + chunk,
                previous_segments=progress["segments"],
            )
            if generation.get("truncated"):
                validation_errors.append("Generation reached its token limit.")
            attempt_records.append({
                "attempt": attempt,
                "raw_response": str(raw_path.resolve()),
                "repair_policy": quotation_retry_policy,
                "generation": generation,
                "validation_errors": list(validation_errors),
            })
            print(
                f"    {generation.get('generated_tokens', 0)}/{max_new_tokens} tokens; "
                f"{generation.get('elapsed_seconds', 0):.1f}s; "
                f"{generation.get('tokens_per_second', 0):.2f} tok/s; "
                + ("valid" if not validation_errors else "; ".join(validation_errors[:2])),
                flush=True,
            )
            if not validation_errors:
                accepted_segments = segments
                break
            if generation.get("truncated"):
                larger_budget = min(config.STORY_MAX_OUTPUT_TOKENS, max_new_tokens * 2)
                if larger_budget == max_new_tokens:
                    break
                max_new_tokens = larger_budget
        if accepted_segments is None:
            progress["chunks"].append({
                "chunk": chunk_number,
                "panel_ids": [panel["panel_id"] for panel in chunk],
                "attempts": attempt_records,
                "status": "failed",
                "reason": list(validation_errors),
            })
            write_json(progress_path, progress)
            if len(chunk) > 1:
                midpoint = len(chunk) // 2
                chunks[0:0] = [chunk[:midpoint], chunk[midpoint:]]
                print(f"    Chunk {chunk_number} failed; progress saved. Retrying as smaller chunks.", flush=True)
            else:
                panel_id = chunk[0]["panel_id"]
                panel_failures[panel_id] = panel_failures.get(panel_id, 0) + 1
                if panel_failures[panel_id] >= config.STORY_MAX_PANEL_FAILURES:
                    raise RuntimeError(
                        f"Panel {panel_id} failed {panel_failures[panel_id]} retry cycles. "
                        f"Progress and failed responses retained: {progress_path}. "
                        f"Last errors: {'; '.join(validation_errors[:2])}. "
                        "Resolve the reported validation errors before resuming."
                    )
                if "Correction prompt exceeds the combined context budget." not in validation_errors:
                    retry_feedback[(chunk[0]["panel_id"],)] = (raw, validation_errors)
                chunks.insert(0, chunk)
                print(f"    Panel still failed validation; progress saved. Retrying in 5 seconds (Ctrl+C to stop).", flush=True)
                time.sleep(5)
            continue

        for segment in accepted_segments:
            segment["generation_chunk"] = chunk_number
        progress["segments"].extend(accepted_segments)
        panel_ids = [panel["panel_id"] for panel in chunk]
        progress["completed_panel_ids"].extend(panel_ids)
        if not progress["chunks"] or progress["chunks"][-1].get("chunk") != chunk_number:
            progress["chunks"].append({
                "chunk": chunk_number,
                "panel_ids": panel_ids,
                "attempts": attempt_records,
                "status": "complete",
            })
        completed_panels.extend(chunk)
        write_json(progress_path, progress)
        show_progress("Story panels", len(completed_panels), total_panels)
        print(
            f"    Completed {len(completed_panels)}/{total_panels} panels "
            f"({len(completed_panels) / total_panels:.0%})",
            flush=True,
        )

    if progress["completed_panel_ids"] != [panel["panel_id"] for panel in prepared["panels"]]:
        raise ValueError("Story generation ended without covering every panel.")
    story = _publish(prepared, progress, output_root)
    write_json(progress_path, progress)
    return story


def run_story_writer(
    manga_dir: Path,
    chapters: list[Path],
    output_root: Path,
    *,
    restart: bool = False,
    narration_style: str = "panel",
) -> dict:
    if narration_style not in {"panel", "scene"}:
        raise ValueError("narration_style must be panel or scene.")
    print("Preparing grounded story evidence...", flush=True)
    prepared = prepare_story(manga_dir, chapters, output_root)
    progress_path = output_root / f"{prepared['manga']}_story_progress.json"
    if narration_style == "panel" and progress_path.is_file() and not restart:
        progress = _load_progress(progress_path, prepared, restart=False)
        if progress.get("complete"):
            print("Story is already complete; restoring published files.", flush=True)
            story = _publish(prepared, progress, output_root)
            write_json(progress_path, progress)
            return story
    print(f"Loading story model: {Path(config.STORY_MODEL_PATH).name} (llama.cpp)...", flush=True)
    processor, model = load_story_model()
    print("Story model ready.", flush=True)
    try:
        if narration_style == "scene":
            from story_writer.scenes import generate_scene_story
            return generate_scene_story(prepared, processor, model, output_root, restart=restart)
        return generate_story(prepared, processor, model, output_root, restart=restart)
    finally:
        model.close()
