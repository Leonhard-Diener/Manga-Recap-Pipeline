"""Page-level grounded prose, with complete panel coverage and resumable attempts."""
from itertools import groupby
import json
import re
import time

from common.json_io import write_json
from common.progress import show_progress
from story_writer import writer
from story_writer import evidence

SCENE_PROMPT_VERSION = "page_story_prose_v7_bound_sentences"


def scene_input(panels, previous_segments=()):
    recent = {use["name"] for row in list(previous_segments)[-2:] for use in row.get("name_uses", [])}
    # Reference permissions may change; accepted source text must not be rewritten.
    panels = [dict(p, characters=[dict(c, allowed_name=None) if c.get("allowed_name") in recent else dict(c)
                                  for c in p["characters"]]) for p in panels]
    combined = dict(panels[-1], description="Scene facts below.",
                    characters=[c for p in panels for c in p["characters"]],
                    accepted_actions=[a for p in panels for a in p["accepted_actions"]],
                    accepted_transformations=[a for p in panels for a in p["accepted_transformations"]],
                    accepted_text_facts=[a for p in panels for a in p.get("accepted_text_facts", [])])
    combined["accepted_text_facts"].extend({"kind": "narration", "text": p["reviewed_narration"]}
                                            for p in panels if p.get("reviewed_narration"))
    record = writer._panel_prompt_record(combined)
    has_story_facts = any(record[key] for key in (
        "story_context", "accepted_actions", "accepted_transformations", "accepted_text_facts", "anonymous_dialogue"))
    facts = []
    relevant_ids = set()
    for panel in panels:
        row = writer._panel_prompt_record(panel)
        fact = {key: row[key] for key in ("story_context", "accepted_actions", "accepted_transformations",
                                        "accepted_text_facts", "anonymous_dialogue")}
        if panel.get("reviewed_narration"):
            fact["story_context"] = fact["story_context"] + [panel["reviewed_narration"]]
        # Poses and gaze stay visible in the video, not in a roll call of every occurrence.
        fact["accepted_actions"] = [action for action in fact["accepted_actions"] if not _routine_action(action)]
        if not has_story_facts and not writer._static_visual_hold(panel):
            fact["visible_observation"] = row["description"]
        facts.append(fact)
        relevant_ids.update(f.get("speaker_instance_id") for f in fact["accepted_text_facts"] if f.get("speaker_instance_id"))
        for event in fact["accepted_actions"] + fact["accepted_transformations"]:
            relevant_ids.update(event.get(key) for key in ("actor_instance_id", "target_instance_id", "source_instance_id") if event.get(key))
    # Only actual speech/event endpoints need reference permissions, not every visible body.
    combined["characters"] = [c for c in combined["characters"] if c["instance_id"] in relevant_ids]
    record = writer._panel_prompt_record(combined)
    compact = {"primary_panel_id": combined["panel_id"], "characters": [
        dict({key: c.get(key) for key in ("instance_id", "entity_id", "allowed_name", "allowed_reference")},
             allowed_reference=writer._brief_character_reference(original, combined),
             pronoun=writer._brief_character_reference(original, combined))
        for c, original in zip(record["characters"], combined["characters"])], "ordered_facts": facts}
    return combined, compact


def _routine_action(action):
    return evidence.incidental_action(action)


def expand_scenes(groups, scenes):
    segments = []
    for panels, scene in zip(groups, scenes):
        ids = [p["panel_id"] for p in panels]
        if scene["segment"].get("narration_policy") == "visual_hold_v1":
            segments.extend({"primary_panel_id": panel_id, "supporting_panel_ids": [panel_id],
                             "text": "", "used_instance_ids": [], "name_uses": [],
                             "narration_policy": "visual_hold_v1"} for panel_id in ids)
            continue
        segments.extend({"primary_panel_id": panel_id, "supporting_panel_ids": [panel_id], "text": "",
                         "used_instance_ids": [], "name_uses": [], "narration_policy": "scene_visual_v1",
                         "narrated_by_panel_id": ids[-1]} for panel_id in ids[:-1])
        segments.append(dict(scene["segment"], supporting_panel_ids=ids, scene_panel_ids=ids))
    return segments


def generate_scene_story(prepared, processor, model, output_root, *, restart=False):
    prepared = dict(prepared, narration_mode=SCENE_PROMPT_VERSION,
                    settings=dict(prepared["settings"], scene_prompt_version=SCENE_PROMPT_VERSION))
    prepared["input_fingerprint"] = writer._stable_hash({
        "input": prepared["input_fingerprint"], "scene_prompt_version": SCENE_PROMPT_VERSION,
        "evidence_version": evidence.EVIDENCE_VERSION, "review_version": evidence.REVIEW_VERSION})
    path = output_root / f"{prepared['manga']}_story_progress.json"
    groups = [list(rows) for _, rows in groupby(prepared["panels"], key=lambda p: (p["chapter"], p["page"]))]
    if path.exists() and (restart or writer._read_json(path).get("input_fingerprint") != prepared["input_fingerprint"]):
        writer._archive_stale_run(path, prepared["manga"])
    elif not path.exists() and any((output_root / f"{prepared['manga']}_{suffix}").exists()
                                   for suffix in ("story.txt", "story.json", "panel_timeline.json")):
        writer._archive_stale_run(path, prepared["manga"])
    state = writer._read_json(path) if path.exists() else {
        "input_fingerprint": prepared["input_fingerprint"], "run_id": str(time.time_ns()),
        "scene_prompt_version": SCENE_PROMPT_VERSION, "scenes": [], "segments": [], "complete": False}
    if state.get("scene_prompt_version") != SCENE_PROMPT_VERSION or len(state["scenes"]) > len(groups):
        raise ValueError("Incompatible scene checkpoint; use --restart.")
    known_names = set(prepared.get("canonical_names", [])) | {
        c["allowed_name"] for p in prepared["panels"] for c in p["characters"] if c.get("allowed_name")}
    previous, ledgers = [], []
    for group, saved in zip(groups, state["scenes"]):
        combined, ledger = _scene_evidence(group, prepared["panels"], previous)
        ledgers.append(ledger)
        errors = evidence.validate_draft(saved.get("draft"), ledger)
        if not errors:
            errors += evidence.validate_review(saved.get("review"), saved["draft"], ledger)
        if not errors:
            segment, errors = _compile_scene(saved["draft"], ledger, combined, known_names, previous)
            if segment != saved.get("segment"):
                errors.append("Saved text/reference bindings differ from the reviewed draft.")
        if (saved.get("panel_ids") != ledger["panel_ids"] or saved.get("ledger_fingerprint") != ledger["fingerprint"]
                or saved.get("review_version") != evidence.REVIEW_VERSION or errors):
            raise ValueError(f"Saved scene evidence is invalid; use --restart: {errors}")
        previous.append(saved["segment"])
    debug = output_root / prepared["manga"] / "story_runs" / state["run_id"]
    debug.mkdir(parents=True, exist_ok=True)
    write_json(path, state)
    for group in groups[len(previous):]:
        combined, ledger = _scene_evidence(group, prepared["panels"], previous)
        ledgers.append(ledger)
        write_json(debug / f"scene_{len(previous) + 1}.evidence.json", ledger)
        schema = evidence.draft_schema(ledger)
        errors = []
        candidate = None
        for attempt in range(1, 5):
            prompt = evidence.DRAFT_PROMPT + "\nEvidence:\n" + json.dumps(evidence.prompt_record(ledger, drafting=True), ensure_ascii=False)
            if errors:
                prompt += "\nRevise this SAME scene's candidate using only the evidence. Preserve supported meaning; fix every error."
                prompt += "\nUntrusted candidate:\n" + json.dumps(candidate, ensure_ascii=False)
                prompt += "\nErrors:\n" + "; ".join(errors)
            target = debug / f"scene_{len(previous) + 1}_{attempt}_{time.time_ns()}"
            print(f"Scene {len(previous) + 1}/{len(groups)}, attempt {attempt}/4", flush=True)
            draft, errors = _request(processor, model, prompt, schema, target,
                                     max(1000, len(ledger["facts"]) * 180))
            candidate = draft
            if not errors:
                errors = evidence.validate_draft(draft, ledger)
            if not errors:
                segment, errors = _compile_scene(draft, ledger, combined, known_names, previous)
            review = None
            if not errors:
                review_prompt = evidence.REVIEW_PROMPT + "\nEvidence:\n" + json.dumps(evidence.prompt_record(ledger), ensure_ascii=False)
                review_prompt += "\nDraft:\n" + json.dumps(draft, ensure_ascii=False)
                review, errors = _request(processor, model, review_prompt, evidence.review_schema(ledger, draft),
                                          target.with_name(target.name + "_review"),
                                          max(1000, (len(ledger["facts"]) + len(draft["sentences"])) * 110))
                if not errors:
                    errors = evidence.validate_review(review, draft, ledger)
            write_json(target.with_suffix(".validation.json"), {"errors": errors, "review_version": evidence.REVIEW_VERSION})
            if not errors:
                break
            print("Retry: " + "; ".join(errors), flush=True)
        if errors:
            raise RuntimeError(f"Scene failed after four attempts; checkpoint retained at {path}: {errors}")
        previous.append(segment)
        state["scenes"].append({"panel_ids": ledger["panel_ids"], "segment": segment,
                                "ledger_fingerprint": ledger["fingerprint"], "draft": draft,
                                "review": review, "review_version": evidence.REVIEW_VERSION})
        write_json(path, state)
        show_progress("Story panels", sum(len(s["panel_ids"]) for s in state["scenes"]), len(prepared["panels"]))
    state["segments"] = expand_scenes(groups, state["scenes"])
    if [s["primary_panel_id"] for s in state["segments"]] != [p["panel_id"] for p in prepared["panels"]]:
        raise ValueError("Scene expansion lost ordered panel coverage.")
    state["quality_checks"] = {
        "policy": evidence.REVIEW_VERSION, "evidence_version": evidence.EVIDENCE_VERSION,
        "automated_checks_passed": True, "scene_count": len(groups),
        "production_approved": False,
        "limitation": "Coverage and same-model semantic review are not independent factual/prose approval.",
        "audit_directory": str(debug.resolve()),
    }
    state["quality_checks"].update(
        required_fact_count=sum(f["required"] for ledger in ledgers for f in ledger["facts"]),
        excluded_fact_count=sum(not f["usable"] for ledger in ledgers for f in ledger["facts"]),
        unresolved_thought_count=sum(f["omission_reason"] == "unattributed_thought"
                                     for ledger in ledgers for f in ledger["facts"]),
    )
    write_json(debug / "quality_report.json", state["quality_checks"])
    story = writer._publish(prepared, state, output_root)
    write_json(path, state)
    return story


def _scene_evidence(group, all_panels, previous):
    combined, _ = scene_input(group, previous)
    ledger = evidence.build_ledger(group, all_panels)
    recent = {use["name"] for row in previous[-2:] for use in row.get("name_uses", [])}
    for character in ledger["characters"]:
        if character["allowed_name"] in recent:
            character["allowed_name"] = None
    current = {c["instance_id"] for c in combined["characters"]}
    lookup = {c["instance_id"]: c for p in all_panels for c in p["characters"]}
    for character in ledger["characters"]:
        if character["instance_id"] not in current:
            combined["characters"].append(dict(lookup[character["instance_id"]], allowed_name=character["allowed_name"]))
    ledger["fingerprint"] = writer._stable_hash({key: value for key, value in ledger.items() if key != "fingerprint"})
    return combined, ledger


def _compile_scene(draft, ledger, combined, known_names, previous):
    """Derive public IDs from validated evidence keys, never from model-supplied IDs."""
    characters = {c["key"]: c for c in ledger["characters"]}
    used, names = [], []
    for sentence in draft["sentences"]:
        narrator_text, _ = evidence.outside_quotations(sentence["text"])
        for key in sentence["characters"]:
            character = characters[key]
            if character["instance_id"] not in used:
                used.append(character["instance_id"])
            name = character["allowed_name"]
            if name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", narrator_text, re.IGNORECASE):
                use = {"name": name, "instance_id": character["instance_id"]}
                if use not in names:
                    names.append(use)
    segment = {"primary_panel_id": combined["panel_id"], "supporting_panel_ids": [combined["panel_id"]],
               "text": " ".join(s["text"].strip() for s in draft["sentences"]),
               "used_instance_ids": used, "name_uses": names}
    if not draft["sentences"]:
        segment["narration_policy"] = "visual_hold_v1"
        # Only exact incidental poses may disappear after an exhaustive omission review.
        combined = dict(combined, accepted_actions=[a for a in combined["accepted_actions"] if not _routine_action(a)])
    rows, errors = writer.validate_story_response(json.dumps({"segments": [segment]}), [combined],
                                                  known_names=known_names, previous_segments=previous)
    return (rows[0] if rows else None), errors


def _request(processor, model, prompt, schema, target, output_tokens):
    target.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
    write_json(target.with_suffix(".schema.json"), schema)
    output_tokens = min(output_tokens, writer.config.STORY_MAX_OUTPUT_TOKENS)
    input_tokens = writer._token_count(processor.tokenizer, prompt)
    if not writer._request_fits(input_tokens, output_tokens, single=True):
        write_json(target.with_suffix(".metrics.json"), {"input_tokens": input_tokens, "error": "context_budget"})
        raise ValueError("Page scene exceeds the story context budget; no evidence was dropped.")
    try:
        raw, metrics = writer.generate_response(processor, model, prompt, max_new_tokens=output_tokens, response_schema=schema)
    except Exception as error:
        write_json(target.with_suffix(".metrics.json"), {"error": str(error), "type": type(error).__name__})
        raise
    target.with_suffix(".raw.txt").write_text(raw, encoding="utf-8")
    errors = ["Response truncated."] if metrics.get("truncated") else []
    try:
        payload = writer._extract_json(raw)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        payload = None
        errors.append(f"Invalid JSON: {error}")
    write_json(target.with_suffix(".metrics.json"), dict(metrics, validation_errors=errors))
    return payload, errors
