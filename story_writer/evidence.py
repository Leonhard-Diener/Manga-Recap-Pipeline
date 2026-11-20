"""Immutable scene evidence and mechanical checks for evidence-bound narration.

Coverage declarations are not semantic proof. A separate source-based review must
still check that each sentence actually says what its citations support.
"""
import hashlib
import json
import re


EVIDENCE_VERSION = "scene_fact_ledger_v5_minimal"
REVIEW_VERSION = "source_semantic_review_v3_minimal"

DRAFT_PROMPT = """Tell the story aloud in third person: what changes, who responds, and why it matters if known.
Keep every required fact in source order. Combine related facts. Skip optional poses unless needed to follow the action.
Retell first-person background without identifying an unknown narrator. Keep beliefs, conditions and threats as such.
Use only grounded pronouns or permitted names. A nearby body or label does not identify a speaker or listener.
Keep unknown dialogue as standalone exact quotes; paraphrase known speech with the supplied speaker.
Return sentences with supporting fact keys, occurrence keys and speaker (null if unknown or narration). Keys never appear in spoken text.
"""

REVIEW_PROMPT = """Compare meanings. Evidence is the source; the draft and its citations are not evidence.
For each fact: is its meaning present? Accept faithful paraphrases and third-person retellings. Optional details may be omitted.
For each sentence: does the source support who did or said what, with the same certainty? Never demand a speaker, recipient or cause the source leaves unknown.
Return checks in order with the draft's 1-based citation indices. Give a concrete discrepancy for a rejection. List paragraph issues only for actual unclear or broken narration.
"""


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def incidental_pose(text):
    """Full matches only: a pose prefix must not erase a meaningful quiet action."""
    return bool(re.fullmatch(
        r"(?:(?:stands?|standing|sits?|sitting|kneeling|crouching)(?: (?:still|upright|nearby|in profile|facing (?:forward|left|right)))?"
        r"|(?:looking|staring|glancing) (?:forward|left|right|ahead|down|up|to the (?:left|right))"
        r"|touching (?:his|her|their) hair)\.?", text.strip(), flags=re.IGNORECASE))


def incidental_action(action):
    text = action["text"].strip()
    # Bare co-location is optional even with a target; blocking or approaching is an event.
    adjacent = re.fullmatch(r"(?:stands?|standing|sits?|sitting) (?:beside|next to) "
                            r"(?:him|her|them|C\d+|(?:the|a) (?:protagonist|boy|girl|man|woman))\.?",
                            text, re.IGNORECASE)
    return bool(adjacent) or (not action.get("target_instance_id") and incidental_pose(text))


def build_ledger(panels, all_panels=None):
    if not panels:
        raise ValueError("A scene requires source panels.")
    source_panels = all_panels if all_panels is not None else panels
    positions = {p["panel_id"]: index for index, p in enumerate(source_panels)}
    occurrences = {}
    for panel in source_panels:
        for character in panel["characters"]:
            identity = character["instance_id"]
            if identity in occurrences:
                raise ValueError(f"Duplicate occurrence in writer evidence: {identity}")
            occurrences[identity] = (panel["panel_id"], character)
    facts, references = [], {}

    def reference(instance_id, panel_id, *, external=False):
        if instance_id is None:
            return None
        if instance_id not in occurrences:
            raise ValueError(f"Missing grounded occurrence {instance_id} in {panel_id}.")
        occurrence_panel, character = occurrences[instance_id]
        if (not external and occurrence_panel != panel_id) or positions[occurrence_panel] > positions[panel_id]:
            raise ValueError("A fact has a stale or future occurrence endpoint.")
        if instance_id not in references:
            from story_writer.writer import _brief_character_reference
            appearance = character.get("visual_description", "")
            female = bool(re.search(r"\b(?:female|girl|woman|she|her)\b", appearance, re.IGNORECASE))
            male = bool(re.search(r"\b(?:male|boy|man|he|him|his)\b", appearance, re.IGNORECASE))
            references[instance_id] = {
                "key": f"C{len(references) + 1}", "instance_id": instance_id, "panel_id": occurrence_panel,
                "allowed_name": character.get("allowed_name"), "entity_id": character.get("entity_id"),
                "pronoun": "she" if female and not male else "he" if male and not female else None,
                "reference": _brief_character_reference(character, source_panels[positions[occurrence_panel]]),
            }
        return references[instance_id]["key"]

    def add(panel_id, kind, text, source_id=None, *, required=True, usable=True,
            reason=None, speaker=None, actor=None, target=None, source=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty {kind} evidence in {panel_id}.")
        body = dict(panel_id=panel_id, kind=kind, text=text, source_id=source_id,
                    speaker=speaker, actor=actor, target=target, source=source,
                    required=required, usable=usable, omission_reason=reason)
        instance_ids = {c["key"]: c["instance_id"] for c in references.values()}
        stable_body = dict(body, **{role: instance_ids.get(body[role]) for role in ("speaker", "actor", "target", "source")})
        stable = "evidence_" + _hash(stable_body)[:24]
        if any(f["id"] == stable for f in facts):
            return
        facts.append(dict(key=f"F{len(facts) + 1}", id=stable, **body))

    for panel in panels:
        panel_id = panel["panel_id"]
        for fact in panel.get("accepted_text_facts", []):
            kind, text = fact["kind"], fact["text"]
            unusable = kind in {"publication", "sound_effect"} or (kind == "thought" and not fact.get("speaker_instance_id"))
            speaker = reference(fact.get("speaker_instance_id"), panel_id) if kind in {"dialogue", "thought"} else None
            reason = ("unattributed_thought" if kind == "thought" else "non_story_text") if unusable else None
            add(panel_id, kind, text, fact.get("fact_id") or fact.get("proposal_id"),
                speaker=speaker, required=not unusable, usable=not unusable, reason=reason)
        if panel.get("reviewed_narration"):
            add(panel_id, "reviewed_narration", panel["reviewed_narration"], "reviewed_narration")
        for event in panel.get("accepted_actions", []):
            if not event.get("actor_instance_id"):
                raise ValueError("An accepted action has no actor occurrence.")
            optional = incidental_action(event)
            add(panel_id, "action", event["text"], event.get("event_id"),
                actor=reference(event["actor_instance_id"], panel_id),
                target=reference(event.get("target_instance_id"), panel_id),
                required=not optional, reason="incidental_pose" if optional else None)
        for event in panel.get("accepted_transformations", []):
            if not event.get("source_instance_id") or not event.get("target_instance_id"):
                raise ValueError("An accepted transformation has no exact endpoints.")
            source = reference(event["source_instance_id"], panel_id, external=True)
            target = reference(event["target_instance_id"], panel_id)
            source_panel = occurrences[event["source_instance_id"]][0]
            if source_panel != event.get("source_panel_id") or panel_id != event.get("target_panel_id"):
                raise ValueError("Transformation endpoint panels do not match current occurrences.")
            add(panel_id, "transformation", event.get("evidence") or event.get("text"), event.get("event_id"),
                source=source, target=target)

    if not any(f["required"] for f in facts):
        from story_writer.writer import _static_visual_hold
        for panel in panels:
            if panel.get("evidence_density") == "uncertain":
                add(panel["panel_id"], "uncertain_observation", panel["description"],
                    required=False, usable=False, reason="source_review_required")
            elif not _static_visual_hold(panel):
                add(panel["panel_id"], "observation", panel["description"], required=False, reason="visual_only")
    ledger = {"version": EVIDENCE_VERSION, "panel_ids": [p["panel_id"] for p in panels],
              "facts": facts, "characters": list(references.values())}
    return dict(ledger, fingerprint=_hash(ledger))


def draft_schema(ledger):
    usable = [f["key"] for f in ledger["facts"] if f["usable"]]
    groups = {}
    for fact in ledger["facts"]:
        if not fact["usable"]:
            continue
        bound = tuple(sorted({fact[role] for role in ("speaker", "actor", "target", "source") if fact[role]}))
        signature = (bound, fact["speaker"], fact["kind"] == "dialogue" and fact["speaker"] is None)
        groups.setdefault(signature, []).append(fact["key"])
    variants = []
    for (characters, speaker, _), fact_keys in groups.items():
        variants.append({"type": "object", "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 800},
            "facts": {"type": "array", "minItems": 1, "maxItems": len(fact_keys),
                      "items": {"type": "string", "enum": fact_keys}},
            "characters": {"type": "array", "minItems": len(characters), "maxItems": len(characters),
                           "items": {"type": "string", "enum": list(characters) or [""]}},
            "speaker": {"type": ["string", "null"], "enum": [speaker]},
        }, "required": ["text", "facts", "characters", "speaker"], "additionalProperties": False})
    return {"type": "object", "properties": {
        "sentences": {"type": "array", "minItems": int(any(f["required"] for f in ledger["facts"])), "maxItems": len(usable),
                      "items": {"anyOf": variants} if variants else {"type": "object"}},
    }, "required": ["sentences"], "additionalProperties": False}


def prompt_record(ledger, *, drafting=False):
    """Keep audit IDs off the model input; opaque IDs are not name permissions."""
    from story_writer.writer import _stable_hash
    positions = {panel_id: i + 1 for i, panel_id in enumerate(ledger["panel_ids"])}
    # Non-story text belongs in the audit/review, not in the narrator's reading material.
    facts = [f for f in ledger["facts"] if not drafting or f["usable"]]
    return {
        "facts": [dict({key: value for key, value in fact.items()
                        if key not in {"id", "source_id", "panel_id", "omission_reason"} and value is not None},
                       panel=positions[fact["panel_id"]]) for fact in facts],
        "characters": [{key: (_stable_hash(value) if key == "entity_id" and value else value)
                        for key, value in character.items() if key not in {"instance_id", "panel_id"}}
                       for character in ledger["characters"]],
    }


def validate_draft(payload, ledger):
    """Check evidence bindings and coverage, not whether paraphrases are entailed."""
    if not isinstance(payload, dict) or set(payload) != {"sentences"} or not isinstance(payload["sentences"], list):
        return ["Return exactly a sentences list."]
    facts = {f["key"]: f for f in ledger["facts"]}
    characters = {c["key"]: c for c in ledger["characters"]}
    errors, covered = [], set()
    positions = {panel_id: index for index, panel_id in enumerate(ledger["panel_ids"])}
    previous_position = -1
    if len(payload["sentences"]) > max(1, sum(f["usable"] for f in facts.values())):
        errors.append("Too many sentences for this scene's evidence budget.")
    for number, sentence in enumerate(payload["sentences"], 1):
        if not isinstance(sentence, dict) or set(sentence) != {"text", "facts", "characters", "speaker"}:
            errors.append(f"Sentence {number} has an invalid shape.")
            continue
        text, cited, used, speaker = (sentence[k] for k in ("text", "facts", "characters", "speaker"))
        if not isinstance(text, str) or not text.strip() or len(text) > 800:
            errors.append(f"Sentence {number} has invalid text.")
            continue
        if (not isinstance(cited, list) or not cited or any(not isinstance(k, str) for k in cited)
                or len(set(cited)) != len(cited) or any(k not in facts or not facts[k]["usable"] for k in cited)):
            errors.append(f"Sentence {number} cites invalid or unusable evidence.")
            continue
        position = min(positions[facts[key]["panel_id"]] for key in cited)
        if position < previous_position:
            errors.append(f"Sentence {number} moves backward in source panel order.")
        previous_position = position
        if (not isinstance(used, list) or any(not isinstance(k, str) for k in used)
                or len(set(used)) != len(used) or any(k not in characters for k in used)):
            errors.append(f"Sentence {number} has invalid occurrence references.")
            continue
        if speaker is not None and (not isinstance(speaker, str) or speaker not in used):
            errors.append(f"Sentence {number} has an invalid speaker binding.")
            continue
        bound = {facts[k][role] for k in cited for role in ("speaker", "actor", "target", "source") if facts[k][role]}
        if not set(used) <= bound:
            errors.append(f"Sentence {number} invents a visible participant for its cited facts.")
        if not bound <= set(used):
            errors.append(f"Sentence {number} omits grounded occurrence bindings from its cited facts.")
        speech = [facts[k] for k in cited if facts[k]["kind"] in {"dialogue", "thought"}]
        known_speakers = {f["speaker"] for f in speech if f["speaker"]}
        unknown_speech = any(f["kind"] == "dialogue" and f["speaker"] is None for f in speech)
        if known_speakers and (len(known_speakers) != 1 or speaker not in known_speakers or speaker not in used):
            errors.append(f"Sentence {number} must preserve its exact known speaker; split different speakers into separate sentences.")
        if unknown_speech and (speaker is not None or used):
            errors.append(f"Sentence {number} assigns unknown speech to a visible occurrence; keep it separate.")
        if speaker is not None and not speech:
            errors.append(f"Sentence {number} assigns speech to evidence that is not speech or thought.")
        outside, quote_errors = outside_quotations(text)
        errors.extend(f"Sentence {number}: {error}" for error in quote_errors)
        for token in re.findall(r"\b[CF]\d+\b", outside):
            if (token in facts or token in characters) and not any(
                    re.search(rf"\b{re.escape(token)}\b", facts[key]["text"]) for key in cited):
                errors.append(f"Sentence {number} prints metadata key {token} in narration.")
        if re.search(r"\b(?:I|me|my|mine|myself|we|us|our|ours|ourselves)\b", outside, re.IGNORECASE):
            errors.append(f"Sentence {number} uses first-person narrator wording outside quotations.")
        if unknown_speech and re.search(
                r"\b(?:he|she|they|the (?:boy|girl|man|woman|character))\s+(?:\w+ly\s+)?"
                r"(?:says?|asks?|replies|shouts?|adds?|explains?|declares?|thinks?)\b", outside, re.IGNORECASE):
            errors.append(f"Sentence {number} attributes unknown speech in its actual wording.")
        if not quote_errors and text.rstrip().rstrip('"”’\'').rstrip()[-1:] not in {".", "!", "?"}:
            errors.append(f"Sentence {number} is unfinished; end it as a complete sentence or quotation.")
        covered.update(cited)
    missing = [f["key"] for f in ledger["facts"] if f["required"] and f["key"] not in covered]
    if missing:
        errors.append("Missing required facts: " + ", ".join(missing))
    unresolved = [f["key"] for f in ledger["facts"] if f["omission_reason"] == "source_review_required"]
    if unresolved:
        errors.append("Source review required; do not invent missing visual facts: " + ", ".join(unresolved))
    return errors


def outside_quotations(text):
    """Mask quoted spans while retaining offsets; apostrophes in words are not quotes."""
    outside, closing = [], None
    for index, char in enumerate(text):
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        word_apostrophe = (char in {"'", "’"} and previous.isalnum()
                           and (following.isalnum() or (previous.lower() == "s" and closing != char)))
        if closing is not None:
            outside.append(" ")
            if char == closing and not word_apostrophe:
                closing = None
        elif char in {'"', "“", "‘"} or (char == "'" and not word_apostrophe):
            closing = {'"': '"', "“": "”", "‘": "’", "'": "'"}[char]
            outside.append(" ")
        elif char in {"”", "’"} and not word_apostrophe:
            outside.append(char)
            return "".join(outside) + text[index + 1:], ["Unmatched closing quotation mark."]
        else:
            outside.append(char)
    return "".join(outside), ["Unclosed quotation."] if closing is not None else []


def review_schema(ledger, draft):
    fact_keys = [f["key"] for f in ledger["facts"]]
    count = len(draft["sentences"])
    return {"type": "object", "properties": {
        "fact_checks": {"type": "array", "minItems": len(fact_keys), "maxItems": len(fact_keys), "items": {
            "type": "object", "properties": {
                "fact": {"type": "string", "enum": fact_keys or [""]},
                "sentences": {"type": "array", "maxItems": count, "items": {"type": "integer", "minimum": 1, "maximum": max(1, count)}},
                "verdict": {"type": "string", "enum": ["preserved", "missing", "distorted", "omitted_incidental", "excluded"]},
                "reason": {"type": "string", "maxLength": 180},
            }, "required": ["fact", "sentences", "verdict", "reason"], "additionalProperties": False,
        }},
        "sentence_checks": {"type": "array", "minItems": count, "maxItems": count, "items": {
            "type": "object", "properties": {
                "sentence": {"type": "integer", "minimum": 1, "maximum": max(1, count)},
                "verdict": {"type": "string", "enum": ["supported", "unsupported", "wrong_attribution", "identity_guess", "ambiguous", "bad_style"]},
                "reason": {"type": "string", "maxLength": 180},
            }, "required": ["sentence", "verdict", "reason"], "additionalProperties": False,
        }},
        "paragraph_issues": {"type": "array", "maxItems": 8, "items": {"type": "string", "minLength": 1, "maxLength": 180}},
    }, "required": ["fact_checks", "sentence_checks", "paragraph_issues"], "additionalProperties": False}


def validate_review(review, draft, ledger):
    """Require exhaustive, consistent checks. This cannot prove a model verdict true."""
    if not isinstance(review, dict) or set(review) != {"fact_checks", "sentence_checks", "paragraph_issues"}:
        return ["Reviewer must return complete fact_checks, sentence_checks and paragraph_issues."]
    checks, sentences, paragraph = (review[key] for key in ("fact_checks", "sentence_checks", "paragraph_issues"))
    if not isinstance(checks, list) or not isinstance(sentences, list) or not isinstance(paragraph, list):
        return ["Reviewer checks must be lists."]
    if ([c.get("fact") if isinstance(c, dict) else None for c in checks] != [f["key"] for f in ledger["facts"]]
            or [c.get("sentence") if isinstance(c, dict) else None for c in sentences] != list(range(1, len(draft["sentences"]) + 1))):
        return ["Reviewer must check every fact and sentence exactly once in order."]
    errors = []
    for fact, check in zip(ledger["facts"], checks):
        if set(check) != {"fact", "sentences", "verdict", "reason"} or not isinstance(check["reason"], str):
            errors.append(f"Invalid review for {fact['key']}.")
            continue
        expected = [i for i, sentence in enumerate(draft["sentences"], 1) if fact["key"] in sentence["facts"]]
        if (not isinstance(check["sentences"], list) or any(type(i) is not int for i in check["sentences"])
                or check["sentences"] != expected):
            errors.append(f"Reviewer coverage for {fact['key']} does not match actual citations.")
        verdict = check["verdict"]
        if not fact["usable"]:
            if verdict != "excluded" or expected:
                errors.append(f"Unusable fact {fact['key']} must remain excluded.")
        elif verdict == "preserved" and expected:
            pass
        elif verdict == "omitted_incidental" and not fact["required"] and not expected and check["reason"].strip():
            pass
        else:
            errors.append(f"Fact {fact['key']}: {verdict}: {check['reason']}")
    for check in sentences:
        if (set(check) != {"sentence", "verdict", "reason"} or type(check["sentence"]) is not int
                or not isinstance(check["reason"], str) or check["verdict"] != "supported"):
            errors.append(f"Sentence {check.get('sentence')}: {check.get('verdict')}: {check.get('reason')}")
    if any(not isinstance(issue, str) or not issue.strip() for issue in paragraph):
        errors.append("Invalid paragraph issue list.")
    else:
        errors.extend("Paragraph: " + issue for issue in paragraph)
    return errors
