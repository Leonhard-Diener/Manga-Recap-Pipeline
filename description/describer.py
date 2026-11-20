"""Describe and filter manga pages, panel by panel.

Numbered whole-page requests return model wording unchanged. Local tasks remain
separate and remote tasks share one combined response. Each request permits one
format repair; unresolved remote pages do not stop later pages from processing.
Chapter checkpoints are written here; runners publish final JSON.
"""

from __future__ import annotations

import os
import json
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import config
from common.json_io import write_json
from common.progress import show_progress
from common.model_loading import configure_windows_safetensors_loader
from common.utils import list_images, move_to_device
from description.format import FormatResponseError, parse_json, validate_format, validate_grounding
from description.quality import (
    character_description_is_useful,
)
from description.response import parse_page_response
from description.schema import (
    build_prefix_allowed_tokens_fn,
    character_response_schema,
    page_response_schema,
    panel_observation_schema,
    panel_description_schema,
    text_evidence_schema,
    transformation_response_schema,
)
from story_entities.manifest import file_hash
from story_entities.matching import stable_id

DESCRIPTION_PIPELINE_VERSION = "focused_extraction_v17_lettering_before_effects"


def _is_remote_backend(processor) -> bool:
    """Return whether ``processor`` is a remote description API client."""
    # ``MagicMock`` fabricates arbitrary attributes, so require the literal
    # marker instead of relying on truthiness (the local unit tests mock the
    # Transformers processor).
    return getattr(processor, "is_remote_description_client", False) is True


def _remote_result(client, image_path, prompt, schema, token_limit, *, _repair=False,
                   character_labels=None, role_only=False) -> dict:
    """Normalize one remote completion to the metrics used by this module."""
    try:
        options = {}
        if getattr(client, "provider_name", None) == "google":
            options["character_labels"] = character_labels
        completion = client.complete_image(
            image_path=Path(image_path), prompt=prompt,
            schema=schema, max_tokens=token_limit, **options,
        )
    except RuntimeError as error:
        if "returned empty content" not in str(error):
            raise
        completion = {"content": "", "finish_reason": "empty"}
    raw = completion.get("wire_content", completion["content"]).strip()
    content = completion["content"].strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL | re.IGNORECASE)
    json_text = fence.group(1).strip() if fence else content
    try:
        if completion.get("validation_error"):
            raise ValueError(completion["validation_error"])
        if role_only and raw not in {"story", "reference", "extra", "uncertain"}:
            raise ValueError("Expected exactly one page-role word")
        if schema is None and json_text in {"story", "reference", "extra", "uncertain"}:
            payload = json_text
        else:
            payload = parse_json(json_text)
        if schema is not None:
            validate_format(payload, schema)
            if character_labels is not None:
                validate_grounding(payload, character_labels)
        if completion.get("finish_reason") == "length":
            raise ValueError("Response was truncated")
    except (TypeError, ValueError) as error:
        if _repair:
            raise FormatResponseError(f"Invalid model response after one format repair: {error}", [raw]) from error
        repair_limit = token_limit
        if completion.get("finish_reason") == "length":
            # Include the provider's thinking allowance when growing a truncated repair.
            repair_limit += int(completion.get("token_limit") or token_limit)
        try:
            repaired = _remote_result(
                client, image_path,
                prompt + f"\nThe previous response failed validation: {error}\n"
                "Return the complete requested JSON with this error corrected. "
                "Keep descriptions grounded in the image; do not guess a replacement label.",
                schema, repair_limit, _repair=True, character_labels=character_labels,
                role_only=role_only,
            )
        except FormatResponseError as failure:
            failure.responses.insert(0, raw)
            raise
        except RuntimeError as failure:
            if getattr(failure, "stop_run", False) is True:
                raise
            raise FormatResponseError(f"Format repair request failed: {failure}", [raw]) from failure
        repaired["prior_raw_responses"] = [raw]
        repaired["retry_count"] = 1
        repaired["generation_time"] += float(completion.get("elapsed_seconds", 0.0))
        repaired["generated_tokens"] += int(completion.get("completion_tokens", 0))
        return repaired
    return {
        "payload": payload,
        "raw_response": raw,
        "generation_time": float(completion.get("elapsed_seconds", 0.0)),
        "generated_tokens": int(completion.get("completion_tokens", 0)),
        "token_limit": completion.get("token_limit", token_limit),
        "truncated": completion.get("finish_reason") == "length",
        "retry_count": 0,
        "prior_raw_responses": [],
        **{key: completion[key] for key in (
            "thought_summary", "raw_api_response", "response_archive", "thinking_tokens",
        ) if key in completion},
    }


def _mark_remote_provenance(value, prefix: str):
    """Replace local-model provenance labels in remote checkpoint output."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "provenance" and isinstance(item, str):
                for source in ("qwen", "openrouter_vision", "google_vision", "remote_vision"):
                    if item.startswith(source + "_"):
                        item = prefix + item[len(source):]
                        break
            result[key] = _mark_remote_provenance(item, prefix)
        return result
    if isinstance(value, list):
        return [_mark_remote_provenance(item, prefix) for item in value]
    return value


def _with_oom_batch_fallback(call, requests):
    """Retry only CUDA-memory failures, recursively reducing batch size."""
    try:
        return call(requests)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower() or len(requests) <= 1:
            raise
        import torch
        torch.cuda.empty_cache()
        midpoint = len(requests) // 2
        return (
            _with_oom_batch_fallback(call, requests[:midpoint])
            + _with_oom_batch_fallback(call, requests[midpoint:])
        )

VISIBLE_EVIDENCE_RULES = """\

Before describing an effect such as lightning, first check whether its shapes
could be Japanese or Chinese characters. You do not need to read the characters.
Only describe a physical effect when you can clearly distinguish it from
lettering. If unsure, skip the effect rather than guess.

First identify the visible posture, then held objects, then any clearly
supported action. Do not infer unseen body positions or motion.
Omit uncertain details; use a:[] when no action is clear.
Green borders with purple numbers mark panels. Cyan boxes with red C labels
mark character occurrences, not panels. Panel descriptions cover the entire
green region, including artwork outside C boxes. Only character appearances
are restricted to their own C boxes.
C labels are not identities. Keep repeated depictions separate;
do not invent interactions between them.
Describe unboxed subjects too, using null action endpoints. Give a target only
when the action is visibly directed at it; otherwise use null.
"""

PAGE_PROMPT = """\
Analyze the visible content of this numbered manga page. Return exactly one compact JSON object using
the short keys below. The program expands them afterward. Do not copy values
from these instructions; derive every value from the supplied image.

Required root keys:
- r: page role
- p: panel objects, each with n (number) and d (description)
- e: event objects

Output minified JSON without spaces or newlines. Each panel description must be
one dense sentence of at most 16 words, retaining the visible characters,
setting, action, and important object. Do not repeat "Panel 1" inside a
description. IGNORE written dialogue, narration, titles, and sound effects when
writing d; never transcribe or quote them there. Event descriptions
must be at most 8 words and event evidence at most 12 words. Never omit any
required JSON key, even when its list is empty.

Page role is classified in a separate request. Always return r:"uncertain";
do not spend effort deciding the page role. Retain and describe all numbered
regions. Do not treat a cover or illustration as a story event.

Describe EVERY numbered panel briefly, including humans, humanoids, creatures,
animals, dragons, demons, and other visible beings. Never filter for humans only.

Also record possible character changes or related events visible on THIS page.
Allowed event types: transformation, reversion, disguise_reveal, possession,
summoning, splitting, fusion. Keep summoning and possession distinct from a
character changing form; they do not prove two beings are the same character.

For each event in e use exactly these short keys: t (type), sp (source panel),
tp (target panel), sd (brief source appearance), td (brief target appearance),
x (evidence), xt (evidence type), and c (certainty from 0 to 100).

evidence_type must be visible_change, explicit_text, or ambiguous_sequence.
The supplied image may have printed text masked. Never use explicit_text when
text is masked and never guess what hidden text said. For ambiguous_sequence explain
the uncertainty (for example, the creature could be a separate arrival).
Use c:100 only when the same individual is directly and continuously shown
changing appearance, with no scene cut, obscuring transition, possible arrival,
or other plausible explanation. Otherwise use a lower value. Explicit text or
similar appearances alone are never c:100.
Both panel references must occur in panels. They may be the same panel if the
change happens within one panel. Do not invent off-page endpoints.
Return e: [] if no event is supported. Do not fabricate an event to fill
the schema. Similar looks, a different facial expression, or two neighboring
panels alone do not demonstrate transformation. Use local visual descriptions,
not guessed names or canonical character IDs. These events are only proposals.
Static co-presence is not summoning. Only report an event when the image shows
the change/arrival itself or clearly readable text explicitly states it.

Rules:
- Describe every visible character or creature.
- Describe their visible appearance briefly.
- Describe where the scene takes place.
- Describe exactly what each character is visibly doing.
- Describe important visible objects when relevant.
- Use only information directly visible in the image.
- Do not guess names, identities, relationships, intentions, or story context.
- Do not invent events.
- Do not infer events on pages that were not supplied.
- Only quote text that is clearly readable.
- Keep each field concise.
- No reasoning.
- No markdown or text outside the JSON object.
""" + VISIBLE_EVIDENCE_RULES

GROUNDING_PROMPT = """\

The page also contains cyan body boxes with red labels C1, C2, and so on.
These are request-local visible-character labels, not names. Use only the
labels listed below and never guess who a character is across pages.

Additionally require root key c. It must be an object with every listed label
as a key and a neutral visible appearance phrase of at most 6 words as its
value, without a name or action. Give concrete visible traits such as hair,
clothing, horns, wings, body type, or species. Do not use a panel number or an
action as an appearance. If the subject is obscured, state the visible traits
and uncertainty instead of inventing hidden features.

Every panel object additionally requires a, an array of directly visible
actions. Each action uses s (actor label or null), t (target label or null), and
x (visible action in at most 8 words). Use null when the actor or target has no
reliable C label. Do not record thoughts, motives,
relationships, dialogue meaning, or off-panel actions.

Every transformation-related event additionally requires si and ti: the exact
source and target C labels. Omit the event if either endpoint lacks a reliable
label. A label may only be referenced from the panel containing its box.
"""

ROLE_PROMPT = """\
Classify this complete original manga page using exactly one word:
story, reference, extra, or uncertain.
story = sequential scene content. reference = profiles, stats, or cast guides.
extra = non-story material or clearly unrelated standalone content.
uncertain = insufficient visual evidence.
Story content takes priority on mixed pages: a title, author credit, or chapter
heading does not make a page extra when it also advances a scene or contains
story dialogue/narration, including a narrative splash page. Use extra only when
there is no story content or the content is clearly unrelated standalone material.
Extra-like styling is not enough. A new scene or unfamiliar character does not
prove unrelatedness. If the distinction is unclear, use uncertain, not extra.
Judge layout and purpose; do not repeat or explain any printed text.
"""

CHARACTER_PROMPT = """\
Describe only the visible appearance of every cyan-boxed character label listed
below. Return one compact JSON object whose exact keys are those labels. Each
value must be {{"d":"visible appearance","r":"type"}}, where r
is live_character, toy, statue, image, or uncertain. A figurine, doll, poster,
photograph, screen depiction, or printed character is not live_character. Use
concrete traits such as hair, clothing, horns, wings, build, or species. Do not describe actions,
expressions, panel numbers, text, names, identities, or relationships. Never
invent hidden features to fill the phrase; state partial/obscured visibility when
necessary, using whatever concrete trait is actually visible. A box can contain
a close-up or a repeated depiction, not a new person or a group of nearby people.
Use at most 10 words, with no minimum length. Omit uncertain traits;
use d:"" if none are clear.
Labels: {labels}
"""

CHARACTER_CROP_PROMPT = """\
The image is a tight crop of one detected character. Describe only that visible
subject's appearance. Return exactly one compact JSON object with key {label}
and value {{"d":"visible appearance","r":"type"}}. Type r
must be live_character, toy, statue, image, or uncertain. A figurine, doll,
poster, photograph, screen depiction, or printed character is not live_character. Use concrete
traits such as hair, clothing, horns, wings, build, or species. Do not describe
actions, expressions, text, annotations, names, identities, or relationships.
State partial visibility when necessary; do not invent features outside the crop.
Use at most 10 words, with no minimum length. Omit uncertain traits;
use d:"" if none are clear.
"""

DESCRIPTION_ONLY_PROMPT = """\
Describe every numbered manga panel using only visible image content. Return
exactly {{"p":[{{"n":number,"d":"description"}}]}} with each required panel
number exactly once. Each description is one complete sentence of at most 16
words covering visible characters, setting, action, and important objects.
Ignore all dialogue, narration, titles, signs, labels, and sound effects. Do not
write panel numbers inside descriptions. Do not infer names, motives, or plot.
Required panel numbers: {panels}
""" + VISIBLE_EVIDENCE_RULES

PANEL_OBSERVATION_PROMPT = """\
Describe every numbered manga panel using only visible image content and record
only directly visible character actions. Return exactly
{{"p":[{{"n":number,"d":"description","a":[]}}]}} with every required panel
number exactly once. Descriptions must be complete sentences of at most 20
words covering visible subjects, setting, action, and important objects. Ignore dialogue,
narration, titles, signs, labels, and sound effects; do not infer their meaning.
Each action must use s (actor label or null), t (target label or null), and x (a
concrete visible verb phrase). Use s:null only when a visible acting character
has no reliable label; this reports missing detector coverage rather than a
grounded action. Emit an action only when it is directly visible; otherwise
return a as an empty array. Appearance and emotion alone are not actions. C
labels are local annotations, never names, and must not appear in d or x.
Required panels: {panels}. Labels by panel: {labels}.
""" + VISIBLE_EVIDENCE_RULES

TEXT_EVIDENCE_PROMPT = """\
Transcribe only clearly readable printed text in every numbered manga panel.
Skip sound effects and decorative lettering.
Read text throughout each green panel region, not just inside C boxes.
Return exactly {{"p":[{{"n":number,"t":[]}}]}} with every required panel number
once. Each t item uses k (dialogue, thought, narration, sign, or publication),
x (the exact visible wording), and s (the visible speaker label or null). Preserve
the wording; do not summarize, translate, complete cut-off text, infer obscured
letters, or use outside story knowledge.
For dialogue, distinguish a speech tail from the bubble's border design.
Bubbles may have jagged edges, sharp corners or repeated spikes to express
shouting or emphasis. These points are not automatically speaker tails.

On an otherwise round bubble, a single pointed extension may be the tail,
even if very small; shape alone does not prove its speaker. With many
sharp corners, be especially cautious: use s:null rather than guess which
point is a tail.

Assign s only when a distinct directional extension, separate from the
general border pattern, clearly points toward one supplied C-labeled
character in the same panel. Do not select a decorative spike merely
because it points toward a nearby person.

Some bubbles have no identifiable tail. If no clear tail exists, or you
cannot distinguish a tail from the border design, use s:null.
Keep the readable text as dialogue. Do not infer its speaker from
proximity, facial expression or what the line means.
Ignore red/cyan annotations and
panel numbers. If text is unreadable or absent, return an empty t array.
Ignore incidental text outside bubbles. Keep story-relevant captions, names,
ranks and reveals as narration, lettering on scene objects as sign, and titles,
credits or promotion as publication; all have s:null. Outside-bubble text is
not dialogue or thought without a visible speech/thought cue. Importance alone
does not make it a character's thought. Do not assign speech by proximity,
matching appearance, guessing who would say it, or naming an addressee.
A speech bubble is dialogue even with an unknown speaker, not a thought.
Required panels: {panels}. Labels by panel: {labels}.
"""

TRANSFORMATION_ONLY_PROMPT = """\
Inspect only for a directly visible character change on this numbered manga
page. Return exactly {{"e":[]}} or event objects using t,sp,tp,sd,td,x,xt,c,si,ti.
Allowed types: transformation, reversion, disguise_reveal, possession,
summoning, splitting, fusion. Labels si and ti must be exact visible C labels
from their respective panels. Similar appearance, scene cuts, co-presence, or
neighboring panels do not prove a change. Never use printed text as evidence.
Use c:100 only for a directly continuous visible change; otherwise be
conservative. Panels: {panels}. Labels: {labels}.
"""

OPENROUTER_COMBINED_PROMPT = """\
Extract only visible evidence from this manga page. Return JSON matching the
schema. Omit uncertain claims rather than guessing. Do not infer names,
identities, motives or off-page events. Ignore colored annotations as content.
""" + VISIBLE_EVIDENCE_RULES + """\

For dialogue, distinguish a speech tail from the bubble's border design.
Bubbles may have jagged edges, sharp corners or repeated spikes to express
shouting or emphasis. These points are not automatically speaker tails.

On an otherwise round bubble, a single pointed extension may be the tail,
even if very small; shape alone does not prove its speaker. With many
sharp corners, be especially cautious: use s:null rather than guess which
point is a tail.

Assign s only when a distinct directional extension, separate from the
general border pattern, clearly points toward one supplied C-labeled
character in the same panel. Do not select a decorative spike merely
because it points toward a nearby person.

Some bubbles have no identifiable tail. If no clear tail exists, or you
cannot distinguish a tail from the border design, use s:null.
Keep the readable text as dialogue. Do not infer its speaker from
proximity, facial expression or what the line means.

p: Every required panel once, with n (number), d (brief visual description),
and a (visible actions). Cover all views in a region without inventing transitions.
Keep printed wording and C labels out of prose. Actions use s (actor label or
null), t (target label or null), x (action). Unknown identity does not hide an action.

c: Every supplied C label, with d (visible appearance, at most 10 words) and
r (live_character, toy, statue, image, uncertain). Depictions and toys are not
live characters. Use traits, not actions or camera-shot names. No minimum
length. Omit uncertain traits; use d:"" if none are clear.

x: {{"p":[{{"n":number,"t":[]}}]}} for every required panel.
Each text item uses k (kind), x (exact readable wording), s (speaker or null).
Skip sound effects and decorative lettering. Do not translate or complete text.
Ignore incidental text outside bubbles. Keep story-relevant captions, names,
ranks and reveals as narration, lettering on scene objects as sign, and titles,
credits or promotion as publication. Outside-bubble text is not dialogue or
thought without a visible speech/thought cue. Importance alone does not make
it a character's thought. A speech bubble is dialogue even with an unknown speaker.
Narration, signs and publication have s:null. Check each bubble independently.
Unreadable text is omitted; readable dialogue stays even with s:null.
Read text throughout each green panel region, not just inside C boxes.

e: Only supported transformation, reversion, disguise_reveal, possession,
summoning, splitting or fusion. Use t (type), sp/tp (source/target panel),
sd/td (appearances), si/ti (labels when supplied), x (evidence), xt (visible_change,
explicit_text or ambiguous_sequence), c (certainty 0-100). Similarity, neighboring
panels or a spoken claim do not prove a change. Return [] when unsupported.

r: story for scene content; reference for profiles/stats/guides; extra only for
non-story material or clearly unrelated standalone content; uncertain if unclear.
On mixed title/story pages, story takes priority. Extra-like styling is not enough.
A new scene or unfamiliar character does not prove unrelatedness. If the
distinction is unclear, use uncertain, not extra.

Required panels: {panels}
Labels by panel: {labels}
"""


def _openrouter_bundle_schema(expected_panels, character_labels) -> dict:
    """One strict response containing all remote page-analysis tasks."""
    labels = character_labels or []
    if labels:
        schema = page_response_schema(expected_panels, labels)
    else:
        # An empty JSON-Schema enum is rejected by some OpenRouter providers.
        # Keep actions available with null endpoints and require an empty
        # character map without introducing impossible label values.
        schema = page_response_schema(expected_panels, None)
        action = {
            "type": "object",
            "properties": {
                "s": {"type": "null"},
                "t": {"type": "null"},
                "x": {
                    "type": "string", "minLength": 1,
                    "maxLength": config.DESCRIPTION_MAX_ACTION_CHARS,
                },
            },
            "required": ["s", "t", "x"],
            "additionalProperties": False,
        }
        panel_item = schema["properties"]["p"]["items"]
        panel_item["properties"]["a"] = {
            "type": "array", "items": action, "maxItems": 8,
        }
        panel_item["required"].append("a")
        schema["properties"]["c"] = {
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        }
        schema["required"].append("c")
    # Remote models have room for multi-action accounts; keep local budgets and
    # constrained decoding limits unchanged.
    panel_item = schema["properties"]["p"]["items"]
    panel_item["properties"]["d"]["maxLength"] = 600
    panel_item["properties"]["a"]["items"]["properties"]["x"]["maxLength"] = 240
    text_schema = text_evidence_schema(expected_panels, character_labels or [])
    schema["properties"]["x"] = text_schema
    schema["required"].append("x")
    # ModelRun's xgrammar compiler gives ``enum`` dispatch priority and rejects
    # range/length siblings even though the schema is valid JSON Schema. Enum
    # values already enforce those constraints, so remove only the redundant
    # siblings from enum nodes in the OpenRouter-specific schema.
    def make_provider_compatible(value) -> None:
        if isinstance(value, dict):
            if "enum" in value:
                for keyword in ("minimum", "maximum", "minLength", "maxLength", "pattern"):
                    value.pop(keyword, None)
            for child in value.values():
                make_provider_compatible(child)
        elif isinstance(value, list):
            for child in value:
                make_provider_compatible(child)

    make_provider_compatible(schema)
    return schema


def load_describer():
    """Load the page-description model, processor, and run a warmup pass."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.backends.cuda.matmul.allow_tf32 = False
    if configure_windows_safetensors_loader():
        print("Using Windows-safe safetensors pread loading.", flush=True)

    processor = AutoProcessor.from_pretrained(
        config.QWEN_MODEL_PATH,
        min_pixels=config.DESCRIPTION_MIN_PIXELS,
        max_pixels=config.DESCRIPTION_MAX_PIXELS,
    )
    # Decoder-only batched generation must pad on the left so every response
    # starts after the same final prompt position.
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        config.QWEN_MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
    ).eval()

    print("Preparing structured-output tokenizer cache...", flush=True)
    build_prefix_allowed_tokens_fn(processor.tokenizer, [1])
    _warmup(processor, model)
    return processor, model


def _warmup(processor, model) -> None:
    """Initialize CUDA/model kernels before real inference."""
    import torch

    messages = [{"role": "user", "content": [{"type": "text", "text": "Reply with exactly: OK"}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=config.DESCRIPTION_ENABLE_THINKING,
    )
    inputs = move_to_device(inputs, model.device)

    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
    torch.cuda.synchronize()


def _parse_role(value: str) -> str:
    return value if value in {"story", "reference", "extra", "uncertain"} else "uncertain"


def _deterministic_page_role(source: dict) -> dict | None:
    """Honor high-confidence non-model metadata; otherwise defer safely."""
    role = source.get("page_role")
    if source.get("page_role_source") == "review" and role in (
        "story", "reference", "extra",
    ):
        return {"page_role": role, "source": "review", "raw_response": ""}
    layout = str(source.get("layout", "")).lower()
    if layout in {"credits", "advertisement"}:
        return {"page_role": "extra", "source": "layout_metadata", "raw_response": ""}
    if layout in {"profile", "character_profile", "stat_sheet", "cast_guide"}:
        return {"page_role": "reference", "source": "layout_metadata", "raw_response": ""}
    return None


def _resolve_model_page_role(decision: dict, source: dict, page_count: int) -> dict:
    """Use the supplied classification without edge-based role rewriting."""
    return {**decision, "source": decision.get("source", "description_classifier")}


def classify_pages(page_paths, processor, model, *, _repair=False) -> list[dict]:
    """Run the small page-role task on unmasked full pages."""
    if not page_paths:
        return []
    token_limit = max(4, config.CLEANUP_MAX_NEW_TOKENS)
    if _is_remote_backend(processor):
        results = []
        for path in page_paths:
            remote = _remote_result(processor, path, ROLE_PROMPT, None, token_limit, role_only=True)
            results.append({
                "page_role": _parse_role(remote["raw_response"]),
                "raw_response": remote["raw_response"],
                "generation_time": remote["generation_time"],
                "generated_tokens": remote["generated_tokens"],
                "retry_count": remote["retry_count"],
                "prior_raw_responses": remote["prior_raw_responses"],
            })
        return results

    import torch
    # apply_chat_template expects a batch as ``list[list[message]]``. A flat
    # list of messages is one conversation, which produces only one output row
    # even when several page images are present.
    conversations = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": str(Path(path).resolve())},
            {"type": "text", "text": ROLE_PROMPT},
        ],
    }] for path in page_paths]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        enable_thinking=False,
    )
    inputs = move_to_device(inputs, model.device)
    input_length = int(inputs["input_ids"].shape[-1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max(4, config.CLEANUP_MAX_NEW_TOKENS),
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    results = []
    for row in range(len(page_paths)):
        generated = output_ids[row, input_length:]
        raw = processor.decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        generated_tokens, _ = _generated_token_info(
            generated, processor.tokenizer.pad_token_id,
            processor.tokenizer.eos_token_id, max(4, config.CLEANUP_MAX_NEW_TOKENS),
        )
        results.append({
            "page_role": _parse_role(raw), "raw_response": raw,
            "generation_time": elapsed / len(page_paths),
            "generated_tokens": generated_tokens,
        })
    for row, result in enumerate(results):
        if result["raw_response"] in {"story", "reference", "extra", "uncertain"}:
            continue
        if _repair:
            raise FormatResponseError("Invalid role response after one format repair.", [result["raw_response"]])
        try:
            retry = classify_pages([page_paths[row]], processor, model, _repair=True)[0]
        except FormatResponseError as failure:
            failure.responses.insert(0, result["raw_response"])
            raise
        retry["generation_time"] += result["generation_time"]
        retry["generated_tokens"] += result["generated_tokens"]
        retry["retry_count"] = 1
        retry["prior_raw_responses"] = [result["raw_response"]]
        results[row] = retry
    return results


_REPRESENTATION_TYPES = {"live_character", "toy", "statue", "image", "uncertain"}


def _focused_character_value(value) -> tuple[str, str]:
    """Parse the typed character value while failing closed on old responses."""
    if isinstance(value, dict):
        description = value.get("d")
        representation = value.get("r")
        if isinstance(description, str) and representation in _REPRESENTATION_TYPES:
            return description, representation
    if isinstance(value, str):
        return value, "uncertain"
    return "", "uncertain"


def describe_character_pages(page_requests, processor, model, *, _repair=False) -> list[dict]:
    """Run a focused appearance-only pass over grounded page previews."""
    if not page_requests:
        return []
    if _is_remote_backend(processor):
        results = []
        for request in page_requests:
            page_path, labels = request[0], request[1]
            bundle = processor.page_bundle(page_path)
            if bundle is None:
                raise RuntimeError(
                    f"Missing combined remote response for {Path(page_path).name}."
                )
            payload = bundle.get("c", {})
            raw_response = json.dumps(payload, separators=(",", ":"))
            characters, errors, rejected = [], [], []
            for label in labels:
                name = label["label"]
                value = payload.get(name) if isinstance(payload, dict) else None
                cleaned, representation = _focused_character_value(value)
                if not character_description_is_useful(cleaned) and not (
                    isinstance(value, dict) and value.get("d") == ""
                    and value.get("r") in _REPRESENTATION_TYPES
                ):
                    errors.append(f"Character {name} has no useful focused descriptor.")
                    rejected.append({
                        "kind": "character_description", "label": name,
                        "reasons": ["invalid_focused_descriptor"], "value": cleaned,
                    })
                    continue
                characters.append({
                    "label": name,
                    "detection_id": label["detection_id"],
                    "panel_number": label.get("panel_number"),
                    "visual_description": cleaned,
                    "representation_type": representation,
                    "representation_status": "model_observation",
                    "status": "pending_reid",
                    "appearance_status": "usable" if cleaned else "unknown",
                    "provenance": (
                        getattr(processor, "provenance_prefix", "remote_vision")
                        + "_focused_character_observation"
                    ),
                })
            results.append({
                "characters": characters,
                "grounding_errors": errors,
                "rejected_records": rejected,
                "raw_response": raw_response,
                "generation_time": 0.0,
                "generated_tokens": 0,
                "token_limit": 0,
                "status": "complete" if isinstance(payload, dict) else "invalid_output",
                "retry_count": 0,
            })
        return results

    import torch
    conversations = []
    prefix_filters = []
    normalized_requests = [
        (request[0], request[1], bool(request[2]) if len(request) >= 3 else False)
        for request in page_requests
    ]
    for page_path, labels, focused_crop in normalized_requests:
        label_names = [row["label"] for row in labels]
        prompt = (
            CHARACTER_CROP_PROMPT.format(label=label_names[0])
            if focused_crop and len(label_names) == 1 else
            CHARACTER_PROMPT.format(labels=", ".join(label_names))
        )
        if _repair:
            prompt += "\nPrevious output had invalid JSON/schema. Return the complete requested format."
        conversations.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(page_path).resolve())},
                {"type": "text", "text": prompt},
            ],
        }])
        prefix_filters.append(build_prefix_allowed_tokens_fn(
            processor.tokenizer,
            response_schema=character_response_schema(labels),
        ))

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        enable_thinking=config.DESCRIPTION_ENABLE_THINKING,
    )
    inputs = move_to_device(inputs, model.device)
    input_length = int(inputs["input_ids"].shape[-1])

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return prefix_filters[batch_id](batch_id, input_ids)

    token_limit = min(1536, max(
        160, max(len(labels) for _, labels, _ in normalized_requests) * 40 + 48,
    ))
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=token_limit,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    results = []
    for row, (_, labels, _) in enumerate(normalized_requests):
        generated_tokens, truncated = _generated_token_info(
            output_ids[row, input_length:], processor.tokenizer.pad_token_id,
            processor.tokenizer.eos_token_id, token_limit,
        )
        raw = processor.decode(
            output_ids[row, input_length:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        try:
            payload = parse_json(raw)
            validate_format(payload, character_response_schema(labels))
        except (TypeError, ValueError):
            payload = {}
        characters, errors, rejected = [], [], []
        for label in labels:
            name = label["label"]
            value = payload.get(name) if isinstance(payload, dict) else None
            cleaned, representation = _focused_character_value(value)
            if not character_description_is_useful(cleaned) and not (
                isinstance(value, dict) and value.get("d") == ""
                and value.get("r") in _REPRESENTATION_TYPES
            ):
                errors.append(f"Character {name} has no useful focused descriptor.")
                rejected.append({
                    "kind": "character_description", "label": name,
                    "reasons": ["invalid_focused_descriptor"], "value": cleaned,
                })
                continue
            characters.append({
                "label": name,
                "detection_id": label["detection_id"],
                "panel_number": label.get("panel_number"),
                "visual_description": cleaned,
                "representation_type": representation,
                "representation_status": "model_observation",
                "status": "pending_reid",
                "appearance_status": "usable" if cleaned else "unknown",
                "provenance": "qwen_focused_character_observation",
            })
        results.append({
            "characters": characters,
            "grounding_errors": errors,
            "rejected_records": rejected,
            "raw_response": raw,
            "generation_time": elapsed / len(page_requests),
            "generated_tokens": generated_tokens,
            "token_limit": token_limit,
            "status": "truncated" if truncated else ("complete" if payload else "invalid_output"),
            "retry_count": 0,
        })
    for row, result in enumerate(results):
        if result["status"] == "complete":
            continue
        if _repair:
            raise FormatResponseError("Invalid character response after one format repair.", [result["raw_response"]])
        try:
            retry = describe_character_pages([page_requests[row]], processor, model, _repair=True)[0]
        except FormatResponseError as failure:
            failure.responses.insert(0, result["raw_response"])
            raise
        retry["prior_raw_responses"] = [result["raw_response"]]
        retry["generation_time"] += result["generation_time"]
        retry["generated_tokens"] += result["generated_tokens"]
        retry["retry_count"] = 1
        results[row] = retry
    return results


def _focused_token_limit(task: str, panel_count: int, *, text_region_count: int = 0,
                         token_multiplier: int = 1) -> int:
    """Size focused output from its actual evidence, not only panel count."""
    if task == "text":
        estimated = 96 + panel_count * 32 + text_region_count * 48
        return min(4096, max(192, estimated * max(1, token_multiplier)))
    per_panel = {
        "observation": 72, "description": 48, "transformation": 36,
    }[task]
    return min(1024, max(128, 48 + panel_count * per_panel))


def describe_focused_pages(page_requests, processor, model, *, task: str, _repair=False) -> list[dict]:
    """Run one small structured visual-evidence task for each supplied page."""
    if not page_requests:
        return []
    normalized_requests = [
        (request[0], request[1], request[2], request[3] if len(request) > 3 else {})
        for request in page_requests
    ]
    conversations, prefix_filters, schemas = [], [], []
    for page_path, panels, labels, _ in normalized_requests:
        if task == "observation":
            mappings = ", ".join(
                f"{row['label']}=panel {row.get('panel_number')}"
                for row in (labels or [])
            ) or "none"
            if _is_remote_backend(processor):
                prompt = OPENROUTER_COMBINED_PROMPT.format(
                    panels=json.dumps(panels), labels=mappings,
                )
                schema = _openrouter_bundle_schema(panels, labels)
            else:
                prompt = PANEL_OBSERVATION_PROMPT.format(
                    panels=json.dumps(panels), labels=mappings,
                )
                schema = panel_observation_schema(panels, labels or [])
        elif task == "description":
            prompt = DESCRIPTION_ONLY_PROMPT.format(panels=json.dumps(panels))
            schema = panel_description_schema(panels)
        elif task == "transformation":
            prompt = TRANSFORMATION_ONLY_PROMPT.format(
                panels=json.dumps(panels),
                labels=", ".join(row["label"] for row in (labels or [])) or "none",
            )
            schema = transformation_response_schema(panels, labels or [])
        elif task == "text":
            mappings = ", ".join(
                f"{row['label']}=panel {row.get('panel_number')}"
                for row in (labels or [])
            ) or "none"
            prompt = TEXT_EVIDENCE_PROMPT.format(
                panels=json.dumps(panels), labels=mappings,
            )
            schema = text_evidence_schema(panels, labels or [])
        else:
            raise ValueError(f"Unsupported focused task: {task}")
        if _repair:
            prompt += "\nPrevious output had invalid JSON/schema. Return the complete requested format."
        conversations.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(page_path).resolve())},
                {"type": "text", "text": prompt},
            ],
        }])
        schemas.append(schema)
        if not _is_remote_backend(processor):
            prefix_filters.append(build_prefix_allowed_tokens_fn(
                processor.tokenizer, response_schema=schema,
            ))

    if _is_remote_backend(processor):
        results = []
        for (page_path, panels, labels, context), conversation, schema in zip(
            normalized_requests, conversations, schemas,
        ):
            if task == "observation":
                token_limit = min(8192, max(
                    config.DESCRIPTION_MAX_NEW_TOKENS,
                    512 + len(panels) * 160 + len(labels or []) * 48
                    + int(context.get("text_region_count", 0)) * 48,
                ))
                prompt = conversation[0]["content"][1]["text"]
                remote = _remote_result(
                    processor, page_path, prompt, schema, token_limit, character_labels=labels or [],
                )
                bundle = remote["payload"] if not remote["truncated"] else {}
                processor.store_page_bundle(page_path, bundle)
                payload = {
                    "r": bundle.get("r", "uncertain"),
                    "p": bundle.get("p", []),
                }
            else:
                bundle = processor.page_bundle(page_path)
                if bundle is None:
                    raise RuntimeError(
                        f"Missing combined remote response for {Path(page_path).name}."
                    )
                if task == "transformation":
                    payload = {"e": bundle.get("e", [])}
                elif task == "text":
                    payload = bundle.get("x", {})
                elif task == "description":
                    payload = {"p": bundle.get("p", [])}
                else:
                    payload = {}
                token_limit = 0
                remote = {
                    "raw_response": json.dumps(payload, separators=(",", ":")),
                    "generation_time": 0.0,
                    "generated_tokens": 0,
                    "truncated": False,
                }
            results.append({
                "payload": payload,
                "raw_response": remote["raw_response"],
                "status": "complete" if payload else (
                    "truncated" if remote["truncated"] else "invalid_output"
                ),
                "generation_time": remote["generation_time"],
                "generated_tokens": remote["generated_tokens"],
                "token_limit": token_limit,
                "task": task,
                "retry_count": remote.get("retry_count", 0),
                "prior_raw_responses": remote.get("prior_raw_responses", []),
            })
        return results

    import torch

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        enable_thinking=config.DESCRIPTION_ENABLE_THINKING,
    )
    inputs = move_to_device(inputs, model.device)
    input_length = int(inputs["input_ids"].shape[-1])

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return prefix_filters[batch_id](batch_id, input_ids)

    token_limit = max(
        _focused_token_limit(
            task,
            len(panels),
            text_region_count=int(context.get("text_region_count", 0)),
            token_multiplier=int(context.get("token_multiplier", 1)),
        )
        for _, panels, _, context in normalized_requests
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=token_limit,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    results = []
    for row in range(len(normalized_requests)):
        generated = output_ids[row, input_length:]
        generated_tokens, truncated = _generated_token_info(
            generated, processor.tokenizer.pad_token_id,
            processor.tokenizer.eos_token_id, token_limit,
        )
        raw = processor.decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        try:
            payload = parse_json(raw) if not truncated else {}
            validate_format(payload, schemas[row])
            validate_grounding(payload, normalized_requests[row][2])
        except (TypeError, ValueError):
            payload = {}
        results.append({
            "payload": payload,
            "raw_response": raw,
            "status": "complete" if payload else ("truncated" if truncated else "invalid_output"),
            "generation_time": elapsed / len(page_requests),
            "generated_tokens": generated_tokens,
            "token_limit": token_limit,
            "task": task,
            "retry_count": 0,
        })
    for row, result in enumerate(results):
        if result["status"] == "complete":
            continue
        if _repair:
            raise FormatResponseError(f"Invalid {task} response after one format repair.", [result["raw_response"]])
        try:
            retry = describe_focused_pages(
                [page_requests[row]], processor, model, task=task, _repair=True,
            )[0]
        except FormatResponseError as failure:
            failure.responses.insert(0, result["raw_response"])
            raise
        retry["prior_raw_responses"] = [result["raw_response"]]
        retry["generation_time"] += result["generation_time"]
        retry["generated_tokens"] += result["generated_tokens"]
        retry["retry_count"] = 1
        results[row] = retry
    return results


def _parse_text_proposals(extraction: dict, source: dict) -> tuple[list[dict], list[str]]:
    """Validate model transcription structure while keeping every row pending review."""
    payload = extraction.get("payload") if isinstance(extraction, dict) else None
    expected = {
        panel.get("panel_number"): panel.get("panel_id")
        for panel in source.get("panels", [])
        if isinstance(panel, dict) and type(panel.get("panel_number")) is int
        and isinstance(panel.get("panel_id"), str)
    }
    labels = {
        row.get("label"): row
        for row in source.get("character_labels", [])
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    }
    rows = payload.get("p") if isinstance(payload, dict) else None
    if extraction.get("status") != "complete" or not isinstance(rows, list):
        return [], ["Focused text transcription output is incomplete."]
    seen, proposals, errors = set(), [], []
    allowed_kinds = {"dialogue", "thought", "narration", "sign", "sound_effect", "publication"}
    for row in rows:
        number = row.get("n") if isinstance(row, dict) else None
        items = row.get("t") if isinstance(row, dict) else None
        if number not in expected or number in seen or not isinstance(items, list):
            errors.append("Text proposal panel coverage is malformed.")
            continue
        seen.add(number)
        for item in items:
            if not isinstance(item, dict):
                errors.append(f"Panel {number} has a malformed text proposal.")
                continue
            kind, text, speaker = item.get("k"), item.get("x"), item.get("s")
            normalized = text if isinstance(text, str) else ""
            mapping = labels.get(speaker) if speaker is not None else None
            if (kind not in allowed_kinds or not normalized
                    or (speaker is not None and (
                        mapping is None or mapping.get("panel_number") != number
                    ))):
                errors.append(f"Panel {number} has an invalid text kind, wording, or speaker.")
                continue
            proposals.append({
                "proposal_id": stable_id(
                    "text_proposal_", source.get("source_hash"), expected[number],
                    kind, normalized, speaker,
                ),
                "panel_number": number,
                "panel_id": expected[number],
                "kind": kind,
                "text": normalized,
                "speaker_label": speaker,
                "speaker_detection_id": mapping.get("detection_id") if mapping else None,
                "status": "pending_review",
                "provenance": "qwen_exact_text_transcription_proposal",
            })
    if seen != set(expected):
        errors.append("Text proposal coverage does not match the page manifest.")
        return [], errors
    return proposals, errors


def _page_messages(page_path: Path, expected_panels=None, character_labels=None) -> list[dict]:
    """Build one page conversation for the multimodal chat template."""
    panel_hint = (
        "\nSTRICT PANEL CONSTRAINT: p must contain each of these panel numbers "
        "exactly once and no other numbers: " + json.dumps(expected_panels)
        + ". Every sp and tp value must be one of these numbers."
        if expected_panels is not None else ""
    )
    grounding_hint = ""
    if character_labels is not None:
        mappings = [
            f"{row['label']}=panel {row.get('panel_number')}"
            for row in character_labels
            if isinstance(row, dict) and isinstance(row.get("label"), str)
        ]
        grounding_hint = GROUNDING_PROMPT + (
            "\nAllowed labels: " + ", ".join(mappings) + "."
            if mappings else
            "\nAllowed labels: none. Return c:{}, every a:[], and e:[]."
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(page_path.resolve())},
                {"type": "text", "text": PAGE_PROMPT + grounding_hint + panel_hint},
            ],
        }
    ]


def _generated_token_info(generated, pad_token_id, eos_token_id,
                          token_limit=None) -> tuple[int, bool]:
    """Return the useful token count and whether generation ended without EOS."""
    try:
        values = generated.detach().cpu().tolist()
        if values and isinstance(values[0], list):
            values = values[0]
    except (AttributeError, TypeError):
        return int(generated.shape[-1]), False

    while values and pad_token_id is not None and values[-1] == pad_token_id:
        values.pop()
    if eos_token_id is not None and eos_token_id in values:
        values = values[:values.index(eos_token_id) + 1]
        return len(values), False
    token_limit = token_limit or config.DESCRIPTION_MAX_NEW_TOKENS
    return len(values), len(values) >= token_limit


def describe_pages(page_requests, processor, model, *, enforce_schema=True) -> list[dict]:
    """Describe a small batch of pages with one GPU generation call."""
    if not page_requests:
        return []

    normalized_requests = [
        (request[0], request[1], request[2] if len(request) >= 3 else None)
        for request in page_requests
    ]
    panel_signatures = {
        tuple(expected_panels) if expected_panels is not None else None
        for _, expected_panels, _ in normalized_requests
    }
    if len(panel_signatures) != 1:
        raise ValueError("Batched pages must have the same expected panel numbers.")
    expected_signature = next(iter(panel_signatures))
    schema_panels = list(expected_signature) if expected_signature is not None else None
    grounded_modes = {
        character_labels is not None
        for _, _, character_labels in normalized_requests
    }
    if len(grounded_modes) != 1:
        raise ValueError("Grounded and legacy page requests cannot share one batch.")
    grounded = next(iter(grounded_modes))
    # Grounded pages emit short descriptors and actions in addition to the
    # original panel summaries. Raising only the ceiling prevents dense pages
    # from truncating; EOS still ends ordinary pages at their normal length.
    character_count = max((len(labels or []) for _, _, labels in normalized_requests), default=0)
    panel_count = len(schema_panels) if schema_panels is not None else 4
    token_limit = max(
        config.DESCRIPTION_MAX_NEW_TOKENS,
        min(2048, 160 + panel_count * 72 + character_count * 18) if grounded else 0,
    )
    prefix_allowed_tokens_fn = None
    if enforce_schema and not _is_remote_backend(processor):
        if grounded:
            # Each page has its own exact C-label set, even when pages share a
            # panel-count signature and can be generated in one batch.
            per_page_prefix_filters = [
                build_prefix_allowed_tokens_fn(
                    processor.tokenizer, schema_panels, character_labels,
                )
                for _, _, character_labels in normalized_requests
            ]

            def prefix_allowed_tokens_fn(batch_id, input_ids):
                return per_page_prefix_filters[batch_id](batch_id, input_ids)
        else:
            prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(
                processor.tokenizer, schema_panels,
            )

    conversations = [
        _page_messages(page_path, expected_panels, character_labels)
        for page_path, expected_panels, character_labels in normalized_requests
    ]

    if _is_remote_backend(processor):
        results = []
        for (page_path, expected_panels, character_labels), conversation in zip(
            normalized_requests, conversations,
        ):
            schema = page_response_schema(expected_panels, character_labels) \
                if enforce_schema else None
            prompt = conversation[0]["content"][1]["text"]
            remote = _remote_result(processor, page_path, prompt, schema, token_limit)
            result = parse_page_response(
                remote["raw_response"],
                truncated=remote["truncated"],
                expected_panels=expected_panels,
                character_labels=character_labels,
            )
            result.update(
                generation_time=remote["generation_time"],
                batch_generation_time=remote["generation_time"],
                batch_size=1,
                generated_tokens=remote["generated_tokens"],
                token_limit=token_limit,
                generation_attempts=1,
                schema_enforced=enforce_schema,
                raw_response=remote["raw_response"],
                task_metrics={
                    "combined_legacy": {
                        "elapsed_seconds": round(remote["generation_time"], 6),
                        "generated_tokens": remote["generated_tokens"],
                        "retry_count": 0,
                        "status": result["status"],
                    }
                },
            )
            results.append(result)
        return results

    import torch

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        enable_thinking=config.DESCRIPTION_ENABLE_THINKING,
    )
    inputs = move_to_device(inputs, model.device)
    input_length = int(inputs["input_ids"].shape[-1])

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generation_options = dict(
            **inputs,
            max_new_tokens=token_limit,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        if prefix_allowed_tokens_fn is not None:
            generation_options["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        output_ids = model.generate(**generation_options)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    per_page_elapsed = elapsed / len(page_requests)

    results = []
    for row, (_, expected_panels, character_labels) in enumerate(normalized_requests):
        generated = output_ids[row, input_length:]
        generated_tokens, truncated = _generated_token_info(
            generated,
            processor.tokenizer.pad_token_id,
            processor.tokenizer.eos_token_id,
            token_limit,
        )
        response = processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        result = parse_page_response(
            response,
            truncated=truncated,
            expected_panels=expected_panels,
            character_labels=character_labels,
        )
        result.update(
            generation_time=per_page_elapsed,
            batch_generation_time=elapsed,
            batch_size=len(page_requests),
            generated_tokens=generated_tokens,
            token_limit=token_limit,
            generation_attempts=1,
            schema_enforced=enforce_schema,
            raw_response=response,
            task_metrics={
                "combined_legacy": {
                    "elapsed_seconds": round(per_page_elapsed, 6),
                    "generated_tokens": generated_tokens,
                    "retry_count": 0,
                    "status": result["status"],
                }
            },
        )
        results.append(result)
    return results


def describe_page(
    page_path: Path,
    processor,
    model,
    *,
    expected_panels=None,
    character_labels=None,
) -> dict:
    """Return descriptions and proposed events from one generation."""
    return describe_pages([(page_path, expected_panels, character_labels)], processor, model)[0]


def _complete_uncertain_panel_coverage(page: dict, expected_panels) -> dict:
    """Check coverage without manufacturing replacement observations."""
    expected = sorted(int(number) for number in (expected_panels or []))
    if not expected:
        return page
    by_number = {
        row.get("panel_number"): row
        for row in page.get("panels", [])
        if isinstance(row, dict) and type(row.get("panel_number")) is int
    }
    if page.get("status") != "complete" or set(by_number) != set(expected):
        raise ValueError("Invalid model panel format/coverage; no placeholder was published.")
    return page


def describe_chapter(
    chapter_output_dir: Path,
    processor,
    model,
    *,
    filter_non_story: bool = False,
    resume_result: dict | None = None,
    progress_callback=None,
) -> dict[str, dict]:
    """Describe every numbered page preview of one chapter (step 3's output)."""
    if _is_remote_backend(processor):
        processor.clear_page_bundles()
    preview_dir = chapter_output_dir / "page_previews"
    pages = list_images(preview_dir)
    manifest_path = chapter_output_dir / "panel_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    metadata = {Path(p["page"]).name: p for p in manifest["pages"]} if manifest else {}
    if manifest is not None:
        # Old previews from removed/replaced inputs must not re-enter the story.
        by_name = {p.name: p for p in pages}
        pages = [by_name[name] for name in metadata if name in by_name]
        missing = set(metadata) - set(by_name)
        if missing:
            raise FileNotFoundError(f"Missing page previews; rerun test_panels.py: {sorted(missing)}")
        if any(not page.get("grounded_preview") for page in manifest.get("pages", [])):
            print(
                "  Warning: legacy panel manifest has no C-label previews; "
                "rerun test_panels.py to enable character-grounded actions.",
                flush=True,
            )

    resume_result = resume_result if isinstance(resume_result, dict) else {}
    results: dict[str, dict] = dict(resume_result.get("descriptions", {}))
    transformations: dict[str, dict] = dict(resume_result.get("transformations", {}))
    page_filter: dict[str, dict] = dict(resume_result.get("page_filter", {}))
    completed_pages = set(transformations)
    progress_label = f"Descriptions {chapter_output_dir.name}"
    show_progress(progress_label, len(completed_pages), len(pages))
    pending_pages = []

    def current_result() -> dict:
        show_progress(progress_label, len(transformations), len(pages))
        value = {
            "descriptions": results,
            "transformations": transformations,
        }
        if filter_non_story:
            value["page_filter"] = page_filter
        if pending_pages:
            value["pending_pages"] = list(pending_pages)
        return value

    jobs = []
    for page_path in pages:
        source = metadata.get(page_path.name, {})
        if source:
            source_path = Path(source["page"])
            if (not source_path.is_file() or file_hash(source_path) != source.get("source_hash")
                    or (source.get("preview_hash") and file_hash(page_path) != source["preview_hash"])):
                raise ValueError(f"Stale page preview/manifest: {page_path.name}; rerun test_panels.py.")
        expected_panels = (
            [panel["panel_number"] for panel in source["panels"]]
            if source else None
        )
        if page_path.name in completed_pages:
            continue
        request_path = page_path
        character_labels = None
        grounded_path = source.get("grounded_preview") if source else None
        if grounded_path:
            request_path = Path(grounded_path)
            if (not request_path.is_file()
                    or file_hash(request_path) != source.get("grounded_preview_hash")):
                raise ValueError(
                    f"Stale grounded page preview/manifest: {page_path.name}; rerun test_panels.py."
                )
            character_labels = source.get("character_labels", [])
        # Preserve the existing whole-page input. Visual-only wording is a
        # prompt instruction, not a reason to rewrite the returned prose.
        jobs.append((request_path, source, expected_panels, character_labels))

    # Role classification deliberately sees the original, unmasked page. A
    # reviewed manifest decision is stronger evidence and is never overwritten.
    classified_roles = {}
    if manifest is not None:
        pending = []
        for _, source, _, _ in jobs:
            name = Path(source["page"]).name
            deterministic = _deterministic_page_role(source)
            if deterministic:
                classified_roles[name] = deterministic
            else:
                page_order = int(source.get("page_order", 0))
                at_edge = page_order <= 2 or page_order >= max(0, len(pages) - 2)
                if at_edge:
                    pending.append((name, Path(source["page"])))
                elif not _is_remote_backend(processor):
                    # Keep the existing local inference schedule: interior
                    # pages have no role request, so no model answer exists.
                    classified_roles[name] = {
                        "page_role": "story", "source": "interior_story_default",
                        "raw_response": "", "generation_time": 0.0,
                        "generated_tokens": 0,
                    }
        if not _is_remote_backend(processor):
            for start in range(0, len(pending), max(1, config.DESCRIPTION_BATCH_SIZE)):
                chunk = pending[start:start + max(1, config.DESCRIPTION_BATCH_SIZE)]
                role_results = _with_oom_batch_fallback(
                    lambda rows: classify_pages(rows, processor, model),
                    [path for _, path in chunk],
                )
                for (name, _), decision in zip(chunk, role_results):
                    source = metadata[name]
                    classified_roles[name] = _resolve_model_page_role(
                        decision, source, len(pages),
                    )

    batch_size = 1 if _is_remote_backend(processor) else max(1, config.DESCRIPTION_BATCH_SIZE)
    batch_start = 0

    def defer_remote_page(error, batch):
        page_path, source, _, _ = batch[0]
        attempts_dir = chapter_output_dir / "description_attempts"
        write_json(attempts_dir / f"{page_path.stem}_{time.time_ns()}.json", {
            "page": source.get("page", page_path.name),
            "source_hash": source.get("source_hash"),
            "geometry_revision": source.get("geometry_revision"),
            "grounding_revision": source.get("grounding_revision"),
            "error": str(error), "raw_responses": getattr(error, "responses", []),
        })
        pending_pages.append(page_path.name)
        print(f"    Unresolved: {page_path.name}: {error}. Saved diagnostics; continuing with later pages.", flush=True)
        if progress_callback is not None:
            progress_callback(current_result())

    while batch_start < len(jobs):
        # Opening pages are commonly covers or title spreads and behave very
        # differently from normal pages. Keep the first one from holding a
        # second page inside the same autoregressive batch if it runs long.
        request_batch_size = 1 if batch_start == 0 and len(jobs) > 2 else batch_size
        batch = jobs[batch_start:batch_start + request_batch_size]
        first = batch_start + 1
        last = batch_start + len(batch)
        print(f"  Pages {first}-{last}/{len(pages)}: "
              + ", ".join(page_path.name for page_path, _, _, _ in batch), flush=True)
        requests = [(page_path, expected_panels, character_labels)
                    for page_path, _, expected_panels, character_labels in batch]
        try:
            if manifest is None:
                # Compatibility for direct library callers without current
                # manifests. The operational runner always uses the focused
                # path below and never pays for this legacy combined request.
                batch_results = describe_pages(requests, processor, model)
                observation_passes = []
            else:
                observation_requests = [
                    (
                        page_path, expected_panels, character_labels,
                        {"text_region_count": len(source.get("text_regions", []))},
                    )
                    for page_path, source, expected_panels, character_labels in batch
                ] if _is_remote_backend(processor) else requests
                observation_passes = _with_oom_batch_fallback(
                    lambda rows: describe_focused_pages(
                        rows, processor, model, task="observation",
                    ),
                    observation_requests,
                )
        except FormatResponseError as error:
            if not _is_remote_backend(processor):
                raise
            defer_remote_page(error, batch)
            batch_start += len(batch)
            continue
        except RuntimeError as error:
            if getattr(error, "stop_run", False) is True:
                raise
            if _is_remote_backend(processor):
                defer_remote_page(error, batch)
                batch_start += len(batch)
                continue
            if len(batch) == 1 or "out of memory" not in str(error).lower():
                raise
            import torch
            torch.cuda.empty_cache()
            batch_size = 1
            print("    Batch exceeded GPU memory; using one page at a time from here on.", flush=True)
            continue

        if manifest is not None:
            batch_results = []
            for observation, (_, expected_panels, character_labels) in zip(
                observation_passes, requests,
            ):
                payload = observation.get("payload")
                page = parse_page_response(
                    json.dumps(payload) if isinstance(payload, dict) else "",
                    expected_panels=expected_panels,
                    character_labels=character_labels,
                    require_characters=False,
                    require_events=False,
                )
                if observation.get("status") != "complete":
                    page["status"] = observation.get("status", "invalid_output")
                    if not page["errors"]:
                        page["errors"].append("Focused observation output is incomplete.")
                page.update(
                    generation_time=observation.get("generation_time", 0.0),
                    generated_tokens=observation.get("generated_tokens", 0),
                    token_limit=observation.get("token_limit", config.DESCRIPTION_MAX_NEW_TOKENS),
                    raw_response=observation.get("raw_response", ""),
                    generation_attempts=1 + observation.get("retry_count", 0),
                    format_repairs={"observation": observation.get("prior_raw_responses", [])},
                    task_metrics={
                        "observation": {
                            "elapsed_seconds": round(observation.get("generation_time", 0.0), 6),
                            "generated_tokens": observation.get("generated_tokens", 0),
                            "retry_count": observation.get("retry_count", 0),
                            "status": observation.get("status", "invalid_output"),
                        },
                    },
                )
                for panel in page.get("panels", []):
                    panel["provenance"] = "qwen_focused_panel_observation"
                batch_results.append(page)

        if manifest is not None:
            focused_requests_all = [
                (page_path, expected_panels, character_labels)
                for page_path, _, expected_panels, character_labels in batch
            ]
            transformation_by_row = {}
            transformation_requests = [
                (row, request) for row, request in enumerate(focused_requests_all)
                if len(request[2] or []) >= 2
            ]
            if transformation_requests:
                transformation_results = _with_oom_batch_fallback(
                    lambda rows: describe_focused_pages(
                        rows, processor, model, task="transformation",
                    ),
                    [request for _, request in transformation_requests],
                )
                transformation_by_row = {
                    row: result for (row, _), result in zip(
                        transformation_requests, transformation_results,
                    )
                }

            text_by_row = {}
            text_requests = [
                (
                    row,
                    (
                        # Speaker references need the same visible C labels as
                        # the supplied mapping; use the existing grounded image.
                        batch[row][0],
                        expected_panels,
                        character_labels,
                        {"text_region_count": len(source.get("text_regions", []))},
                    ),
                )
                for row, (_, source, expected_panels, character_labels) in enumerate(batch)
                if source.get("text_regions")
            ]
            if text_requests:
                text_results = _with_oom_batch_fallback(
                    lambda rows: describe_focused_pages(rows, processor, model, task="text"),
                    [request for _, request in text_requests],
                )
                text_by_row = {
                    row: result for (row, _), result in zip(text_requests, text_results)
                }

            for row, (page, observation_pass, request) in enumerate(zip(
                batch_results, observation_passes, focused_requests_all,
            )):
                _, expected_panels, character_labels = request
                transformation_pass = transformation_by_row.get(row, {
                    "payload": {"e": []}, "raw_response": "", "status": "skipped_impossible",
                    "generation_time": 0.0, "generated_tokens": 0,
                })
                event_payload = transformation_pass["payload"]
                compact_panels = [{
                    "n": panel["panel_number"], "d": panel["description"], "a": [],
                } for panel in page.get("panels", [])]
                synthetic = {
                    "r": "uncertain", "p": compact_panels,
                    "e": event_payload.get("e", []) if isinstance(event_payload, dict) else [],
                }
                parsed_events = parse_page_response(
                    json.dumps(synthetic),
                    expected_panels=expected_panels,
                    character_labels=character_labels,
                    require_characters=False,
                )
                page["events"] = parsed_events["events"]
                page["transformation_status"] = transformation_pass["status"]
                page["transformation_errors"] = list(parsed_events.get("errors", []))
                if transformation_pass.get("status") not in {"complete", "skipped_impossible"}:
                    page["transformation_errors"].append(
                        "Focused transformation output is incomplete; no negative finding was accepted."
                    )
                page["focused_responses"] = {
                    "observation": observation_pass["raw_response"],
                    "transformation": transformation_pass["raw_response"],
                }
                page.setdefault("format_repairs", {})["transformation"] = transformation_pass.get("prior_raw_responses", [])
                page["generation_time"] += transformation_pass["generation_time"]
                page["generated_tokens"] += transformation_pass["generated_tokens"]
                page.setdefault("task_metrics", {}).update({
                    "transformation": {
                        "elapsed_seconds": round(transformation_pass["generation_time"], 6),
                        "generated_tokens": transformation_pass["generated_tokens"],
                        "retry_count": transformation_pass.get("retry_count", 0),
                        "status": transformation_pass["status"],
                    },
                })
                text_pass = text_by_row.get(row)
                if text_pass is None:
                    page["text_proposals"] = []
                    page["text_proposal_errors"] = []
                    text_status = "skipped_no_detected_text"
                else:
                    source = batch[row][1]
                    proposals, proposal_errors = _parse_text_proposals(text_pass, source)
                    page["text_proposals"] = proposals
                    page["text_proposal_errors"] = proposal_errors
                    page["focused_responses"]["text"] = text_pass["raw_response"]
                    page.setdefault("format_repairs", {})["text"] = text_pass.get("prior_raw_responses", [])
                    page["generation_time"] += text_pass["generation_time"]
                    page["generated_tokens"] += text_pass["generated_tokens"]
                    text_status = text_pass["status"] if not proposal_errors else "needs_review"
                page.setdefault("task_metrics", {})["text"] = {
                    "elapsed_seconds": round(text_pass.get("generation_time", 0.0), 6)
                    if text_pass else 0.0,
                    "generated_tokens": text_pass.get("generated_tokens", 0) if text_pass else 0,
                    "retry_count": text_pass.get("retry_count", 0) if text_pass else 0,
                    "status": text_status,
                }

        focused_by_row = {}
        focused_requests = [
            (row, page_path, character_labels)
            for row, (page_path, _, _, character_labels) in enumerate(batch)
            if character_labels
        ]
        if focused_requests:
            focused_results = _with_oom_batch_fallback(
                lambda rows: describe_character_pages(rows, processor, model),
                [(path, labels) for _, path, labels in focused_requests],
            )
            focused_by_row = {
                request[0]: focused
                for request, focused in zip(focused_requests, focused_results)
            }

        for row, page in enumerate(batch_results):
            focused = focused_by_row.get(row)
            if not focused:
                continue
            page["characters"] = focused["characters"]
            page["grounding_errors"] = [
                error for error in page.get("grounding_errors", [])
                if not error.startswith("Character ")
                and "descriptor coverage" not in error
            ] + focused["grounding_errors"]
            page.setdefault("rejected_records", []).extend(focused["rejected_records"])
            page["character_raw_response"] = focused["raw_response"]
            page.setdefault("format_repairs", {})["character"] = focused.get("prior_raw_responses", [])
            page["character_retry_raw_responses"] = focused.get(
                "retry_raw_responses", {},
            )
            page["generation_time"] += focused.get("generation_time", 0.0)
            page["generated_tokens"] += focused.get("generated_tokens", 0)
            page.setdefault("task_metrics", {})["character"] = {
                "elapsed_seconds": round(focused.get("generation_time", 0.0), 6),
                "generated_tokens": focused.get("generated_tokens", 0),
                "retry_count": focused.get("retry_count", 0),
                "status": focused.get("status", "complete"),
            }

        for (page_path, source, expected_panels, _), page in zip(batch, batch_results):
            role_decision = classified_roles.get(Path(source.get("page", page_path.name)).name)
            if role_decision is None and _is_remote_backend(processor):
                role_decision = _resolve_model_page_role({
                    "page_role": page.get("page_role", "uncertain"),
                    "source": "openrouter_combined_page_analysis",
                    "raw_response": page.get("raw_response", ""),
                    "generation_time": 0.0,
                    "generated_tokens": 0,
                }, source, len(pages))
            if role_decision:
                page["page_role"] = role_decision["page_role"]
                page.setdefault("format_repairs", {})["page_role"] = role_decision.get("prior_raw_responses", [])
                page["generation_time"] += role_decision.get("generation_time", 0.0)
                page["generated_tokens"] += role_decision.get("generated_tokens", 0)
                page.setdefault("task_metrics", {})["page_role"] = {
                    "elapsed_seconds": round(role_decision.get("generation_time", 0.0), 6),
                    "generated_tokens": role_decision.get("generated_tokens", 0),
                    "retry_count": role_decision.get("retry_count", 0),
                    "status": "complete",
                }
            page = _complete_uncertain_panel_coverage(page, expected_panels)
            page_role = page.get("page_role", "uncertain")
            included = page["status"] in {"complete", "complete_with_uncertainty"} \
                and page_role in {"story", "uncertain"}
            print(f"    {page_path.name}: {page['generation_time']:.2f}s/image", flush=True)
            print(
                f"    Tokens:   {page['generated_tokens']}/{page.get('token_limit', config.DESCRIPTION_MAX_NEW_TOKENS)}",
                flush=True,
            )
            print(f"    Panels:   {len(page['panels'])}", flush=True)
            print(f"    Events:   {len(page['events'])} ({page['status']})", flush=True)

            for panel in page["panels"]:
                print(
                    f"    Panel {panel['panel_number']}: {panel['description']}",
                    flush=True,
                )

            if page["status"] != "complete":
                print(f"    Warning: {page['errors']}", flush=True)
            if page.get("grounding_errors"):
                print(f"    Grounding warning: {page['grounding_errors']}", flush=True)
            if filter_non_story and not included:
                print(f"    Filter:   excluded ({page_role})", flush=True)

            if included or not filter_non_story:
                results[page_path.name] = {
                    "page_role": page_role,
                    "source_hash": source.get("source_hash"),
                    "geometry_revision": source.get("geometry_revision"),
                    "grounding_revision": source.get("grounding_revision"),
                    "panels": page["panels"],
                    "characters": page.get("characters", []),
                    "grounding_errors": page.get("grounding_errors", []),
                    "quality_errors": page.get("quality_errors", {}),
                    "rejected_records": page.get("rejected_records", []),
                    "panel_retries": page.get("panel_retries", []),
                    "character_raw_response": page.get("character_raw_response", ""),
                    "character_retry_raw_responses": page.get(
                        "character_retry_raw_responses", {},
                    ),
                    "text_proposals": page.get("text_proposals", []),
                    "text_proposal_errors": page.get("text_proposal_errors", []),
                    "transformation_status": page.get("transformation_status"),
                    "transformation_errors": page.get("transformation_errors", []),
                    "focused_responses": page.get("focused_responses", {}),
                    "format_repairs": page.get("format_repairs", {}),
                    "generation_time": page["generation_time"],
                    "generated_tokens": page["generated_tokens"],
                    "task_metrics": page.get("task_metrics", {}),
                    "extraction_status": page["status"],
                }
            transformations[page_path.name] = {
                "page_role": page_role,
                "source_hash": source.get("source_hash"),
                "geometry_revision": source.get("geometry_revision"),
                "grounding_revision": source.get("grounding_revision"),
                "page_id": source.get("page_id"),
                "events": page["events"] if included or not filter_non_story else [],
                "extraction_status": page.get("transformation_status", page["status"]),
                "errors": page.get("transformation_errors", []),
                "quality_errors": page.get("quality_errors", {}),
            }
            transformations[page_path.name]["raw_response"] = page.get(
                "focused_responses", {}
            ).get("transformation", page["raw_response"])
            if filter_non_story:
                page_filter[page_path.name] = {
                    "page_role": page_role,
                    "included_in_story": included,
                    "reason": (
                        f"included_{page_role}"
                        if included else
                        f"extraction_{page['status']}"
                        if page["status"] != "complete" else
                        f"page_role_{page_role}"
                    ),
                    "source_hash": source.get("source_hash"),
                    "geometry_revision": source.get("geometry_revision"),
                    "grounding_revision": source.get("grounding_revision"),
                    "page_id": source.get("page_id"),
                    "extraction_status": page["status"],
                    "classification_source": role_decision.get("source") if role_decision else "combined_generation",
                    "classification_raw_response": role_decision.get("raw_response", "") if role_decision else "",
                    "proposed_page_role": role_decision.get("proposed_page_role") if role_decision else None,
                }
            if progress_callback is not None:
                progress_callback(current_result())

        batch_start += len(batch)

    result = current_result()
    return result


def describe_manga(
    manga_output_dir: Path,
    chapter_dirs: list[Path],
    *,
    filter_non_story: bool = False,
    backend=None,
    progress_callback=None,
) -> dict:
    """Return manga-wide results, reusing atomic current chapter checkpoints."""
    checkpoint_inputs = {}
    checkpoint_name = (
        getattr(backend, "checkpoint_name", "description_checkpoint_openrouter.json")
        if backend is not None else "description_checkpoint.json"
    )
    for chapter_dir in chapter_dirs:
        chapter_output = manga_output_dir / chapter_dir.name
        manifest_path = chapter_output / "panel_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_path}; rerun test_panels.py before descriptions."
            )
        model_config = Path(config.QWEN_MODEL_PATH) / "config.json"
        backend_fingerprint = (
            getattr(backend, "cache_key", ("remote",))
            if backend is not None else
            file_hash(model_config) if model_config.is_file() else None
        )
        backend_prompts = (OPENROUTER_COMBINED_PROMPT,) if backend is not None else ()
        fingerprint = stable_id(
            "description_checkpoint_", DESCRIPTION_PIPELINE_VERSION,
            file_hash(manifest_path), backend_fingerprint,
            config.DESCRIPTION_MIN_PIXELS, config.DESCRIPTION_MAX_PIXELS,
            config.DESCRIPTION_MAX_NEW_TOKENS, config.DESCRIPTION_BATCH_SIZE,
            config.DESCRIPTION_MAX_DESCRIPTION_CHARS, config.DESCRIPTION_MAX_ACTION_CHARS,
            config.DESCRIPTION_ENABLE_THINKING,
            PAGE_PROMPT, ROLE_PROMPT, CHARACTER_PROMPT,
            CHARACTER_CROP_PROMPT, PANEL_OBSERVATION_PROMPT,
            TEXT_EVIDENCE_PROMPT, DESCRIPTION_ONLY_PROMPT, TRANSFORMATION_ONLY_PROMPT,
            *backend_prompts,
            filter_non_story,
        )
        checkpoint_inputs[chapter_dir.name] = (
            chapter_output / checkpoint_name, fingerprint,
        )

    cached = {}
    partial = {}
    pending = []
    for chapter_dir in chapter_dirs:
        path, fingerprint = checkpoint_inputs[chapter_dir.name]
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except (OSError, ValueError, TypeError):
            value = None
        current_checkpoint = (
            isinstance(value, dict) and value.get("schema_version") == 1
            and value.get("fingerprint") == fingerprint
            and isinstance(value.get("result"), dict)
        )
        if current_checkpoint and value.get("complete", True):
            cached[chapter_dir.name] = value["result"]
        else:
            if current_checkpoint:
                partial[chapter_dir.name] = value["result"]
            pending.append(chapter_dir)

    processor = model = backend
    if pending:
        if backend is None:
            import torch
            print("Loading page-description model...")
            processor, model = load_describer()
            print("Model loaded.")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: {torch.version.cuda}")
            print(f"Device: {model.device}")
            print(f"Dtype: {next(model.parameters()).dtype}")
            print("Warmup complete.\n")
        else:
            print(
                f"Using remote description backend: {getattr(backend, 'model', 'unknown')}",
                flush=True,
            )

    all_results: dict[str, dict] = {}
    transformations: dict[str, dict] = {}
    page_filter: dict[str, dict] = {}
    unresolved: dict[str, list[str]] = {}
    # Keep every already-complete chapter in each incremental publication,
    # including cached chapters that occur later in the requested order.
    for chapter_name, chapter_result in cached.items():
        all_results[chapter_name] = chapter_result["descriptions"]
        transformations[chapter_name] = chapter_result["transformations"]
        if filter_non_story:
            page_filter[chapter_name] = chapter_result["page_filter"]

    def combined_result() -> dict:
        value = {
            "descriptions": dict(all_results),
            "transformations": {
                "schema_version": 1,
                "scope": "within_page",
                "chapters": dict(transformations),
            },
        }
        if filter_non_story:
            value["page_filter"] = {
                "schema_version": 1,
                "policy": "model_roles_mixed_story_priority_v4",
                "chapters": dict(page_filter),
            }
        if backend is not None:
            value["inference_backend"] = {
                "provider": getattr(backend, "provider_name", "openrouter"),
                "model": getattr(backend, "model", None),
                "base_url": getattr(backend, "base_url", None),
            }
        if unresolved:
            value["pending_pages"] = dict(unresolved)
        return value

    for chapter_dir in chapter_dirs:
        chapter_output_dir = manga_output_dir / chapter_dir.name
        print(f"Chapter: {chapter_dir.name}")
        chapter = cached.get(chapter_dir.name)
        if chapter is None:
            checkpoint_path, fingerprint = checkpoint_inputs[chapter_dir.name]

            def publish_page(chapter_progress, *, chapter_name=chapter_dir.name):
                published = chapter_progress
                if backend is not None:
                    published = _mark_remote_provenance(
                        chapter_progress,
                        getattr(backend, "provenance_prefix", "remote_vision"),
                    )
                write_json(checkpoint_path, {
                    "schema_version": 1,
                    "fingerprint": fingerprint,
                    "pipeline_version": DESCRIPTION_PIPELINE_VERSION,
                    "complete": False,
                    "result": published,
                })
                all_results[chapter_name] = published["descriptions"]
                transformations[chapter_name] = published["transformations"]
                if filter_non_story:
                    page_filter[chapter_name] = published["page_filter"]
                if published.get("pending_pages"):
                    unresolved[chapter_name] = published["pending_pages"]
                if progress_callback is not None:
                    progress_callback(combined_result())

            try:
                chapter = describe_chapter(
                    chapter_output_dir, processor, model,
                    filter_non_story=filter_non_story,
                    resume_result=partial.get(chapter_dir.name),
                    progress_callback=publish_page,
                )
            except FormatResponseError as error:
                write_json(chapter_output_dir / (checkpoint_name + ".failure.json"), {
                    "fingerprint": fingerprint,
                    "error": str(error), "raw_responses": error.responses,
                })
                raise
            if backend is not None:
                chapter = _mark_remote_provenance(
                    chapter, getattr(backend, "provenance_prefix", "remote_vision"),
                )
            write_json(checkpoint_path, {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "pipeline_version": DESCRIPTION_PIPELINE_VERSION,
                "complete": not bool(chapter.get("pending_pages")),
                "result": chapter,
            })
        else:
            print("  Reusing current extraction checkpoint.", flush=True)
            count = len(chapter["transformations"])
            show_progress(f"Descriptions {chapter_dir.name} (cached)", count, count)
        all_results[chapter_dir.name] = chapter["descriptions"]
        if chapter.get("pending_pages"):
            unresolved[chapter_dir.name] = chapter["pending_pages"]
        transformations[chapter_dir.name] = chapter["transformations"]
        if filter_non_story:
            page_filter[chapter_dir.name] = chapter["page_filter"]

    return combined_result()
