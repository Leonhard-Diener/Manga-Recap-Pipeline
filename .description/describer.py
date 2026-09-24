"""Step 4: describe what happens on each page, panel by panel.

One numbered page preview is analyzed in a single inference, returning panel
descriptions, proposed character events, and grounded real-name evidence. This module returns data;
callers decide whether to serialize it. No output files are written here.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import config
from common.utils import list_images, move_to_device
from description.response import parse_page_response
from story_entities.manifest import file_hash

PAGE_PROMPT = """\
Analyze this numbered manga page. Return exactly one compact JSON object using
the short keys below. The program expands them afterward. Do not copy values
from these instructions; derive every value from the supplied image.

Required root keys:
- r: page role
- p: panel objects, each with n (number) and d (description)
- e: event objects
- m: grounded name-mention objects

Be extremely compact so generation is fast. Each panel description must be one
dense sentence of at most 24 words, retaining the visible characters, setting,
action, and important object. Do not repeat "Panel 1" inside a description.
Summarize only plot-relevant readable dialogue; do not transcribe ordinary
dialogue or sound effects. Event descriptions must be at most 8 words and event
evidence at most 12 words. Name evidence remains an exact short quote. Never
omit any required JSON key, even when its list is empty.

page_role must be story, reference (profiles or character sheets), extra (credits,
covers, advertisements or decorative artwork), or uncertain. Retain and describe
all numbered regions even on non-story pages. Do not treat a cover as a story event.

Describe EVERY numbered panel briefly, including humans, humanoids, creatures,
animals, dragons, demons, and other visible beings. Never filter for humans only.

Also record possible character changes or related events visible on THIS page.
Allowed event types: transformation, reversion, disguise_reveal, possession,
summoning, splitting, fusion. Keep summoning and possession distinct from a
character changing form; they do not prove two beings are the same character.

For each event in e use exactly these short keys: t (type), sp (source panel),
tp (target panel), sd (brief source appearance), td (brief target appearance),
x (evidence), and xt (evidence type).

evidence_type must be visible_change, explicit_text, or ambiguous_sequence.
For explicit_text quote the short readable text. For ambiguous_sequence explain
the uncertainty (for example, the creature could be a separate arrival).
Both panel references must occur in panels. They may be the same panel if the
change happens within one panel. Do not invent off-page endpoints.
Return e: [] if no event is supported. Do not fabricate an event to fill
the schema. Similar looks, a different facial expression, or two neighboring
panels alone do not demonstrate transformation. Use local visual descriptions,
not guessed names or canonical character IDs. These events are only proposals.

Also extract real character names that are explicitly readable on this page.
For each usable name mention in m use exactly these short keys: n (name),
p (panel number), b (character box), x (quoted evidence), and xt (evidence type).

character_box is the visible named character, not the speech bubble, using
coordinates from 0 to 1000 relative to the entire supplied page: [left, top,
right, bottom]. evidence must be a short exact quote containing the name.
evidence_type must be self_introduction, introduced_by_other, profile_label,
or direct_address. Only return a mention when the text is clearly readable and
the named character is visibly present and can be boxed. A shouted or addressed
name refers to the addressee, not the speaker. Do not return titles, roles,
species, names inferred from appearance, or names learned from prior pages.
Return m: [] when there is no grounded explicit name evidence.

Rules:
- Describe every visible character or creature.
- Describe their visible appearance briefly.
- Describe where the scene takes place.
- Describe exactly what each character is visibly doing.
- Describe important visible objects when relevant.
- Use only information directly visible in the image.
- Do not guess names, identities, relationships, intentions, or story context;
  a name is allowed only under the explicit name_mentions evidence rules above.
- Do not invent events.
- Do not infer events on pages that were not supplied.
- Only quote text that is clearly readable.
- Keep each field concise.
- No reasoning.
- No markdown or text outside the JSON object.
"""


def load_describer():
    """Load the page-description model, processor, and run a warmup pass."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.backends.cuda.matmul.allow_tf32 = False

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


def _page_messages(page_path: Path, expected_panels=None) -> list[dict]:
    """Build one page conversation for the multimodal chat template."""
    panel_hint = (
        "\nSTRICT PANEL CONSTRAINT: p must contain each of these panel numbers "
        "exactly once and no other numbers: " + json.dumps(expected_panels)
        + ". Every sp, tp, and name-mention p value must be one of these numbers."
        if expected_panels is not None else ""
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(page_path.resolve())},
                {"type": "text", "text": PAGE_PROMPT + panel_hint},
            ],
        }
    ]


def _generated_token_info(generated, pad_token_id, eos_token_id) -> tuple[int, bool]:
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
    return len(values), len(values) >= config.DESCRIPTION_MAX_NEW_TOKENS


def describe_pages(page_requests, processor, model) -> list[dict]:
    """Describe a small batch of pages with one GPU generation call."""
    import torch

    if not page_requests:
        return []

    conversations = [
        _page_messages(page_path, expected_panels)
        for page_path, expected_panels in page_requests
    ]

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
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.DESCRIPTION_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    per_page_elapsed = elapsed / len(page_requests)

    results = []
    for row, (_, expected_panels) in enumerate(page_requests):
        generated = output_ids[row, input_length:]
        generated_tokens, truncated = _generated_token_info(
            generated,
            processor.tokenizer.pad_token_id,
            processor.tokenizer.eos_token_id,
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
        )
        result.update(
            generation_time=per_page_elapsed,
            batch_generation_time=elapsed,
            batch_size=len(page_requests),
            generated_tokens=generated_tokens,
            token_limit=config.DESCRIPTION_MAX_NEW_TOKENS,
            generation_attempts=1,
            raw_response=response,
        )
        results.append(result)
    return results


def describe_page(
    page_path: Path,
    processor,
    model,
    *,
    expected_panels=None,
) -> dict:
    """Return descriptions, proposed events, and names from one generation."""
    return describe_pages([(page_path, expected_panels)], processor, model)[0]


def describe_chapter(chapter_output_dir: Path, processor, model) -> dict[str, dict]:
    """Describe every numbered page preview of one chapter (step 3's output)."""
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

    results: dict[str, dict] = {}
    transformations: dict[str, dict] = {}
    names: dict[str, dict] = {}
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
        jobs.append((page_path, source, expected_panels))

    batch_size = max(1, config.DESCRIPTION_BATCH_SIZE)
    batch_start = 0
    while batch_start < len(jobs):
        # Opening pages are commonly covers or title spreads and behave very
        # differently from normal pages. Keep the first one from holding a
        # second page inside the same autoregressive batch if it runs long.
        request_batch_size = 1 if batch_start == 0 and len(jobs) > 2 else batch_size
        batch = jobs[batch_start:batch_start + request_batch_size]
        first = batch_start + 1
        last = batch_start + len(batch)
        print(f"  Pages {first}-{last}/{len(pages)}: "
              + ", ".join(page_path.name for page_path, _, _ in batch), flush=True)
        requests = [(page_path, expected_panels)
                    for page_path, _, expected_panels in batch]
        try:
            batch_results = describe_pages(requests, processor, model)
        except RuntimeError as error:
            if len(batch) == 1 or "out of memory" not in str(error).lower():
                raise
            import torch
            torch.cuda.empty_cache()
            batch_size = 1
            print("    Batch exceeded GPU memory; using one page at a time from here on.", flush=True)
            continue

        for (page_path, source, _), page in zip(batch, batch_results):
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

            results[page_path.name] = {
                "page_role": page.get("page_role", "uncertain"),
                "source_hash": source.get("source_hash"),
                "geometry_revision": source.get("geometry_revision"),
                "panels": page["panels"],
                "generation_time": page["generation_time"],
                "generated_tokens": page["generated_tokens"],
                "extraction_status": page["status"],
            }
            transformations[page_path.name] = {
                "page_role": page.get("page_role", "uncertain"),
                "source_hash": source.get("source_hash"),
                "geometry_revision": source.get("geometry_revision"),
                "page_id": source.get("page_id"),
                "events": page["events"],
                "extraction_status": page["status"],
                "errors": page["errors"],
            }
            transformations[page_path.name]["raw_response"] = page["raw_response"]

            # Qwen grounds in a stable 0..1000 page coordinate space. Export the
            # evidence in original pixels so ReID can compare it with detector boxes.
            source_width = source.get("width")
            source_height = source.get("height")
            mentions = []
            for mention in page.get("name_mentions", []):
                item = dict(mention)
                normalized_box = item.pop("character_box")
                item["character_box_normalized"] = normalized_box
                if source_width and source_height:
                    item["character_box"] = [
                        round(normalized_box[0] * source_width / 1000),
                        round(normalized_box[1] * source_height / 1000),
                        round(normalized_box[2] * source_width / 1000),
                        round(normalized_box[3] * source_height / 1000),
                    ]
                    item["coordinate_space"] = "original_page_pixels"
                else:
                    item["character_box"] = None
                    item["coordinate_space"] = "unverified"
                mentions.append(item)
            names[page_path.name] = {
                "page_role": page.get("page_role", "uncertain"),
                "source_hash": source.get("source_hash"),
                "geometry_revision": source.get("geometry_revision"),
                "page_id": source.get("page_id"),
                "mentions": mentions,
                "extraction_status": page.get("name_status", "complete"),
                "errors": page.get("name_errors", []),
            }
        batch_start += len(batch)

    return {"descriptions": results, "transformations": transformations, "names": names}


def describe_manga(
    manga_output_dir: Path,
    chapter_dirs: list[Path],
) -> dict:
    """Return all manga-wide results without writing JSON or other artifacts."""
    import torch

    print("Loading page-description model...")
    processor, model = load_describer()
    print("Model loaded.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {model.device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print("Warmup complete.\n")

    all_results: dict[str, dict] = {}
    transformations: dict[str, dict] = {}
    names: dict[str, dict] = {}
    for chapter_dir in chapter_dirs:
        chapter_output_dir = manga_output_dir / chapter_dir.name
        print(f"Chapter: {chapter_dir.name}")
        chapter = describe_chapter(chapter_output_dir, processor, model)
        all_results[chapter_dir.name] = chapter["descriptions"]
        transformations[chapter_dir.name] = chapter["transformations"]
        names[chapter_dir.name] = chapter["names"]

    return {
        "descriptions": all_results,
        "transformations": {
            "schema_version": 1,
            "scope": "within_page",
            "chapters": transformations,
        },
        "names": {
            "schema_version": 1,
            "scope": "grounded_visible_mentions",
            "chapters": names,
        },
    }
