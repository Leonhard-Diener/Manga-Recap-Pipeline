"""Step 4: describe what happens on each page, panel by panel.

One numbered page preview is analyzed in a single inference, returning both
panel descriptions and proposed character events. This module returns data;
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
Analyze this numbered manga page. Return exactly one JSON object:
{"page_role": "story", "panels": [{"panel_number": 1, "description": "Characters: ... Location: ... Actions: ..."}],
 "events": []}

page_role must be story, reference (profiles or character sheets), extra (credits,
covers, advertisements or decorative artwork), or uncertain. Retain and describe
all numbered regions even on non-story pages. Do not treat a cover as a story event.

Describe EVERY numbered panel briefly, including humans, humanoids, creatures,
animals, dragons, demons, and other visible beings. Never filter for humans only.

Also record possible character changes or related events visible on THIS page.
Allowed event types: transformation, reversion, disguise_reveal, possession,
summoning, splitting, fusion. Keep summoning and possession distinct from a
character changing form; they do not prove two beings are the same character.

For each event use exactly these fields:
{"type": "transformation", "source_panel": 1, "target_panel": 2,
 "source_description": "brief visible description of the being before the event",
 "target_description": "brief visible description of the being after the event",
 "evidence": "specific visible change or clearly readable text supporting the event",
 "evidence_type": "visible_change"}

evidence_type must be visible_change, explicit_text, or ambiguous_sequence.
For explicit_text quote the short readable text. For ambiguous_sequence explain
the uncertainty (for example, the creature could be a separate arrival).
Both panel references must occur in panels. They may be the same panel if the
change happens within one panel. Do not invent off-page endpoints.
Return events: [] if no event is supported. Do not fabricate an event to fill
the schema. Similar looks, a different facial expression, or two neighboring
panels alone do not demonstrate transformation. Use local visual descriptions,
not guessed names or canonical character IDs. These events are only proposals.

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


def describe_page(
    page_path: Path,
    processor,
    model,
    *,
    expected_panels=None,
) -> dict:
    """Return descriptions and proposed events from the same generation."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(page_path.resolve())},
                {"type": "text", "text": PAGE_PROMPT + (
                    "\nExpected panel numbers: " + json.dumps(expected_panels) if expected_panels is not None else "")},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=config.DESCRIPTION_ENABLE_THINKING,
    )
    inputs = move_to_device(inputs, model.device)

    torch.cuda.synchronize()
    start = time.perf_counter()

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
    elapsed = time.perf_counter() - start

    input_length = int(inputs["input_ids"].shape[-1])
    generated = output_ids[0, input_length:]
    generated_tokens = int(generated.shape[-1])
    response = processor.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    result = parse_page_response(
        response, truncated=generated_tokens >= config.DESCRIPTION_MAX_NEW_TOKENS,
        expected_panels=expected_panels,
    )
    result.update(
        generation_time=elapsed, generated_tokens=generated_tokens, raw_response=response
    )
    return result


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
    for index, page_path in enumerate(pages, start=1):
        print(f"  Page {index}/{len(pages)}: {page_path.name}", flush=True)
        source = metadata.get(page_path.name, {})
        if source:
            source_path = Path(source["page"])
            if (not source_path.is_file() or file_hash(source_path) != source.get("source_hash")
                    or (source.get("preview_hash") and file_hash(page_path) != source["preview_hash"])):
                raise ValueError(f"Stale page preview/manifest: {page_path.name}; rerun test_panels.py.")
            page = describe_page(page_path, processor, model,
                                 expected_panels=[p["panel_number"] for p in source["panels"]])
        else:
            page = describe_page(page_path, processor, model)
        print(f"    Generate: {page['generation_time']:.2f}s", flush=True)
        print(
            f"    Tokens:   {page['generated_tokens']}/{config.DESCRIPTION_MAX_NEW_TOKENS}",
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

    return {"descriptions": results, "transformations": transformations}


def describe_manga(
    manga_output_dir: Path,
    chapter_dirs: list[Path],
) -> dict:
    """Return both manga-wide results without writing JSON or other artifacts."""
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
    for chapter_dir in chapter_dirs:
        chapter_output_dir = manga_output_dir / chapter_dir.name
        print(f"Chapter: {chapter_dir.name}")
        chapter = describe_chapter(chapter_output_dir, processor, model)
        all_results[chapter_dir.name] = chapter["descriptions"]
        transformations[chapter_dir.name] = chapter["transformations"]

    return {
        "descriptions": all_results,
        "transformations": {
            "schema_version": 1,
            "scope": "within_page",
            "chapters": transformations,
        },
    }
