"""Step 4: describe what happens on each page, panel by panel.

One complete manga page (the numbered preview from step 3) is analyzed in a
single inference. The model replies with plain text:

    PANEL 1: ...
    PANEL 2: ...

which is then parsed into JSON. No detections.json is required or produced
by this step.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

import config
from common.utils import list_images, move_to_device

PAGE_PROMPT = """\
Analyze the image visually.

For each panel, output:
PANEL N:
Characters: ...
Location: ...
Actions: ...

Rules:
- Describe every visible character or creature.
- Describe their visible appearance briefly.
- Describe where the scene takes place.
- Describe exactly what each character is visibly doing.
- Describe important visible objects when relevant.
- Use only information directly visible in the image.
- Do not guess names, identities, relationships, intentions, or story context.
- Do not invent events.
- Do not infer what happened before or after the scene.
- Only quote text that is clearly readable.
- Keep each field concise.
- No reasoning.
- No JSON.
"""

_PANEL_HEADER = re.compile(
    r"^\s*(?:[*_`#]+\s*)?PANEL\s+(\d+)\s*:?[ \t]*(?:[*_`#]+[ \t]*)?(.*)$",
    re.IGNORECASE,
)


def load_describer():
    """Load the page-description model, processor, and run a warmup pass."""
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


def parse_response(response: str) -> dict[int, str]:
    """Extract PANEL N descriptions from one-line or multiline model output."""
    descriptions: dict[int, str] = {}
    current_panel: int | None = None
    current_lines: list[str] = []

    def save_current() -> None:
        if current_panel is None:
            return
        description = " ".join(" ".join(current_lines).split()).strip()
        if description:
            descriptions[current_panel] = description

    for line in response.splitlines():
        match = _PANEL_HEADER.match(line)
        if match:
            save_current()
            current_panel = int(match.group(1))
            current_lines = [match.group(2)] if match.group(2).strip() else []
        elif current_panel is not None:
            current_lines.append(line)

    save_current()
    return descriptions


def describe_page(
    page_path: Path,
    processor,
    model,
) -> tuple[dict[int, str], float, int, str]:
    """Describe one page and return descriptions, time, token count, and raw output."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(page_path.resolve())},
                {"type": "text", "text": PAGE_PROMPT},
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

    return parse_response(response), elapsed, generated_tokens, response


def describe_chapter(chapter_output_dir: Path, processor, model) -> dict[str, dict]:
    """Describe every numbered page preview of one chapter (step 3's output)."""
    preview_dir = chapter_output_dir / "page_previews"
    pages = list_images(preview_dir)

    results: dict[str, list[dict]] = {}
    for index, page_path in enumerate(pages, start=1):
        print(f"  Page {index}/{len(pages)}: {page_path.name}", flush=True)
        descriptions, elapsed, generated_tokens, raw_response = describe_page(
            page_path, processor, model
        )
        print(f"    Generate: {elapsed:.2f}s", flush=True)
        print(
            f"    Tokens:   {generated_tokens}/{config.DESCRIPTION_MAX_NEW_TOKENS}",
            flush=True,
        )
        print(f"    Panels:   {len(descriptions)}", flush=True)

        for panel_number in sorted(descriptions):
            print(
                f"    Panel {panel_number}: {descriptions[panel_number]}",
                flush=True,
            )

        if not descriptions:
            print(f"    Raw output: {raw_response!r}", flush=True)

        results[page_path.name] = {
            "panels": [
                {"panel_number": number, "description": descriptions[number]}
                for number in sorted(descriptions)
            ],
            "generation_time": elapsed,
            "generated_tokens": generated_tokens,
        }

    return results


def describe_manga(
    manga_output_dir: Path,
    chapter_dirs: list[Path],
    output_file: Path,
) -> None:
    """Load the description model once and describe every chapter."""
    print("Loading page-description model...")
    processor, model = load_describer()
    print("Model loaded.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {model.device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print("Warmup complete.\n")

    all_results: dict[str, dict] = {}
    for chapter_dir in chapter_dirs:
        chapter_output_dir = manga_output_dir / chapter_dir.name
        print(f"Chapter: {chapter_dir.name}")
        all_results[chapter_dir.name] = describe_chapter(chapter_output_dir, processor, model)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved descriptions to {output_file}")
