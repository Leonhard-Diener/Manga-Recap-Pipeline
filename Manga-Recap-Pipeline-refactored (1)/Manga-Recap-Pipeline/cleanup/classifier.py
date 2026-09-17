"""Step 2: remove non-story pages (credits, shoutouts, intros, ...) from a
downloaded chapter.

A small vision-language model classifies each page as MANGA or EXTRA. Each
chapter is scanned inward from both ends until the first MANGA page is found
on each side; everything outside that range is deleted. Pages in the middle
of a chapter are never touched, matching how these extras actually appear
(only at the very start/end of a chapter).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

import config
from common.utils import list_images, move_to_device

CLASSIFICATION_PROMPT = """
You are classifying pages of a manga chapter.

Return exactly one label:

MANGA
EXTRA

MANGA = content that belongs to the actual manga reading experience.

This includes:
- story pages
- normal manga panels
- splash pages showing a story scene
- pages with characters interacting or performing an action
- dialogue or narration
- chapter title pages
- covers and official color pages
- official bonus pages containing meaningful manga/story content

EXTRA = material outside the actual manga reading content.

This includes:
- translator shoutouts
- translator messages
- scanlation group messages
- translation credits
- scanlation credits
- uploader messages
- thank-you pages
- release information
- website or social-media promotion
- donation or recruitment messages
- advertisements
- character sheets or profiles
- standalone character illustrations
- character group illustrations with no story event
- promotional artwork without meaningful story content
- decorative pages with no narrative meaning

IMPORTANT:
A translator or scanlation page is ALWAYS EXTRA.

A page can contain manga characters and still be EXTRA.

A page can be officially published by the manga author and still be EXTRA.

A standalone character illustration with no meaningful story is EXTRA.

A page showing characters actually interacting, performing an action,
speaking, or participating in a meaningful scene is MANGA.

The goal is to identify the first and last pages that belong to the
meaningful manga reading content.

Reply with exactly one word:

MANGA
or
EXTRA
"""

_LABEL_PATTERN = re.compile(r"\b(MANGA|EXTRA)\b")


def load_classifier():
    """Load the page-classification model and processor."""
    processor = AutoProcessor.from_pretrained(
        config.QWEN_MODEL_PATH,
        min_pixels=config.CLASSIFIER_MIN_PIXELS,
        max_pixels=config.CLASSIFIER_MAX_PIXELS,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        config.QWEN_MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
    ).eval()
    return processor, model


def classify_page(image_path: Path, processor, model) -> str:
    """Classify one page as MANGA, EXTRA, or UNKNOWN (unparsable output)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": CLASSIFICATION_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_to_device(inputs, model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.CLEANUP_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    generated = output_ids[0, inputs["input_ids"].shape[-1]:]
    raw_output = processor.decode(generated, skip_special_tokens=True).strip()

    match = _LABEL_PATTERN.search(raw_output.upper())
    return match.group(1) if match else "UNKNOWN"


def _scan_from_start(pages: list[Path], processor, model) -> int:
    """Return the index of the first MANGA page, scanning from the start."""
    for index, page in enumerate(pages):
        print(f"  [start {index + 1}/{len(pages)}] {page.name} ...", end=" ", flush=True)
        label = classify_page(page, processor, model)
        print(label)
        if label == "MANGA":
            return index
    raise RuntimeError("No MANGA page found when scanning from the start.")


def _scan_from_end(pages: list[Path], processor, model, first_index: int) -> int:
    """Return the index of the last MANGA page, scanning from the end."""
    for index in range(len(pages) - 1, first_index - 1, -1):
        page = pages[index]
        print(f"  [end {len(pages) - index}/{len(pages)}] {page.name} ...", end=" ", flush=True)
        label = classify_page(page, processor, model)
        print(label)
        if label == "MANGA":
            return index
    raise RuntimeError("No MANGA page found when scanning from the end.")


def clean_chapter(chapter_dir: Path, processor, model) -> dict:
    """Delete the extra pages at the start/end of one chapter.

    Returns a small stats dict with the number of pages removed and kept.
    """
    pages = list_images(chapter_dir)
    if not pages:
        print(f"  No images in {chapter_dir}, skipping.")
        return {"removed": 0, "kept": 0}

    print(f"Chapter: {chapter_dir.name} ({len(pages)} pages)")

    first_index = _scan_from_start(pages, processor, model)
    last_index = _scan_from_end(pages, processor, model, first_index)

    extra_pages = pages[:first_index] + pages[last_index + 1:]
    for page in extra_pages:
        page.unlink()

    kept = last_index - first_index + 1
    print(
        f"  -> kept {pages[first_index].name}..{pages[last_index].name} "
        f"({kept} pages), removed {len(extra_pages)} extra pages."
    )
    return {"removed": len(extra_pages), "kept": kept}


def clean_manga(chapter_dirs: list[Path]) -> None:
    """Load the classification model once and clean every chapter."""
    print("Loading page-classification model...")
    processor, model = load_classifier()
    print("Model loaded.\n")

    for chapter_dir in chapter_dirs:
        clean_chapter(chapter_dir, processor, model)
