
import re
import time
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "./models/Qwen3.5-2B"
INPUT_ROOT = "./input"

MANGA_NAME = (
    "Maou-sama no Machizukuri! ~Saikyou no Dungeon wa Kindai Toshi~"
)

# The model only needs a very short classification.
MAX_NEW_TOKENS = 4

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# Classification prompt
# ============================================================

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


# ============================================================
# Natural sorting
# ============================================================

def natural_sort_key(path: Path):
    """Sort filenames and directories numerically."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(
            r"(\d+)",
            path.name,
        )
    ]


# ============================================================
# Find chapters
# ============================================================

def get_chapter_directories(
    manga_directory: Path,
) -> list[Path]:
    """Return all chapter directories in natural order."""

    if not manga_directory.exists():
        raise FileNotFoundError(
            f"Manga directory does not exist: "
            f"{manga_directory}"
        )

    chapters = [
        directory
        for directory in manga_directory.iterdir()
        if directory.is_dir()
    ]

    return sorted(
        chapters,
        key=natural_sort_key,
    )


# ============================================================
# Find pages
# ============================================================

def get_image_files(
    chapter_directory: Path,
) -> list[Path]:
    """Return all supported image files in natural order."""

    files = [
        file
        for file in chapter_directory.iterdir()
        if (
            file.is_file()
            and file.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(
        files,
        key=natural_sort_key,
    )


# ============================================================
# Parse model output
# ============================================================

def parse_classification(
    raw_output: str,
) -> str:
    """
    Extract MANGA or EXTRA from the model output.

    Additional generated text is tolerated.
    """

    match = re.search(
        r"\b(MANGA|EXTRA)\b",
        raw_output.upper(),
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


# ============================================================
# Classify one page
# ============================================================

def classify_page(
    image_path: Path,
    model,
    processor,
) -> tuple[str, float, str]:
    """
    Classify one page as MANGA or EXTRA.

    Returns:
        (classification, inference_time_seconds, raw_output)
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(
                        image_path.resolve()
                    ),
                },
                {
                    "type": "text",
                    "text": CLASSIFICATION_PROMPT,
                },
            ],
        }
    ]

    # --------------------------------------------------------
    # Build native Qwen3.5 multimodal input.
    # --------------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Move tensors to model device.
    # Floating-point tensors use FP16 on the GTX 1080.
    # --------------------------------------------------------

    for key, value in inputs.items():

        if not torch.is_tensor(value):
            continue

        if value.is_floating_point():
            inputs[key] = value.to(
                device=model.device,
                dtype=torch.float16,
            )
        else:
            inputs[key] = value.to(
                device=model.device,
            )

    # --------------------------------------------------------
    # Measure actual generation time.
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Decode only newly generated tokens.
    # --------------------------------------------------------

    input_length = (
        inputs["input_ids"].shape[-1]
    )

    generated_ids = output_ids[
        0,
        input_length:
    ]

    raw_output = processor.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    # --------------------------------------------------------
    # Parse classification.
    # --------------------------------------------------------

    classification = parse_classification(
        raw_output
    )

    print(
        f"    Raw Qwen3.5 output: "
        f"{raw_output!r}",
        flush=True,
    )

    print(
        f"    Parsed result:      "
        f"{classification}",
        flush=True,
    )

    return (
        classification,
        elapsed,
        raw_output,
    )


# ============================================================
# Find first manga page
# ============================================================

def find_first_manga_page(
    pages: list[Path],
    model,
    processor,
) -> tuple[int, list[float]]:
    """Scan from the beginning until the first MANGA page."""

    print("\nScanning from beginning...")

    times = []

    for index, page in enumerate(pages):

        print(
            f"[START {index + 1}/{len(pages)}] "
            f"{page.name}",
            flush=True,
        )

        result, elapsed, _ = classify_page(
            page,
            model,
            processor,
        )

        times.append(elapsed)

        print(
            f"    -> {result}",
            flush=True,
        )

        print(
            f"    Time: {elapsed:.2f} s",
            flush=True,
        )

        if result == "MANGA":
            return index, times

    raise RuntimeError(
        "No MANGA page found when scanning "
        "from the beginning."
    )


# ============================================================
# Find last manga page
# ============================================================

def find_last_manga_page(
    pages: list[Path],
    model,
    processor,
    first_manga_index: int,
) -> tuple[int, list[float]]:
    """Scan backwards until the first MANGA page."""

    print("\nScanning from end...")

    times = []

    for index in range(
        len(pages) - 1,
        first_manga_index - 1,
        -1,
    ):

        page = pages[index]

        print(
            f"[END {len(pages) - index}/{len(pages)}] "
            f"{page.name}",
            flush=True,
        )

        result, elapsed, _ = classify_page(
            page,
            model,
            processor,
        )

        times.append(elapsed)

        print(
            f"    -> {result}",
            flush=True,
        )

        print(
            f"    Time: {elapsed:.2f} s",
            flush=True,
        )

        if result == "MANGA":
            return index, times

    raise RuntimeError(
        "No MANGA page found when scanning "
        "from the end."
    )


# ============================================================
# Scan one chapter
# ============================================================

def scan_chapter(
    chapter_directory: Path,
    model,
    processor,
):
    """Find the first and last actual manga page."""

    pages = get_image_files(
        chapter_directory
    )

    if not pages:
        print(
            "No images found. Skipping chapter."
        )
        return None

    print("\n" + "=" * 60)
    print(
        f"CHAPTER: {chapter_directory.name}"
    )
    print(
        f"Pages: {len(pages)}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Scan from beginning.
    # --------------------------------------------------------

    first_manga_index, start_times = (
        find_first_manga_page(
            pages,
            model,
            processor,
        )
    )

    # --------------------------------------------------------
    # Scan from end.
    # --------------------------------------------------------

    last_manga_index, end_times = (
        find_last_manga_page(
            pages,
            model,
            processor,
            first_manga_index,
        )
    )

    # --------------------------------------------------------
    # Calculate results.
    # --------------------------------------------------------

    first_manga_page = pages[
        first_manga_index
    ]

    last_manga_page = pages[
        last_manga_index
    ]

    all_times = (
        start_times + end_times
    )

    average_time = (
        sum(all_times) / len(all_times)
    )

    print("\n" + "-" * 60)
    print("CHAPTER RESULT")
    print("-" * 60)

    print(
        f"First manga page:   "
        f"{first_manga_page.name}"
    )

    print(
        f"Last manga page:    "
        f"{last_manga_page.name}"
    )

    print(
        f"Manga pages:        "
        f"{last_manga_index - first_manga_index + 1}"
    )

    print(
        f"Extra pages before: "
        f"{first_manga_index}"
    )

    print(
        f"Extra pages after:  "
        f"{len(pages) - last_manga_index - 1}"
    )

    print(
        f"Average inference:  "
        f"{average_time:.2f} s/page"
    )

    return {
        "first_manga_page": first_manga_page,
        "last_manga_page": last_manga_page,
        "first_index": first_manga_index,
        "last_index": last_manga_index,
        "average_time": average_time,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("QWEN3.5-2B MANGA PAGE SCANNER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model.
    # --------------------------------------------------------

    print(
        "\nLoading Qwen3.5-2B...",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
    ).eval()

    print(
        "Model loaded successfully.",
        flush=True,
    )

    print(
        f"Model device: {model.device}",
        flush=True,
    )

    print(
        f"Model dtype: "
        f"{next(model.parameters()).dtype}",
        flush=True,
    )

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    # --------------------------------------------------------
    # Find manga directory.
    # --------------------------------------------------------

    manga_directory = (
        Path(INPUT_ROOT) / MANGA_NAME
    )

    chapters = get_chapter_directories(
        manga_directory
    )

    if not chapters:
        print(
            "No chapter directories found."
        )
        return

    print(
        f"\nFound {len(chapters)} chapters.",
        flush=True,
    )

    # --------------------------------------------------------
    # Scan chapters.
    # --------------------------------------------------------

    results = {}

    for chapter_index, chapter_directory in (
        enumerate(chapters, start=1)
    ):

        print(
            f"\nProcessing chapter "
            f"{chapter_index}/{len(chapters)}",
            flush=True,
        )

        result = scan_chapter(
            chapter_directory,
            model,
            processor,
        )

        if result is not None:
            results[
                chapter_directory.name
            ] = result

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)

    for chapter_name, result in (
        results.items()
    ):

        print(
            f"{chapter_name}: "
            f"{result['first_manga_page'].name} -> "
            f"{result['last_manga_page'].name} "
            f"({result['average_time']:.2f} s/page)"
        )


if __name__ == "__main__":
    main()
