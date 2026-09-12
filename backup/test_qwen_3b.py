
import re
import time
from pathlib import Path

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "./models/Qwen2.5-VL-3B-Instruct"
INPUT_ROOT = "./input"

MANGA_NAME = (
    "Maou-sama no Machizukuri! ~Saikyou no Dungeon wa Kindai Toshi~"
)

# The model only needs to generate a very short classification.
MAX_NEW_TOKENS = 4

# Lower visual token counts improve inference speed.
# Qwen recommends controlling image resolution with these values.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

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
Classify this manga page as exactly one label:

MANGA
EXTRA

MANGA = meaningful manga reading content, including:
- story pages
- normal panels
- splash pages
- story scenes with little or no text
- chapter title pages
- covers and official color pages
- official bonus pages containing meaningful manga/story content

EXTRA = non-story material, including:
- translator or scanlation messages
- credits
- thank-you pages
- advertisements
- release information
- character sheets or profiles
- standalone character illustrations
- character group artwork with no story event or narrative
- promotional artwork without meaningful story content

IMPORTANT:
A page can be officially published by the manga author and still be EXTRA.
A purely decorative character illustration is EXTRA.
A page showing characters actually interacting or participating in a story scene is MANGA.

Reply with exactly one word:
MANGA
or
EXTRA
"""


# ============================================================
# Natural sorting
# ============================================================

def natural_sort_key(path: Path):
    """Sort filenames/directories numerically."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


# ============================================================
# Find chapters
# ============================================================

def get_chapter_directories(manga_directory: Path) -> list[Path]:
    """Return all chapter directories in natural order."""

    if not manga_directory.exists():
        raise FileNotFoundError(
            f"Manga directory does not exist: {manga_directory}"
        )

    chapters = [
        directory
        for directory in manga_directory.iterdir()
        if directory.is_dir()
    ]

    return sorted(chapters, key=natural_sort_key)


# ============================================================
# Find pages
# ============================================================

def get_image_files(chapter_directory: Path) -> list[Path]:
    """Return all supported image files in natural order."""

    files = [
        file
        for file in chapter_directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files, key=natural_sort_key)


# ============================================================
# Parse model output
# ============================================================

def parse_classification(raw_output: str) -> str:
    """
    Extract MANGA or EXTRA from the model output.

    The model occasionally returns additional text despite the prompt.
    Matching the actual label anywhere in the output is more robust than
    requiring the response to start with the label.
    """

    match = re.search(r"\b(MANGA|EXTRA)\b", raw_output.upper())

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
        (classification, inference_time_seconds, raw_model_output)
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path.resolve()),
                },
                {
                    "type": "text",
                    "text": CLASSIFICATION_PROMPT,
                },
            ],
        }
    ]

    # --------------------------------------------------------
    # Prepare text and image input
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    # --------------------------------------------------------
    # Measure actual generation time
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Decode generated tokens
    # --------------------------------------------------------

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    raw_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # --------------------------------------------------------
    # Parse classification
    # --------------------------------------------------------

    classification = parse_classification(raw_output)

    # Show the exact model output for debugging.
    print(
        f"    Raw Qwen output: {raw_output!r}",
        flush=True,
    )

    print(
        f"    Parsed result:   {classification}",
        flush=True,
    )

    return classification, elapsed, raw_output


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
            f"[START {index + 1}/{len(pages)}] {page.name}",
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
        "No MANGA page found when scanning from the beginning."
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
            f"[END {len(pages) - index}/{len(pages)}] {page.name}",
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
        "No MANGA page found when scanning from the end."
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

    pages = get_image_files(chapter_directory)

    if not pages:
        print("No images found. Skipping chapter.")
        return None

    print("\n" + "=" * 60)
    print(f"CHAPTER: {chapter_directory.name}")
    print(f"Pages: {len(pages)}")
    print("=" * 60)

    # --------------------------------------------------------
    # Scan from beginning
    # --------------------------------------------------------

    first_manga_index, start_times = find_first_manga_page(
        pages,
        model,
        processor,
    )

    # --------------------------------------------------------
    # Scan from end
    # --------------------------------------------------------

    last_manga_index, end_times = find_last_manga_page(
        pages,
        model,
        processor,
        first_manga_index,
    )

    # --------------------------------------------------------
    # Calculate results
    # --------------------------------------------------------

    first_manga_page = pages[first_manga_index]
    last_manga_page = pages[last_manga_index]

    all_times = start_times + end_times
    average_time = sum(all_times) / len(all_times)

    print("\n" + "-" * 60)
    print("CHAPTER RESULT")
    print("-" * 60)

    print(
        f"First manga page:   {first_manga_page.name}"
    )

    print(
        f"Last manga page:    {last_manga_page.name}"
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
    print("QWEN MANGA PAGE SCANNER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Qwen...", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
    )

    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("Model loaded successfully.", flush=True)

    # --------------------------------------------------------
    # Find manga directory
    # --------------------------------------------------------

    manga_directory = Path(INPUT_ROOT) / MANGA_NAME

    chapters = get_chapter_directories(
        manga_directory
    )

    if not chapters:
        print("No chapter directories found.")
        return

    print(
        f"\nFound {len(chapters)} chapters.",
        flush=True,
    )

    # --------------------------------------------------------
    # Scan chapters
    # --------------------------------------------------------

    results = {}

    for chapter_index, chapter_directory in enumerate(
        chapters,
        start=1,
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
            results[chapter_directory.name] = result

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)

    for chapter_name, result in results.items():
        print(
            f"{chapter_name}: "
            f"{result['first_manga_page'].name} -> "
            f"{result['last_manga_page'].name} "
            f"({result['average_time']:.2f} s/page)"
        )


if __name__ == "__main__":
    main()
