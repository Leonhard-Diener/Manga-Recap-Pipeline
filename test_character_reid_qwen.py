"""Merge CCIP visual-form clusters into character families with Qwen.

CCIP is intentionally allowed to split a person into visually distinct forms.
Qwen only reviews the nearest *form* pairs, using several representative
images from each form.  This keeps an uncertain VLM answer from becoming the
initial identity signal.

The result has two layers:
  * form_id: a conservative CCIP visual cluster;
  * character_id: one or more forms Qwen judged to be the same person.

Characters with three or fewer total appearances are retained in the raw
audit data but omitted from the final named-character output.

Usage (run CCIP first for every chapter):
    python test_character_reid_ccip.py "Manga title"
    python test_character_reid_qwen.py "Manga title"
"""

from __future__ import annotations

import json
import re
import sys
import time
import gc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor

import config
from common.utils import find_chapter_dirs, find_manga_dir
from test_character_reid import sanitize_filename
from test_character_reid_ccip import extract_features, load_sessions


CCIP_FILE_NAME = "character_classification_ccip.json"
OUTPUT_FILE_NAME = "character_classification_qwen.json"
MANGA_OUTPUT_FILE_NAME = "character_classification_qwen.json"
QWEN_MODEL_DIR = Path("./models/Qwen3.5-2B")
DEBUG_ROOT = Path("./debug/character_reid_qwen")

# Keep only characters that appear at least four times in the entire manga.
# The raw output still records filtered characters and all Qwen decisions.
MIN_CHARACTER_APPEARANCES = 4
REPRESENTATIVES_PER_FORM = 4
NEAREST_FORM_CANDIDATES = 3
QWEN_MERGE_CONFIDENCE = 0.78
# Keep this verification pass fast and deterministic, like test_description.
# It receives several representative crops, so the prompt carries the context
# rather than asking this small model for a long reasoning trace.
QWEN_ENABLE_THINKING = False
QWEN_MAX_NEW_TOKENS = 64


@dataclass
class Appearance:
    chapter: str
    page: str
    character_index: int
    box: list[float]
    detector_score: float
    crop_path: Path
    ccip_identity: str


@dataclass
class Form:
    form_id: str
    chapter: str
    ccip_identity: str
    appearances: list[Appearance]


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _read_ccip_forms(manga_dir: Path, chapters: list[Path]) -> list[Form]:
    forms: list[Form] = []
    for chapter in chapters:
        result_file = config.OUTPUT_DIR / manga_dir.name / chapter.name / CCIP_FILE_NAME
        if not result_file.is_file():
            raise FileNotFoundError(
                f"Missing {result_file}. Run test_character_reid_ccip.py for every chapter first."
            )
        data = json.loads(result_file.read_text(encoding="utf-8"))
        grouped: dict[str, list[Appearance]] = defaultdict(list)
        for item in data.get("characters", []):
            if not isinstance(item, dict) or not item.get("identity"):
                continue
            page = Path(str(item.get("page", "")))
            if not page.is_file():
                continue
            crop_path = Path(str(item.get("person_detection_crop") or ""))
            grouped[str(item["identity"])].append(
                Appearance(
                    chapter=chapter.name,
                    page=str(page),
                    character_index=int(item.get("character_index", 0)),
                    box=[float(value) for value in item.get("box", [])],
                    detector_score=float(item.get("detector_score", 0.0)),
                    crop_path=crop_path,
                    ccip_identity=str(item["identity"]),
                )
            )
        for identity, appearances in grouped.items():
            forms.append(Form(
                form_id=f"{chapter.name}:{identity}", chapter=chapter.name,
                ccip_identity=identity, appearances=appearances,
            ))
    return forms


def _crop(appearance: Appearance) -> Image.Image:
    """Use the original page, ensuring the verifier sees the exact CCIP crop."""
    with Image.open(appearance.page) as source:
        image = source.convert("RGB")
        x1, y1, x2, y2 = (int(round(value)) for value in appearance.box)
        box = (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid crop for {appearance.page} #{appearance.character_index}")
        return image.crop(box)


def _representatives(form: Form) -> list[Appearance]:
    """Choose sharp, large, page-diverse evidence rather than random examples."""
    def rank(item: Appearance) -> tuple[float, float]:
        width = max(0.0, item.box[2] - item.box[0])
        height = max(0.0, item.box[3] - item.box[1])
        return (width * height, item.detector_score)

    ordered = sorted(form.appearances, key=rank, reverse=True)
    selected: list[Appearance] = []
    seen_pages: set[str] = set()
    for item in ordered:
        if item.page not in seen_pages:
            selected.append(item)
            seen_pages.add(item.page)
        if len(selected) == REPRESENTATIVES_PER_FORM:
            return selected
    for item in ordered:
        if item not in selected:
            selected.append(item)
        if len(selected) == REPRESENTATIVES_PER_FORM:
            break
    return selected


def _make_comparison_sheet(left: Form, right: Form) -> Image.Image:
    """Render two labelled groups; labels are forms, never assumed names."""
    tile, header, margin = 256, 32, 8
    left_items, right_items = _representatives(left), _representatives(right)
    rows = max(len(left_items), len(right_items), 1)
    canvas = Image.new("RGB", (tile * 2 + margin * 3, header + rows * (tile + margin) + margin), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((margin, 8), f"FORM A: {left.form_id}", fill="black", font=font)
    draw.text((tile + margin * 2, 8), f"FORM B: {right.form_id}", fill="black", font=font)
    for column, items in enumerate((left_items, right_items)):
        for row, item in enumerate(items):
            crop = ImageOps.pad(_crop(item), (tile, tile), color=(180, 180, 180), method=Image.Resampling.LANCZOS)
            x, y = margin + column * (tile + margin), header + margin + row * (tile + margin)
            canvas.paste(crop, (x, y))
    return canvas


def _load_qwen() -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    if not QWEN_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Qwen model not found: {QWEN_MODEL_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen verification requires CUDA; no CUDA device is available.")
    print("Loading Qwen3.5-2B on CUDA...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_DIR, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN_MODEL_DIR, local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={"": 0},
    ).eval()
    return processor, model


def _parse_verdict(text: str) -> tuple[str, float, str]:
    """Read the final JSON after Qwen's optional <think> reasoning trace."""
    payload: dict = {}
    # Thinking may itself quote JSON. The final valid JSON object is the
    # answer, so walk all candidates from right to left.
    for match in reversed(list(re.finditer(r"\{.*?\}", text, re.DOTALL))):
        try:
            payload = json.loads(match.group(0))
            break
        except json.JSONDecodeError:
            continue
    verdict = str(payload.get("verdict", "unsure")).lower()
    if verdict not in {"same_character", "different_character", "unsure"}:
        verdict = "unsure"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return verdict, min(1.0, max(0.0, confidence)), str(payload.get("reason", ""))[:300]


def _verify_pair(processor: AutoProcessor, model: AutoModelForImageTextToText, left: Form, right: Form) -> tuple[str, float, str, Image.Image]:
    sheet = _make_comparison_sheet(left, right)
    prompt = (
        "You are a conservative manga character identity verifier. Compare FORM A and FORM B. "
        "They may show the same fictional character in different poses, expressions, outfits, or a supernatural transformation. "
        "They may also be different characters with similar manga art. Do not merge based only on art style or gender. "
        "Return exactly one JSON object with no markdown: "
        '{"verdict":"same_character|different_character|unsure","confidence":0.0,"reason":"short visual reason"}. '
        "Use unsure when the visual evidence is insufficient."
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": sheet}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=QWEN_ENABLE_THINKING,
    )
    inputs = processor(text=[rendered], images=[sheet], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    verdict, confidence, reason = _parse_verdict(response)
    return verdict, confidence, reason, sheet


def _candidate_pairs(forms: list[Form]) -> list[tuple[Form, Form]]:
    """Review only the nearest CCIP form pairs, never every possible pair."""
    feature_session, metric_session, _ = load_sessions()
    representative_forms: list[Form] = []
    representative_images: list[Image.Image] = []
    for form in forms:
        for appearance in _representatives(form):
            representative_forms.append(form)
            representative_images.append(_crop(appearance))
    features = extract_features(feature_session, representative_images)
    differences = np.asarray(metric_session.run(None, {"input": features})[0], dtype=np.float32)
    # The Qwen model nearly fills an 8 GB GPU. Release CCIP before Qwen loads.
    del features, feature_session, metric_session
    gc.collect()
    distances: dict[tuple[str, str], list[float]] = defaultdict(list)
    for first, left in enumerate(representative_forms):
        for second in range(first + 1, len(representative_forms)):
            right = representative_forms[second]
            if left.form_id != right.form_id:
                distances[tuple(sorted((left.form_id, right.form_id)))].append(float(differences[first, second]))
    # One compatible view is sufficient to ask Qwen; Qwen then sees all
    # selected evidence and is free to reject visually similar characters.
    form_distance = {key: min(values) for key, values in distances.items()}
    pairs: dict[tuple[str, str], tuple[Form, Form]] = {}
    for left in forms:
        candidates = sorted(
            ((form_distance[tuple(sorted((left.form_id, right.form_id)))], right)
             for right in forms if right.form_id != left.form_id),
            key=lambda item: item[0],
        )[:NEAREST_FORM_CANDIDATES]
        for _, right in candidates:
            key = tuple(sorted((left.form_id, right.form_id)))
            pairs[key] = (left, right)
    return [pairs[key] for key in sorted(pairs)]


def process_manga(manga_dir: Path, chapters: list[Path]) -> dict:
    forms = _read_ccip_forms(manga_dir, chapters)
    if not forms:
        return {"manga": manga_dir.name, "forms": [], "characters": []}
    pairs = _candidate_pairs(forms)
    # ONNX Runtime can retain a CUDA allocation briefly after its sessions are
    # collected. Synchronising here makes the Qwen load reliable on 8 GB GPUs.
    gc.collect()
    torch.cuda.empty_cache()
    processor, model = _load_qwen()
    debug_dir = DEBUG_ROOT / sanitize_filename(manga_dir.name)
    debug_dir.mkdir(parents=True, exist_ok=True)
    decisions: list[dict] = []
    union_find = UnionFind([form.form_id for form in forms])
    print(f"Qwen candidate form comparisons: {len(pairs)}")
    for number, (left, right) in enumerate(pairs, start=1):
        start = time.perf_counter()
        verdict, confidence, reason, sheet = _verify_pair(processor, model, left, right)
        accepted = verdict == "same_character" and confidence >= QWEN_MERGE_CONFIDENCE
        if accepted:
            union_find.union(left.form_id, right.form_id)
        sheet_path = debug_dir / f"{number:03d}_{sanitize_filename(left.form_id)}__{sanitize_filename(right.form_id)}_{verdict}.jpg"
        sheet.save(sheet_path, quality=95)
        decisions.append({"form_a": left.form_id, "form_b": right.form_id, "verdict": verdict,
                          "confidence": round(confidence, 3), "accepted_merge": accepted,
                          "reason": reason, "comparison_sheet": str(sheet_path),
                          "seconds": round(time.perf_counter() - start, 2)})
        print(f"  {number}/{len(pairs)} {left.form_id} vs {right.form_id}: {verdict} ({confidence:.2f})")

    families: dict[str, list[Form]] = defaultdict(list)
    for form in forms:
        families[union_find.find(form.form_id)].append(form)
    ordered_families = sorted(families.values(), key=lambda group: (-sum(len(form.appearances) for form in group), min(form.form_id for form in group)))
    form_to_character: dict[str, tuple[str, int, bool]] = {}
    character_records: list[dict] = []
    for index, family in enumerate(ordered_families, start=1):
        appearances = sum(len(form.appearances) for form in family)
        character_id = f"character_{index:03d}"
        included = appearances >= MIN_CHARACTER_APPEARANCES
        for form in family:
            form_to_character[form.form_id] = (character_id, appearances, included)
        character_records.append({"character_id": character_id, "appearance_count": appearances,
                                  "form_ids": [form.form_id for form in family], "included_in_final": included})

    per_chapter: dict[str, list[dict]] = defaultdict(list)
    for form in forms:
        character_id, family_count, included = form_to_character[form.form_id]
        for appearance in form.appearances:
            per_chapter[appearance.chapter].append({"page": appearance.page, "character_index": appearance.character_index,
                "box": [round(value, 2) for value in appearance.box], "detector_score": round(appearance.detector_score, 4),
                "form_id": form.form_id, "character_id": character_id if included else None,
                "appearance_count_in_manga": family_count, "included_in_final": included})
    result = {"manga": manga_dir.name, "primary_form_model": "CCIP", "qwen_model": str(QWEN_MODEL_DIR),
              "min_character_appearances": MIN_CHARACTER_APPEARANCES, "qwen_merge_confidence": QWEN_MERGE_CONFIDENCE,
              "forms": [{"form_id": form.form_id, "appearance_count": len(form.appearances)} for form in forms],
              "merge_decisions": decisions, "characters": character_records,
              "final_named_characters": [record for record in character_records if record["included_in_final"]]}
    manga_output = config.OUTPUT_DIR / manga_dir.name / MANGA_OUTPUT_FILE_NAME
    manga_output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for chapter in chapters:
        chapter_result = dict(result)
        chapter_result["chapter"] = chapter.name
        chapter_result["appearances"] = sorted(per_chapter[chapter.name], key=lambda item: (Path(item["page"]).name, item["character_index"]))
        output_file = config.OUTPUT_DIR / manga_dir.name / chapter.name / OUTPUT_FILE_NAME
        output_file.write_text(json.dumps(chapter_result, indent=2, ensure_ascii=False), encoding="utf-8")
    kept = sum(record["included_in_final"] for record in character_records)
    print(f"Saved {kept} final characters; filtered {len(character_records) - kept} identities with <= {MIN_CHARACTER_APPEARANCES - 1} appearances.")
    return result


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else (config.MANGA_TITLE or None)
    requested_chapter = sys.argv[2] if len(sys.argv) > 2 else None
    manga_dir = find_manga_dir(config.INPUT_DIR, title)
    chapters = find_chapter_dirs(manga_dir)
    if requested_chapter:
        chapters = [chapter for chapter in chapters if chapter.name == requested_chapter]
    if not chapters:
        raise FileNotFoundError(f"Requested chapter not found in {manga_dir}.")
    process_manga(manga_dir, chapters)


if __name__ == "__main__":
    main()
