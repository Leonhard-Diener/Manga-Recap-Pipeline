"""High-level extraction pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

import numpy as np

from .detector import IMAGE_SUFFIXES, load_session, page_tiles, predict
from .export import build_result, save_panel_crop
from .postprocess import (
    debug_image,
    deduplicate,
    estimate_coverage,
    filter_small_panels,
    final_nested_cleanup,
    protect_content,
)
from .types import Detection, PanelDetection


def extract_panels(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    *,
    image_size: int,
    panel_confidence: float,
    text_confidence: float,
    body_confidence: float,
    enable_tiled_fallback: bool,
    fallback_uncovered_ratio: float,
    fallback_grid: tuple[int, int],
    fallback_overlap: float,
    minimum_panel_ratio: float,
    duplicate_iou: float,
    duplicate_containment: float,
    text_padding_x: float,
    text_padding_y: float,
    text_safety_pixels: int,
    body_padding_x: float,
    body_padding_y: float,
    body_safety_pixels: int,
    max_content_expansion_ratio: float,
    jpeg_quality: int,
    save_debug: bool,
) -> list[PanelDetection]:
    """Extract all final panels from the images in ``input_dir``."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    session = load_session(model_path)

    pages = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not pages:
        raise FileNotFoundError(f"No image pages found in {input_dir}")

    results: list[PanelDetection] = []
    next_index = 1

    for page_path in pages:
        image = Image.open(page_path).convert("RGB")
        width, height = image.size
        page_area = width * height

        raw_panels, texts, bodies = predict(
            session,
            image,
            image_size=image_size,
            panel_confidence=panel_confidence,
            text_confidence=text_confidence,
            body_confidence=body_confidence,
        )

        coverage_before = estimate_coverage(raw_panels, width, height)
        fallback_added = 0

        # Use tiled inference only when full-page detection leaves a substantial
        # portion of the page unexplained. Recovered boxes are deduplicated again.
        if enable_tiled_fallback and (1.0 - coverage_before) >= fallback_uncovered_ratio:
            recovered_panels, recovered_texts, recovered_bodies = page_tiles(
                image,
                session,
                image_size=image_size,
                panel_confidence=max(0.10, panel_confidence - 0.10),
                text_confidence=text_confidence,
                body_confidence=body_confidence,
                grid=fallback_grid,
                overlap=fallback_overlap,
            )
            before = len(raw_panels)
            raw_panels.extend(recovered_panels)
            texts.extend(recovered_texts)
            bodies.extend(recovered_bodies)
            raw_panels, _ = deduplicate(
                raw_panels,
                iou_threshold=duplicate_iou,
                containment_threshold=duplicate_containment,
            )
            fallback_added = max(0, len(raw_panels) - before)

        deduped, duplicate_removed = deduplicate(
            raw_panels,
            iou_threshold=duplicate_iou,
            containment_threshold=duplicate_containment,
        )
        filtered, small_removed = filter_small_panels(
            deduped,
            page_area=page_area,
            minimum_ratio=minimum_panel_ratio,
        )

        final_pairs: list[tuple[np.ndarray, object]] = []
        for panel in filtered:
            final_box = protect_content(
                panel.box,
                texts=texts,
                bodies=bodies,
                image_size=(width, height),
                text_padding=(text_padding_x, text_padding_y),
                text_safety=text_safety_pixels,
                body_padding=(body_padding_x, body_padding_y),
                body_safety=body_safety_pixels,
                max_expand_ratio=max_content_expansion_ratio,
            )
            final_pairs.append((final_box, panel))

        final_pairs, nested_removed = final_nested_cleanup(
            final_pairs,
            containment_threshold=duplicate_containment,
        )

        coverage_after = estimate_coverage(
            [Detection(np.asarray(box), 1.0, 2) for box, _ in final_pairs],
            width,
            height,
        )

        if save_debug:
            debug = debug_image(
                image,
                raw_panels=raw_panels,
                texts=texts,
                bodies=bodies,
                final_boxes=[box for box, _ in final_pairs],
            )
            debug.save(output_dir / f"debug_{page_path.stem}.jpg", quality=92)

        for final_box, source in final_pairs:
            output_path = output_dir / f"panel_{next_index:04d}.jpg"
            integer_box = save_panel_crop(
                image,
                final_box,
                output_path,
                jpeg_quality=jpeg_quality,
            )
            results.append(
                build_result(
                    next_index,
                    page_path.name,
                    source.score,
                    integer_box,
                    output_path,
                )
            )
            next_index += 1

        print(
            f"{page_path.name}: "
            f"frames={len(raw_panels)} | "
            f"coverage={coverage_before:.1%}->{coverage_after:.1%} | "
            f"fallback_added={fallback_added} | "
            f"duplicates_removed={duplicate_removed} | "
            f"small_removed={small_removed} | "
            f"nested_removed={nested_removed} | "
            f"final={len(final_pairs)}"
        )

    return results
