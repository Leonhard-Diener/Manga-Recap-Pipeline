"""High-level extraction pipeline.

Three public functions are provided:

``detect_panels(input_dir, model_path, ...)``
    Runs the RT-DETR detector on every page in *input_dir*, applies all
    post-processing, sorts the final boxes into manga reading order, and saves
    annotated preview images.  Returns one :class:`PageDetection` per page.
    No panel crops are written.

``crop_panels(detections, output_dir, ...)``
    Takes the list of :class:`PageDetection` objects produced by
    ``detect_panels`` and saves one JPEG crop per panel.  The detector is
    never run again.  Returns one :class:`PanelDetection` per saved crop.

``extract_panels(input_dir, output_dir, model_path, ...)``
    Convenience wrapper that calls ``detect_panels`` then ``crop_panels`` in
    one shot — used by the production pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .detector import IMAGE_SUFFIXES, load_session, page_tiles, predict
from .export import build_result, save_panel_crop
from .page_preview import save_page_preview, _manga_reading_order
from .postprocess import (
    debug_image,
    deduplicate,
    estimate_coverage,
    filter_small_panels,
    final_nested_cleanup,
    protect_content,
)
from .types import Detection, PageDetection, PanelDetection


# ── detect_panels ─────────────────────────────────────────────────────────────

def detect_panels(
    input_dir: Path,
    model_path: Path,
    output_dir: Path,
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
    # ── Post-processing step toggles ──────────────────────────────────────────
    enable_deduplication: bool = True,
    enable_small_panel_filter: bool = True,
    enable_content_protection: bool = True,
    enable_nested_cleanup: bool = True,
    # ─────────────────────────────────────────────────────────────────────────
    text_padding_x: float,
    text_padding_y: float,
    text_safety_pixels: int,
    body_padding_x: float,
    body_padding_y: float,
    body_safety_pixels: int,
    max_content_expansion_ratio: float,
    jpeg_quality: int,
    save_debug: bool,
) -> list[PageDetection]:
    """Run detection and post-processing on every page in *input_dir*.

    Saves annotated preview images (numbered borders) to
    ``output_dir/page_previews/`` and optional debug overlays to
    ``output_dir/``.  No panel crops are written.

    Returns
    -------
    One :class:`PageDetection` per page, in page order.  Each object holds
    the final boxes already sorted into manga reading order.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    session = load_session(model_path)

    pages = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not pages:
        raise FileNotFoundError(f"No image pages found in {input_dir}")

    results: list[PageDetection] = []

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
            if enable_deduplication:
                raw_panels, _ = deduplicate(
                    raw_panels,
                    iou_threshold=duplicate_iou,
                    containment_threshold=duplicate_containment,
                )
            fallback_added = max(0, len(raw_panels) - before)

        # ── Step 1: Deduplication ─────────────────────────────────────────────
        if enable_deduplication:
            deduped, duplicate_removed = deduplicate(
                raw_panels,
                iou_threshold=duplicate_iou,
                containment_threshold=duplicate_containment,
            )
        else:
            deduped, duplicate_removed = list(raw_panels), 0

        # ── Step 2: Small-panel filter ────────────────────────────────────────
        if enable_small_panel_filter:
            filtered, small_removed = filter_small_panels(
                deduped,
                page_area=page_area,
                minimum_ratio=minimum_panel_ratio,
            )
        else:
            filtered, small_removed = list(deduped), 0

        # ── Step 3: Content protection ────────────────────────────────────────
        final_pairs: list[tuple[np.ndarray, Detection]] = []
        for panel in filtered:
            if enable_content_protection:
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
            else:
                final_box = panel.box.copy()
            final_pairs.append((final_box, panel))

        # ── Step 4: Nested-box cleanup ────────────────────────────────────────
        if enable_nested_cleanup:
            final_pairs, nested_removed = final_nested_cleanup(
                final_pairs,
                containment_threshold=duplicate_containment,
            )
        else:
            nested_removed = 0

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

        # Sort into manga reading order and save the annotated preview.
        raw_boxes = [box for box, _ in final_pairs]
        raw_scores = [float(src.score) for _, src in final_pairs]

        ordered_indices = _manga_reading_order(raw_boxes)
        ordered_boxes = [
            [int(round(v)) for v in raw_boxes[i]] for i in ordered_indices
        ]
        ordered_scores = [raw_scores[i] for i in ordered_indices]

        if raw_boxes:
            save_page_preview(
                image,
                raw_boxes,
                page_path,
                output_dir,
                jpeg_quality=jpeg_quality,
            )

        results.append(
            PageDetection(
                page_path=page_path,
                boxes=ordered_boxes,
                scores=ordered_scores,
            )
        )

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


# ── crop_panels ───────────────────────────────────────────────────────────────

def crop_panels(
    detections: list[PageDetection],
    output_dir: Path,
    *,
    jpeg_quality: int,
) -> list[PanelDetection]:
    """Crop and save one JPEG per panel from pre-computed detections.

    Parameters
    ----------
    detections:
        Output of :func:`detect_panels` — one entry per page, boxes already in
        manga reading order.
    output_dir:
        Directory where ``panel_XXXX.jpg`` files are written.
    jpeg_quality:
        JPEG quality for the saved crops.

    Returns
    -------
    One :class:`PanelDetection` per saved crop, in reading order across all
    pages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[PanelDetection] = []
    next_index = 1

    for page_detection in detections:
        image = Image.open(page_detection.page_path).convert("RGB")

        for box, score in zip(page_detection.boxes, page_detection.scores):
            output_path = output_dir / f"panel_{next_index:04d}.jpg"
            integer_box = save_panel_crop(
                image,
                np.asarray(box, dtype=np.float32),
                output_path,
                jpeg_quality=jpeg_quality,
            )
            results.append(
                build_result(
                    next_index,
                    page_detection.page_path.name,
                    score,
                    integer_box,
                    output_path,
                )
            )
            next_index += 1

    return results


# ── extract_panels (convenience wrapper) ──────────────────────────────────────

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
    enable_deduplication: bool = True,
    enable_small_panel_filter: bool = True,
    enable_content_protection: bool = True,
    enable_nested_cleanup: bool = True,
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
    """Detect panels then crop them in one call.

    This is the entry point used by the production pipeline.  It calls
    :func:`detect_panels` followed by :func:`crop_panels` without writing any
    intermediate JSON.
    """
    detections = detect_panels(
        input_dir=input_dir,
        model_path=model_path,
        output_dir=output_dir,
        image_size=image_size,
        panel_confidence=panel_confidence,
        text_confidence=text_confidence,
        body_confidence=body_confidence,
        enable_tiled_fallback=enable_tiled_fallback,
        fallback_uncovered_ratio=fallback_uncovered_ratio,
        fallback_grid=fallback_grid,
        fallback_overlap=fallback_overlap,
        minimum_panel_ratio=minimum_panel_ratio,
        duplicate_iou=duplicate_iou,
        duplicate_containment=duplicate_containment,
        enable_deduplication=enable_deduplication,
        enable_small_panel_filter=enable_small_panel_filter,
        enable_content_protection=enable_content_protection,
        enable_nested_cleanup=enable_nested_cleanup,
        text_padding_x=text_padding_x,
        text_padding_y=text_padding_y,
        text_safety_pixels=text_safety_pixels,
        body_padding_x=body_padding_x,
        body_padding_y=body_padding_y,
        body_safety_pixels=body_safety_pixels,
        max_content_expansion_ratio=max_content_expansion_ratio,
        jpeg_quality=jpeg_quality,
        save_debug=save_debug,
    )
    return crop_panels(
        detections,
        output_dir,
        jpeg_quality=jpeg_quality,
    )
