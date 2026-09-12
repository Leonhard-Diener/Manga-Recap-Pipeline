"""Command-line entry point for manga panel extraction."""

import config
from panel_extractor import extract_panels


def main() -> None:
    """Run extraction using the central application configuration."""
    panels = extract_panels(
        input_dir=config.INPUT_DIR,
        output_dir=config.OUTPUT_DIR,
        model_path=config.MODEL_PATH,
        image_size=config.IMAGE_SIZE,
        panel_confidence=config.PANEL_CONFIDENCE,
        text_confidence=config.TEXT_CONFIDENCE,
        body_confidence=config.BODY_CONFIDENCE,
        enable_tiled_fallback=config.ENABLE_TILED_FALLBACK,
        fallback_uncovered_ratio=config.FALLBACK_UNCOVERED_RATIO,
        fallback_grid=config.FALLBACK_GRID,
        fallback_overlap=config.FALLBACK_OVERLAP,
        minimum_panel_ratio=config.MINIMUM_PANEL_RATIO,
        duplicate_iou=config.DUPLICATE_IOU,
        duplicate_containment=config.DUPLICATE_CONTAINMENT,
        text_padding_x=config.TEXT_PADDING_X,
        text_padding_y=config.TEXT_PADDING_Y,
        text_safety_pixels=config.TEXT_SAFETY_PIXELS,
        body_padding_x=config.BODY_PADDING_X,
        body_padding_y=config.BODY_PADDING_Y,
        body_safety_pixels=config.BODY_SAFETY_PIXELS,
        max_content_expansion_ratio=config.MAX_CONTENT_EXPANSION_RATIO,
        jpeg_quality=config.JPEG_QUALITY,
        save_debug=config.SAVE_DEBUG,
    )

    page_count = len({panel.source_page for panel in panels})
    for panel in panels:
        print(
            f"Panel {panel.index:04d} | "
            f"Page: {panel.source_page} | "
            f"Confidence: {panel.confidence:.3f} | "
            f"Box: {list(panel.box)} | "
            f"Saved: {panel.output_path}"
        )

    print(
        f"\nDone. Extracted {len(panels)} panels "
        f"from {page_count} pages."
    )


if __name__ == "__main__":
    main()
