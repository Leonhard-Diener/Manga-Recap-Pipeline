"""Entry point for the classical-CV manga panel extractor."""

from pathlib import Path

import config
from panel_extractor_C.manga_panel_detector import process_chapter


def main() -> None:
    """Process the configured input chapter and print the results."""
    panels = process_chapter(
        Path(config.INPUT_DIR),
        Path(config.OUTPUT_DIR),
        reading_order="rtl" if config.READING_ORDER_RTL else "ltr",
        split_spread_flag=False,
        fmt="jpg",
        pad_ratio=config.TRIM_MARGIN / 100.0,
        deskew=True,
        crops=True,
        jpeg_quality=config.JPEG_QUALITY,
        save_json=True,
        separator_first=True,
        separator_min_score=0.42,
        separator_line_ratio=0.05,
        separator_min_span_ratio=0.30,
        separator_min_region_width_ratio=0.08,
        separator_min_region_height_ratio=0.08,
        separator_max_depth=8,
        separator_min_leaf_count=2,
        enable_contour_method=True,
        enable_watershed_method=True,
        enable_grid_fallback=True,
        denoise=True,
        denoise_kernel_ratio=0.003,
        denoise_strength=1,
        bleed_border=True,
        bleed_border_ratio=0.015,
        allow_rotated=True,
        rotation_threshold_deg=3.0,
        edge_method="black",
        black_threshold=config.DARK_PIXEL_THRESHOLD,
        white_threshold=235,
        min_panel_area_ratio=0.008,
        max_panel_area_ratio=0.65,
        min_aspect_ratio=0.15,
        max_aspect_ratio=6.0,
        debug=True,
    )

    page_count = len({panel["source_page"] for panel in panels})

    for panel in panels:
        print(
            f"Panel {panel['reading_index']:03d} | "
            f"Page: {panel['source_page']} | "
            f"Method: {panel.get('method', 'unknown')} | "
            f"Score: {panel.get('score', 0.0):.2f} | "
            f"Box: {panel['bbox']} | "
            f"Saved: {panel.get('file')}"
        )

    print(
        f"\nDone. Extracted {len(panels)} panels "
        f"from {page_count} pages."
    )


if __name__ == "__main__":
    main()
