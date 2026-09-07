# Entry point for testing the ShadowB panel extraction model.

import config

from panel_extractor_B.panel_extractor import extract_panels


def main():
    """Run ShadowB panel extraction and print the results."""
    panels = extract_panels(config.INPUT_DIR)

    for panel in panels:
        print(
            f"Panel {panel.index:03d} | "
            f"Page: {panel.source_page} | "
            f"Confidence: {panel.confidence:.2f} | "
            f"Box: {list(panel.box)} | "
            f"Saved: {panel.output_path}"
        )

    page_count = len(
        {panel.source_page for panel in panels}
    )

    print(
        f"\nDone. Extracted {len(panels)} panels "
        f"from {page_count} pages."
    )


if __name__ == "__main__":
    main()