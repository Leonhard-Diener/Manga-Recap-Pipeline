# Entry point: extracts panels from every page in the configured input folder.

import config
from panel_extractor.panel_extractor import extract_panels


def main():
    # downloads the manga pages


    # extracts panels from every page in the configured input folder
    panels = extract_panels(config.INPUT_DIR)

    for panel in panels:
        print(
            f"Panel {panel.index:03d} | "
            f"Page: {panel.source_page} | "
            f"Confidence: {panel.confidence:.2f} | "
            f"Box: {list(panel.box)} | "
            f"Saved: {panel.output_path}"
        )

    page_count = len({panel.source_page for panel in panels})
    print(f"\nDone. Extracted {len(panels)} panels from {page_count} pages.")


    # creates a video from the extracted panels

if __name__ == "__main__":
    main()
