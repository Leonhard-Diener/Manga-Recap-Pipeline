from config import IMAGE_PATH
from panel_extractor import extract_panels


def main():
    panels = extract_panels(IMAGE_PATH)

    for panel in panels:
        print(
            f"Panel {panel.index:02d} | "
            f"Confidence: {panel.confidence:.2f} | "
            f"Box: {list(panel.box)} | "
            f"Saved: {panel.output_path}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()