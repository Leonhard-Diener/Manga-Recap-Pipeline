from ultralytics import YOLO
import config
import os
from PIL import Image
from image_utils import trim_white_border
from dataclasses import dataclass


@dataclass
class Panel:
    index: int
    box: tuple[int, int, int, int]
    confidence: float
    output_path: str
    

def detect_panels(model: YOLO, image_path: str) -> list:
    # Run YOLO inference and return the detected bounding boxes.
    results = model.predict(
        image_path,
        conf=config.CONFIDENCE,
        classes=[config.PANEL_CLASS_ID],
        verbose=False,
    )
    return results[0].boxes


def extract_panels(image_path: str) -> list:
    # Detect panels, crop them, and save the results into output_dir.
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    model = YOLO(config.MODEL_PATH)
    boxes = detect_panels(model, image_path)

    panels = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

        panel_image = trim_white_border(image.crop((x1, y1, x2, y2)))

        output_path = os.path.join(config.OUTPUT_DIR, f"{i + 1}.jpg")
        panel_image.save(output_path, quality=config.JPEG_QUALITY)

        panels.append(Panel(
            index=i + 1,
            box=(x1, y1, x2, y2),
            confidence=float(box.conf[0]),
            output_path=output_path,
        ))

    return panels