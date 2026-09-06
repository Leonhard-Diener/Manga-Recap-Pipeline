from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

#config
IMAGE_PATH = "test/01.jpg"
MODEL_PATH = "models/manga_panel_detector_fp32.pt"
OUTPUT_DIR = "output"

CONFIDENCE = 0.25



def trim_white_border(image):

    img = np.array(image)

    # RGB -> Gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    dark = gray < 100

    height, width = gray.shape


    left = 0

    for x in range(width):
        dark_ratio = np.mean(dark[:, x])

        if dark_ratio > 0.05:
            left = x
            break


    right = width - 1

    for x in range(width - 1, -1, -1):
        dark_ratio = np.mean(dark[:, x])

        if dark_ratio > 0.05:
            right = x
            break


    top = 0

    for y in range(height):
        dark_ratio = np.mean(dark[y, :])

        if dark_ratio > 0.05:
            top = y
            break

    bottom = height - 1

    for y in range(height - 1, -1, -1):
        dark_ratio = np.mean(dark[y, :])

        if dark_ratio > 0.05:
            bottom = y
            break

    margin = 2

    left = min(left + margin, width - 1)
    right = max(right - margin, left + 1)

    top = min(top + margin, height - 1)
    bottom = max(bottom - margin, top + 1)

    return image.crop((
        left,
        top,
        right + 1,
        bottom + 1
    ))


os.makedirs(OUTPUT_DIR, exist_ok=True)
image = Image.open(IMAGE_PATH).convert("RGB")

model = YOLO(MODEL_PATH)


results = model.predict(
    IMAGE_PATH,
    conf=CONFIDENCE,
    classes=[0],      # 0 = panel
    verbose=False
)



for i, box in enumerate(results[0].boxes):

    # Bounding Box
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)


    panel = image.crop((
        x1,
        y1,
        x2,
        y2
    ))

    panel = trim_white_border(panel)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{i + 1}.jpg"
    )

    panel.save(
        output_path,
        quality=95
    )


    confidence = float(box.conf[0])

    print(
        f"Panel {i + 1:02d} | "
        f"Confidence: {confidence:.2f} | "
        f"Box: [{x1}, {y1}, {x2}, {y2}] | "
        f"Saved: {output_path}"
    )


print("\nFertig.")
