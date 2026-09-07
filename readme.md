# RT-DETRv4-X Manga Panel Extractor

A clean, modular implementation of the working RT-DETRv4-X Manga109-s panel extraction pipeline.

## Pipeline

1. RT-DETRv4-X detects `body`, `text`, and `frame`.
2. A page-level tiled fallback recovers panels when more than 35% of the page is not covered by frame detections.
3. Duplicate frame detections are removed using IoU and containment checks.
4. Panels below 5% of page area are discarded.
5. Text and body detections may expand a frame only when the detected content actually crosses the frame boundary.
6. A final nested-box check prevents duplicate exports.
7. Crops are written directly to `output/` with global numbering.

White-border trimming and recursive panel detection are intentionally not used.

## Model

Place the RT-DETRv4-X Manga109-s ONNX model at:

`models/model.onnx`

or run:

```bash
python download_model.py
```

## Run

```bash
pip install -r requirements.txt
python main3.py
```

Input pages go in `input/`.
All crops go directly into `output/`.

## Debug colors

- Orange: raw frame/panel detection
- Magenta: text detection
- Cyan: body/character detection
- Lime: final crop
