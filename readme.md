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
python main.py "Manga title"
```

`main.py` downloads the selected MangaDex chapters, asks MangaLMM whether the
pages directly at each chapter's start and end are story pages, deletes only
the surrounding credits/shoutouts/end cards, then passes the remaining pages
to the existing panel extractor. It never crops or changes the source pages'
panels during recognition.

MangaLMM is downloaded automatically from Hugging Face into `models/MangaLMM/`
on the first run. It is an 8B visual-language model, so a GPU with sufficient
VRAM is strongly recommended. The same model is exposed through
`MangaLMMRecognizer` in `manga_recognizer.py` for later panel descriptions and
character/action analysis.

To download only the first chapter:

```bash
python main.py "Manga title" --no-all-chapters
```

Check your real downloaded page directory with MangaLMM (without changing a
file) using:

```bash
python test_main.py
```

After reviewing its list, delete the detected credits/end pages with:

```bash
python test_main.py --apply
```

## Debug colors

- Orange: raw frame/panel detection
- Magenta: text detection
- Cyan: body/character detection
- Lime: final crop
