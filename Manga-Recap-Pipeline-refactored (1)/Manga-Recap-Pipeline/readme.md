# Manga Recap Pipeline

Four steps, each runnable on its own or all together via `main.py`.

1. **Download** (`downloader/`) — fetch chapters from MangaDex into
   `input/<manga title>/chapter_XXX/`.
2. **Cleanup** (`cleanup/`) — a small VLM classifies each page as
   `MANGA`/`EXTRA` and deletes the credits/shoutout/intro pages at the
   start and end of every chapter. Only edge pages are touched.
3. **Panels** (`panel_extractor/`) — RT-DETR detects panels, crops them,
   and writes numbered page previews (used later to know what each
   number refers to on the page).
4. **Description** (`description/`) — the same VLM describes each
   numbered panel on every page preview from step 3.

## Setup

```bash
pip install -r requirements.txt
python download_model.py        # RT-DETR ONNX model -> models/model.onnx
```

Place the Qwen3.5-2B model at `models/Qwen3.5-2B` (used for steps 2 and 4).

## Run everything

```bash
python main.py "Manga title"
python main.py "Manga title" --no-all-chapters   # only the first chapter
```

Each step's model is loaded once and reused across all chapters before
moving to the next step.

## Run one step at a time

```bash
python test_download.py "Manga title"      # step 1
python test_cleanup.py "Manga title"       # step 2 — deletes extra pages!
python test_panels.py "Manga title"        # step 3
python test_description.py "Manga title"   # step 4
```

The title argument is optional if `config.MANGA_TITLE` is set, or if
`input/` already contains exactly one manga folder.

## Output layout

```
output/<manga title>/<chapter>/
    panel_0001.jpg ...            # step 3 crops
    page_previews/<page>.jpg      # step 3 numbered previews
output/<manga title>_descriptions.json   # step 4
```

## Debug colors (panel previews)

- Orange: raw frame/panel detection
- Magenta: text detection
- Cyan: body/character detection
- Lime: final crop
