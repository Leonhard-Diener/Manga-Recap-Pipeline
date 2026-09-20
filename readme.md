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
output/<manga title>_transformations.json # test_description.py only
```

`test_description.py` writes both files from the same Qwen page-generation pass.
The second file contains proposed transformations, reversions, disguise reveals,
possession, summoning, splitting, and fusion events, grouped by chapter and page.
Events include source/target panel numbers, visual descriptions, and evidence;
they are always marked `pending_review` and never assign character identities.
This first implementation sees one page at a time, so cross-page transitions
are not inferred. Invalid or truncated responses are flagged separately from
successful pages containing no events.
New exports also retain source-image hashes and panel-geometry revisions, allowing
the standalone identity runner to reject stale event-to-crop mappings. Expected
panel coverage is checked against the panel manifest.

The reusable API writes no files:

```python
result = describe_manga(manga_output_dir, chapter_dirs)
descriptions = result["descriptions"]
transformations = result["transformations"]
```

JSON serialization happens in the runners. `main.py` keeps its existing
descriptions export and has the event data available in memory for later stages.
The combined response has a 1024-token maximum per page (configurable in
`config.py`); longer outputs can increase generation time. Thinking remains off.

## Character identity experiment

Run `python -X utf8 test_character_reid.py "Manga title"` after `test_panels.py`.
It uses cached DINOv3 features, matches across all chapters, and exports tentative
visual groups separately from accepted identities. No Qwen model is loaded.
Unknown is preferred over a wrong name; automatic propagation requires reviewed
references and calibrated thresholds. See [CHARACTER_REID_USAGE.md](CHARACTER_REID_USAGE.md)
for review files, transformations, naming rules, current limitations, and the
reusable Python return API.
The standalone runner also saves a manga-scoped SQLite revision history and
source-linked review galleries. This workflow does not require running `main.py`.

## Debug colors (panel previews)

- Orange: raw frame/panel detection
- Magenta: text detection
- Cyan: body/character detection
- Lime: final crop
