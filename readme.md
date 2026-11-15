# Manga Recap Pipeline

The analysis stages are standalone. `old/main.py` is an archived legacy entry
point, not the current workflow. Old backups and stale outputs are in `old/`;
see `old/README.md` for the archive contents.

1. **Download** (`downloader/`) — fetch chapters from MangaDex into
   `input/<manga title>/chapter_XXX/`.
2. **Panels** (`panel_extractor/`) — RT-DETR detects panels, text, and bodies; crops panels,
   and writes numbered page previews (used later to know what each
   number refers to on the page).
3. **Description** (`description/`) — Qwen produces focused visual observations,
   grounded action proposals, appearance descriptors, form-change proposals, and
   pending exact-text transcriptions.
4. **Character identity** (`story_entities/`) — ReID assigns stable instances,
   visual groups, permitted names, grounded actions, and accepted form changes.
5. **Story writer** (`story_writer/`) — text-only Qwen generation turns the
   accepted evidence into a resumable panel-by-panel recap.

## Setup

```bash
pip install -r requirements.txt
python download_model.py        # RT-DETR ONNX model -> models/model.onnx
```

Place the Qwen3.5-2B model at `models/Qwen3.5-2B` (used for description and writing).

## Run one step at a time

```bash
python test_download.py "Manga title"      # optional source download
python test_panels.py "Manga title"
python test_description.py "Manga title"
python -X utf8 test_character_reid.py "Manga title"
python test_write_storry.py "Manga title"
```

The title argument is optional if `config.MANGA_TITLE` is set, or if
`input/` already contains exactly one manga folder.

## Cloud descriptions: change models in one place

Use `test_description_openrouter.py` instead of the local description runner.
Copy `.env.example` to `.env` if needed, insert your OpenRouter key, and change
only this line whenever you want a different model:

```dotenv
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
```

A full link such as
`https://openrouter.ai/google/gemma-4-26b-a4b-it:free?view=api` also works.

```powershell
python test_description_openrouter.py "Manga title"
# Optional override for just this run:
python test_description_openrouter.py --model "https://openrouter.ai/google/gemma-4-26b-a4b-it:free?view=api"
```

The client automatically checks image support and chooses a compatible JSON
format from endpoint metadata once per run, without generating model output.
Local JSON validation and one format-only repair remain active. It never changes
your chosen model or switches a free model to a paid one. CLI options override
environment variables, which override `.env`, then defaults in `config.py`.
If an old shell-level `OPENROUTER_MODEL` is set, remove it to use `.env`.
Local Qwen settings are unaffected.

Advanced: compatible custom APIs without model metadata can use
`--response-format json_schema`, `json_object`, or `text` instead of `auto`.

For a native Google AI Studio run, put the key in `.env` and use the separate
Google runner:

```dotenv
GEMINI_API_KEY=YOUR_GOOGLE_AI_STUDIO_KEY
GOOGLE_MODEL=gemini-3.5-flash-lite
GOOGLE_MEDIA_RESOLUTION=high
```

```powershell
python -X utf8 test_description_google.py "Manga title"
```

Google runs use their own resumable chapter checkpoint. They publish the same
description JSON contract as the local and OpenRouter runners.

## Output layout

```
output/<manga title>/<chapter>/
    panel_0001.jpg ...            # step 3 crops
    page_previews/<page>.jpg      # step 3 numbered previews
    grounded_page_previews/<page>.jpg  # panel numbers + request-local C labels
output/<manga title>_descriptions.json   # descriptions + transformations
output/<manga title>/character_classification.json
output/<manga title>_story.txt
output/<manga title>_story.json
output/<manga title>_panel_timeline.json
```

`test_description.py` writes one combined file from focused Qwen passes.
Its `transformations` section contains proposed transformations, reversions, disguise reveals,
possession, summoning, splitting, and fusion events, grouped by chapter and page.
Events include source/target panel numbers, exact local character labels, visual descriptions, and evidence;
they begin as `pending_review`. During ReID, only 100%-certain direct visible
changes with current, high-quality labeled adjacent-panel endpoints can connect
separate forms under one entity. Legacy unlabeled events still require one unique
character in each endpoint panel. All other events remain pending.
Grounded panel actions use the same labels. ReID expands them to stable instance,
entity, form, display-name, and stable-description fields in `story_events`.
Clearly readable dialogue, thoughts, narration, signs, and sound effects are kept
as `pending_review` transcription proposals. They reach the writer only after a
source- and revision-bound review; model transcription alone is never a story fact.
Each result still interprets one page independently, so cross-page transitions
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

JSON serialization happens in the runners. The model emits a compact internal
schema that is expanded before export. Token-level schema enforcement guarantees
the required description and event lists instead of relying on prompt
compliance. Its tokenizer grammar is cached beside the local Qwen model after the
first build. Description inference batches compatible pairs of pages into one GPU
generation call, while each page keeps a 1024-token ceiling and is never regenerated
because of a smaller first-pass budget. Thinking remains off.

## Character identity experiment

Run `python -X utf8 test_character_reid.py "Manga title"` after `test_panels.py`.
It uses cached DINOv3 features, matches across all chapters, and exports tentative
visual groups separately from accepted identities. No Qwen model is loaded.
Canonical identities remain conservative. Retained visual groups automatically receive
a stable pseudorandom display name from a bundled pool of 200 common Japanese surnames;
ungrouped observations remain unnamed. See [CHARACTER_REID_USAGE.md](CHARACTER_REID_USAGE.md)
for review files, transformations, display-name behavior, current limitations, and the
reusable Python return API.
The standalone runner also saves a manga-scoped SQLite revision history and
source-linked review galleries. This workflow does not require running `main.py`.

## Story writer

Run `test_write_storry.py` after the panel, description, and character-ReID
stages. It rejects stale hashes and old exports that do not contain grounded
actions. It sends only accepted actions and accepted transformations to Qwen,
uses the per-instance permitted name or stable visual reference, and validates
the returned panel order, support IDs, instance IDs, name uses, and approximate
segment length.

Progress is written after every successful chunk and automatically resumes when
the inputs and settings still match. `--restart` archives compatible progress
and starts a new run. Raw responses from both allowed attempts are retained in
`output/<manga title>/story_runs/<run id>/`; a chunk that is still invalid after
one correction retry stops without publishing a partial final story.

For fidelity, the whole-story outline is assembled from existing descriptions
instead of asking the model to invent a summary. The final boundary pass only
removes an exactly duplicated sentence across adjacent chunks; it does not
rewrite accepted story facts.

## Debug colors (panel previews)

- Orange: raw frame/panel detection
- Magenta: text detection
- Cyan: body/character detection
- Lime: final crop
