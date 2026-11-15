# Manga Recap Project Guide

## Project purpose

This repository builds a grounded manga-recap dataset and narration from manga page images. The intended evidence flow is:

```text
MangaDex/source pages
  -> panel, text, and body detection
  -> stable page/panel/detection manifests and crops
  -> panel descriptions and grounded visual-event proposals
  -> manga-wide character ReID and reviewed identity decisions
  -> evidence-constrained story narration and panel timeline
```

The important design rule is that uncertainty is preserved. A visual group is not automatically a character identity, model confidence is not proof, and pending actions or transformations must not be narrated as facts.

## Current entry points and execution order

Use the standalone scripts in this order:

1. `test_download.py ["Manga title"]` — optional MangaDex download into `input/<manga>/chapter_*/`.
2. `test_panels.py ["Manga title"]` — panel/body/text detection, crops, previews, and manifests.
3. `test_description.py ["Manga title"]` — page-role filtering, descriptions, character observations, grounded action proposals, and transformation proposals.
4. `test_character_reid.py ["Manga title"]` — manga-wide visual matching, identities, names, grounded events, reviews, and revision history.
5. `test_write_storry.py ["Manga title"]` — resumable narration generation and panel timing. The `storry` filename is intentionally retained for compatibility.

Files named `test_*.py` at the repository root are operational stage runners, not unit tests. Tests under `tests/` are the actual automated test suite.

`old/main.py` is archived legacy code and is not the authoritative workflow. It still imports the removed `cleanup` package and only represents an older four-stage pipeline. Do not use it as the basis for new work unless the task explicitly includes repairing/replacing the legacy entry point.

## Source and output layout

```text
input/<manga>/<chapter>/
    page images                         # preserved source evidence

output/<manga>/<chapter>/
    panel_0001.jpg ...                  # ordered panel crops
    page_previews/<page>.jpg            # numbered panel overlay
    grounded_page_previews/<page>.jpg   # panel numbers and page-local C labels
    panel_manifest.json                 # hashes, IDs, geometry/order, roles, regions
    character_detections.json           # durable body proposals and C-label mappings
    character_classification.json       # chapter subset of the ReID result

output/<manga>_descriptions.json        # combined description/page-filter/event export
output/<manga>/character_classification.json
output/<manga>/character_identities.sqlite3
output/<manga>/reid_cache/
output/<manga>/reid_runs/<run id>/
output/<manga>/story_runs/<run id>/     # raw writer attempts and diagnostics
output/<manga>_story_progress.json
output/<manga>_story.txt
output/<manga>_story.json
output/<manga>_panel_timeline.json
```

Generated files are evidence-bearing artifacts. Prefer regenerating them through their runner or importing an explicit reviewed correction instead of hand-editing final JSON. Source hashes, geometry revisions, grounding revisions, identity revisions, and writer fingerprints deliberately invalidate stale downstream artifacts.

## File and package responsibilities

### Configuration and shared utilities

- `config.py` — all project paths, detector thresholds, crop/grounding settings, Qwen pixel/token/batch settings, and story chunk/timing settings.
- `common/utils.py` — natural image ordering, manga/chapter discovery, image listing, and tensor device transfer.
- `common/json_io.py` — atomic JSON publication. Runners should use `write_json` rather than partially writing output files.
- `download_model.py` — downloads the RT-DETR ONNX detector used by the panel stage.
- `requirements.txt` — Python dependencies; GPU ONNX deployments may additionally require the appropriate ONNX Runtime GPU installation.

### Download stage: `downloader/`

- `get_manga_id.py` and `get_chapter_ids.py` — MangaDex lookup.
- `download_chapter.py` — concurrent chapter/page download with shared rate limiting and source refresh behavior.
- `mangadex_api.py` — HTTP retry and MangaDex request behavior.

The downloader owns source acquisition only. Later stages must not mutate downloaded page pixels.

### Detection stage: `panel_extractor/`

- `detector.py` — ONNX session, preprocessing, full-page/tiled RT-DETR inference, and raw frame/text/body detections.
- `pipeline.py` — high-level detection, fallback tiling, post-processing, reading-order sorting, preview creation, and crop orchestration.
- `postprocess.py` — panel filtering, deduplication, coverage checks, nesting cleanup, and optional content protection.
- `splitter.py` — detects/splits merged panels using frame-line evidence and an optional detector fallback.
- `page_preview.py` — right-to-left manga reading order plus numbered and grounded overlay rendering.
- `geometry.py` — box intersection, containment, clipping, and area primitives.
- `export.py` and `types.py` — crop/result serialization and detection data types.
- `runner.py` — stage publication: crops, grounded previews, `panel_manifest.json`, and `character_detections.json`.

Panel geometry and reading order are upstream evidence. A description fix cannot recover a missing/merged panel or correct a wrong order without updating the manifest revision.

### Description stage: `description/`

- `describer.py` — Qwen loading, page-role classification, batching, panel/character/action/transformation requests, focused retries, and manga/chapter orchestration. `describe_manga(...)` returns data; the CLI runner owns the combined JSON write.
- `images.py` — temporary text masking and clean panel cropping helpers. Derived images must remain in source coordinates and must not replace original pages.
- `schema.py` — compact JSON schemas and token-level `lm-format-enforcer` prefix filters/caches.
- `response.py` — parses compact model JSON into the public schema, validates panel and label mappings, and fails closed on malformed/truncated evidence.
- `quality.py` — semantic checks for fragments, placeholders, repeated descriptions, unusable appearances, and nonvisual/ungrounded actions.

`C1`, `C2`, etc. are request-local labels, never names or global identities. Their `detection_id` is the durable bridge to ReID. Keep all public description/page-filter/transformation sections and stable mappings when refactoring; new status/provenance fields should normally be additive.

### Character identity stage: `story_entities/`

- `manifest.py` — source hashing, persistent page/panel/detection IDs, geometry/grounding revisions, and normalized body proposals.
- `pipeline.py` — collects instances/crops, caches encoder features, constructs distance evidence, resolves reviews and transformations, attaches grounded descriptions, and creates story events.
- `matching.py` — deterministic conservative grouping and identity/form propagation. Visual similarity creates proposals, not truth.
- `naming.py` and `manga_names.json` — deterministic fallback display aliases and reviewed-name handling.
- `storage.py` — transactional manga-scoped SQLite revision history; revoked/older evidence is retained.
- `runner.py` — classification JSON, per-chapter subsets, source-linked review gallery, run snapshots, and database publication.
- `CHARACTER_REID_USAGE.md` — current CLI/review/calibration format and limitations.
- `CHARACTER_IDENTITY_PIPELINE_SPEC.md` — design background; some sections describe future work, so confirm behavior in code/tests.

The normal primary encoder is DINOv3; CCIP can supply style-link evidence. ReID does not need Qwen inference. Exact reviewed assignments override proposals only through validated review records. Unknown identity is preferable to a wrong merge.

### Story stage: `story_writer/`

- `writer.py` — validates all upstream files, joins panels/instances/events, builds prompts, validates model JSON, checkpoints chunks, resumes compatible runs, archives stale/restarted outputs, and publishes narration/timeline files.
- `__init__.py` — public `run_story_writer` export.
- `STORY_WRITER_SPEC.md` — intended evidence and output contract; check implementation/tests where the spec and current behavior differ.

The writer is text-only and uses a private llama.cpp server with Qwen3.5-4B Q8_0 GGUF, independently of the description checkpoint. It must consume only current, accepted evidence, retain ordered panel coverage, enforce exact IDs/name permissions, and never silently publish a partial failed run.

## Models and their roles

- `models/model.onnx` — RT-DETRv4-X Manga109 detector for frames, text regions, and body proposals.
- `models/Qwen3.5-2B/` by default — page roles, descriptions, and focused visual extraction. The path is controlled by `config.QWEN_MODEL_PATH`.
- `models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf` — text-only story writing through `models/llama.cpp-b12292/llama-server.exe` (CUDA 12.4 for GTX 1080 support). Paths are controlled by `config.STORY_MODEL_PATH` and `config.STORY_LLAMA_SERVER`.
- `models/DINOv3/dino_v3.onnx` plus its data file — primary manga-wide visual embeddings.
- `models/CCIP/ccip-caformer_b36-24/` — optional secondary style-link evidence.

Changing a model, preprocessing rule, pixel budget, prompt, schema, or acceptance policy must change the relevant fingerprint/revision so cached evidence is not silently reused.

## Core invariants

- Preserve original pages. All masks, overlays, crops, and galleries are derived assets.
- Preserve panel coverage and manga reading order. Never improve metrics by dropping difficult panels or detections.
- Stable IDs come from source identity, hashes, and geometry—not array positions or display labels.
- Every grounded character label must resolve to exactly one current occurrence before downstream use; an occurrence may still remain unknown or unusable.
- Do not convert visual groups, aliases, model confidence, neighboring panels, or similar descriptions into confirmed identity.
- Pending/rejected actions and transformations are not story facts. Identity links and event acceptance are separate decisions.
- Page roles and all event endpoints must match current source, geometry, and grounding revisions.
- Structural JSON validity is not factual validity. Keep semantic quality status and raw failed responses available for audit.
- Reusable library functions return structured data; command runners own final publication.
- Preserve existing final output fields and the three story outputs unless a requested migration explicitly changes their contract.

## Tests and verification

Run the model-free suite with:

```powershell
python -X utf8 -m unittest discover -s tests -v
```

Important focused suites:

- `tests/test_description_output.py` — schema, batching, parsing, provenance, truncation, and coverage.
- `tests/test_description_quality.py` — observed Qwen failure regressions, masks, focused repair, and semantic rejection.
- `tests/test_entity_matching.py` — conservative grouping, recurrence, naming, and abstention.
- `tests/test_identity_workflow.py` — style links, reviews, transformation/action grounding, revisions, and export history.
- `tests/test_story_writer.py` — input joins, stale evidence, response validation, retries, resume, publication, and timeline.
- `tests/test_downloader.py` and `tests/test_mangadex_api.py` — concurrency, refresh, retry, and rate-limit behavior.

Prefer narrow tests while iterating, then run discovery before handoff. Unit tests mock model calls and should not require loading Qwen. For image/geometry changes, also inspect representative source pages and generated previews; text-only assertions cannot validate visual correctness.

## Planning and status documents

- `STORY_READINESS_PLAN.md` is the current proposed plan for making the result usable and faster. It records unimplemented gates and fallbacks; do not describe its proposals as completed code.
- `optimization.md` records the earlier Qwen failure analysis and optimization ideas.
- `todo.md` contains historical observations and future product stages such as TTS/video; parts may be stale.
- `readme.md` is useful background but still describes the removed cleanup stage and legacy `main.py`. The current code, tests, and this guide take precedence when they conflict.

## Change guidance

- Trace both the producer and every consumer before changing a JSON field or acceptance status.
- Keep description, ReID, and writer policy names synchronized; consumers currently recognize specific acceptance policies.
- Invalidate only affected downstream artifacts when possible. Description-only changes do not require rerunning panel detection if source/geometry/grounding stayed current.
- Do not delete caches, run snapshots, review history, or user-modified artifacts as part of ordinary cleanup.
- The worktree may contain ongoing changes. Preserve unrelated edits and inspect diffs narrowly.
