# Character identity runner

`test_character_reid.py` now performs manga-wide, precision-first visual matching.
It does not load Qwen, invent names, or automatically equate a creature with a
human because a description suggests a transformation. The reusable Python
implementation is in `story_entities/`; JSON export belongs to the runner.

## 1. Prepare and run

```powershell
python -X utf8 test_panels.py "Maou-sama"
python -X utf8 test_character_reid.py "Maou-sama"
```

`main.py` is not part of this workflow. Re-run only the ReID command after editing
reviews; it reuses cached embeddings and does not run the detector or Qwen.

The panel step supplies full-page body proposals plus a `panel_manifest.json`
containing panel boxes, reading-order numbers, and image hashes. The runner also
reads existing `person_detections.json` if available; running the person-only
detector is not required. Nearly identical boxes are deduplicated. All available
chapters participate, even when the legacy chapter argument is supplied.
New detector exports include source hashes. Mismatched hashes are rejected;
legacy detections without hashes remain visible but cannot seed visual matching.
Run the panel command once to refresh those inputs.

For experiments without replacing existing classification exports:

```powershell
python -X utf8 test_character_reid.py "Maou-sama" --export-dir debug/reid_evaluation
```

The default uses the existing local DINOv3 Anime-ReID ONNX model for strict core
groups, then the existing CCIP model as a second opinion for simple-versus-detailed
drawing links. `--style-linker none` disables that second pass. `--encoder ccip`
selects CCIP as the primary encoder and therefore requires `--style-linker none`.
`--provider cpu` forces CPU; otherwise CUDA is used when available. Features are
cached by exact model/preprocessing and image/crop fingerprints. Cached reruns avoid
loading either feature model. Unusable crops are retained for inspection but not
embedded. No new model is downloaded, and neither pass uses Qwen.

DINO visual grouping uses a `0.20` cosine-distance cutoff. The earlier DINO
cutoff of `0.08` fragmented drawing styles; CCIP at `0.20` mixed different people.
The inspected development comparison favored DINO at `0.20`. To adjust it, pass
`--group-distance 0.18` (stricter) or another chosen value. This changes visual
proposals only; calibrated identity thresholds and competitor checks remain
separate. CCIP is retained as an alternative at its stricter `0.14`.

The style linker can attach a 2-20 crop subgroup to a larger DINO core only when
DINO and CCIP independently choose the same target with a clear margin. It requires
two supporting matches from distinct source panels and distinct target panels,
respects reviewed different-identity constraints, and never uses singleton crops.
After a strong link, the subgroup stops acting as a competing character; its crops
still cannot become evidence for another link. This anchor-only refinement joins
separate face, full-body, chibi, and rear-view groups without similarity chaining.
Exactly two-crop subgroups from two distinct panels can use a stricter anchor rule;
single crops and three-crop decorative lookalikes remain unclassified.
The exported evidence is marked `proposed_style_link`; it does not confirm an
identity or transfer a name. On the 169 manually inspected development crops,
the current linking and frequency policy increased grouped same-character pairs
from 1,854 to 3,740 while the count of labeled cross-character pairs remained zero.

After grouping, proposals occurring in fewer than 5% as many distinct panels as
the most frequent proposal are removed from the main group list. Their crops are
kept as unclassified records in `ungrouped_instance_ids` and `ungrouped.html`.
This filter counts panels rather than duplicate detector boxes.
The comparison is development evidence, not held-out calibration for automatic
identity acceptance. See `debug/reid_model_comparison/comparison.json` and the
accompanying development labels/contact sheets for the local comparison.

## 2. Understand the first result

The runner writes `output/<manga>/character_classification.json`, chapter-local
views, and a timestamped `reid_runs/` snapshot containing an HTML review gallery,
contact sheets, the reviews used, and an empty review template. Previous
classification exports are backed up in the new run directory before replacement.
The gallery links each crop to a saved copy of its original page and displays
pending form-change events when a description export is available.

`output/<manga>/character_identities.sqlite3` keeps resolution revisions, review
decisions, page/panel manifests, instances, entities, groups, and event evidence.
Removing a review changes the current resolution while retaining the old revision
for inspection. No previous decision is silently reused as current evidence.

There are two deliberately separate outputs:

- `visual_groups`: tentative lookalike groups, not confirmed people. Group IDs
  can change when their membership changes. Complete-link grouping prevents a
  chain of individually similar pairs from joining mutually dissimilar crops.
  Any simple/detail attachment is separately recorded in `style_links` with both
  model distances and margins.
  Single-crop groups are omitted. Their occurrence records have
  `visual_group_id: null` and remain listed in `ungrouped_instance_ids`.
- `characters`: one record per detected occurrence. `entity_id: null` means
  unknown. `identity_status` is `unknown`, `reviewed`, or `accepted_visual`.
  Only accepted identities enter the manga-wide `identities` list.

With no review file or validated calibration, the first run creates useful visual
proposals but **zero confirmed identities**. This is expected, not evidence of
measured classifier accuracy. A group must not be copied wholesale into narration.
The main gallery shows only groups with at least two crops. A separate
`ungrouped.html` archive retains single observations for later evidence/review.

## 3. Review references and keep IDs consistent

Open the printed gallery path. Inspect crops individually, checking their source
pages for context. Copy the exact `instance_id` values into
`output/<manga>/character_reviews.json`. Start with several clean, unambiguous
references from different panels for each recurring character and each form.
Choose permanent manga-wide IDs; never recycle an ID for a different character.

Example structure (replace the illustrative IDs and evidence):

```json
{
  "schema_version": 1,
  "assignments": [
    {
      "instance_id": "inst_COPY_EXACT_HUMAN_CROP_ID",
      "entity_id": "entity_001",
      "form_id": "entity_001_human",
      "accepted": true,
      "evidence": "Reviewed chapter 1 page 3 panel 2; clear reference."
    },
    {
      "instance_id": "inst_COPY_EXACT_DRAGON_CROP_ID",
      "entity_id": "entity_001",
      "form_id": "entity_001_dragon",
      "accepted": true,
      "evidence": "Reviewed the actual transformation sequence, chapter 2 pages 8-9."
    }
  ],
  "different": [],
  "names": [
    {
      "entity_id": "entity_001",
      "name": "Actual name from the manga",
      "source_instance_id": "inst_COPY_EXACT_HUMAN_CROP_ID",
      "accepted": true,
      "evidence": "Unambiguous self-introduction in the referenced panel."
    }
  ]
}
```

Same entity, different forms: share `entity_id`, use separate globally unique
`form_id` values, and document the reviewed bridge. Visual galleries remain
separate across forms. Never use this bridge for a summon, a possessed host, a
fusion, or a split unless an explicit identity decision really establishes it.

To forbid a mistaken association, add
`{"left":"inst_A","right":"inst_B","evidence":"Reviewed different beings"}`
to `different`. Contradictory accepted reviews fail with an error. Co-occurrence
alone is not treated as a separation: mirrors and transformation sequences exist.

Re-run after editing reviews. Remove/reject an assignment to revoke it; all
decisions are rebuilt from current evidence, not inherited from old output.
Changed image pixels or boxes invalidate the corresponding old instance IDs and
produce stale-review warnings. Legacy A/B overrides are intentionally not migrated
automatically: their chapter-local labels are unsafe as global identities.

## 4. Automatic propagation requires measured thresholds

Without calibration, only explicitly reviewed crops receive canonical IDs.
Automatic propagation additionally needs at least two clean reviewed references
from distinct panels other than the query panel, agreement with all retained
references in the chosen form, and a margin over competing candidates. Unreviewed
lookalikes outside the query's proposal group also count as competitors. Automatic
matches never become new reference prototypes, preventing self-reinforcing drift.
Even a single reviewed example of a competing character can veto a match, including
inside the same proposal group. Equal-distance ties are never accepted.

Pass a calibration file with `--calibration path.json`. It must contain:

- `encoder_fingerprint`: copied from this model's result, including preprocessing.
- `validated: true` and `evaluation_reference`: the held-out evaluation artifact.
- `max_distance` and `min_margin`: nonnegative measured thresholds, not invented
  confidence probabilities. Smaller distances mean more similar crops.

Do not set `validated` merely to switch automation on. Evaluate the full acceptance
rule on held-out manga panels/chapters with known same/different identities,
lookalikes, nonhuman forms, and bad detections. Prioritize wrong assignments to
major characters over coverage. No such benchmark or calibration is supplied by
this change; the loader checks metadata, not whether the claimed study is honest.
The proposal cutoffs (DINO distance 0.20, CCIP 0.14) are development heuristics,
not validated identity thresholds. Do not reuse a human-only calibration for
creature-heavy material without testing it.

## 5. Names, minor characters, and recap safety

Count distinct accepted story-panel appearances across all chapters and forms.
Multiple crops in one panel count once. At 1-3 panels, no name is released. At
4+, a reviewed source-grounded name becomes eligible; recurrence does not prove
identity or invent a name. An explicit `promote: true` on a reviewed name record
can override the recurrence minimum for an important brief appearance.
Identical imported page pixels and panel boxes count once across chapters, and
cannot supply two independent reference panels. This conservative rule can also
undercount deliberately repeated identical page artwork. Whole-page fallback
regions remain provisional and do not increase recurrence.

Canonical names belong to entities and stay consistent across chapters. Name
records must point to an accepted crop of that entity. The entity registry may
store the name, but occurrence records expose it only at/after its reviewed reveal
position (`naming_allowed`). Missing panel order prevents release. Keep aliases
out of the canonical-name list; conflicting canonical names fail validation.

For a masked identity or a form whose connection is revealed later, add
`identity_reveal_instance_id` to an accepted assignment. It identifies a crop at
the reveal position and delays naming for every occurrence of that entity/form.
A stale reveal ID disables naming until corrected. This is separate from the
name's own reveal position.

`narration_allowed` remains false: this stage cannot prove that a particular
visible character performed an action described in a panel. A later actor-grounding
stage must check that before using proper names. Unknown identity also does not
justify guessing gender; use a grounded descriptor, pronoun, or generic wording.

## 6. Transformation proposals from descriptions

If `output/<manga>_transformations.json` exists, its successfully extracted events
are attached as `transformation_proposals`, with possible source/target crop IDs
based on panel numbers. They always remain `pending_review`, regardless of model
confidence. Every crop in that panel is only a candidate, not a resolved actor.
Incomplete description pages are ignored for identity decisions. The current
description pass cannot infer cross-page transformations. Event files also lack
identity decisions: a human must still inspect source pages before accepting a
form bridge. New events include source-image hashes and panel-geometry revisions.
Only exact matches receive candidate instance IDs. Stale or legacy unverified
events remain visible with empty candidates; they never remap to changed pages.

`test_description.py` already writes `output/<manga>_transformations.json` alongside
the description JSON, using the same page-generation call. It checks expected
panel coverage against the manifest and preserves raw responses for inspection.

## 7. Creature detection and remaining limitations

There is no species filter. Body, person, and optional `entity_detections.json`
proposals all enter the same registry. An optional entity file is an object with
a `detections` list; each entry needs `page`, `box: [x1,y1,x2,y2]` in original page
pixels, `source_hash` (SHA-256 of the original image file), and `score` or
`detector_score`. It can contain goblins, dragons, animals,
or other acting beings. Tiny/weak/crowded/uncertain crops are preserved as unknown.

Removing a human-only filter does not teach existing detectors to find every
creature. Missed detections require improved proposals or detector training;
CCIP's creature discrimination is not established here either. New panel manifests
start with `page_role: "uncertain"`; existing roles survive only for unchanged source
images. Set `page_role` explicitly and `page_role_source: "review"` for reviewed
pages. Alternatively, run `test_description.py`: its grounded `page_role` results
can resolve new uncertain pages during ReID. Only `story` counts toward recurrence
or supplies automatic identity references; `reference`, `extra`, `cover`, `recap`,
and `uncertain` do not. Clean visual proposals can still group on those pages.
Panel ordering/geometry errors
still require correction upstream. This change does not implement the spec's
future universal detector, learned quality model, actor grounding, or benchmark.

Magi is not used. Existing checkpoint and training-data commercial permissions
still need a separate license audit; this implementation is not legal clearance.
Matching is currently quadratic in crop count; very large series will need indexed
candidate retrieval and bounded galleries. Snapshots/caches are not auto-deleted.

## 8. Reuse without classification JSON

Call `story_entities.pipeline.classify_manga(...)` with the manga/chapter paths,
proposal-output directory, an `Encoder`, cache directory, and optional in-memory
reviews/calibration/transformations. It returns `(result, crops, distances)`.
Only embedding-cache files are written there; the runner owns classification JSON
and review-gallery exports, plus the SQLite history. For staged callers,
`prepare_manga(...)` returns CPU-side evidence and `resolve_prepared(...)` replays
decisions without loading any model. The public API validates calibration metadata
just like the CLI. `main.py` remains outside this standalone identity workflow.

Regression checks: `python -X utf8 -m unittest discover -s tests -v`.
