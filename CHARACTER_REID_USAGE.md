# Character identity runner

`test_character_reid.py` now performs manga-wide, precision-first visual matching.
It does not load Qwen or automatically equate a creature with a human because a
description suggests a transformation. It gives each retained visual group a
stable fallback display name. The reusable Python implementation is in
`story_entities/`; JSON export belongs to the runner.

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
form-change evidence and its pending or automatically accepted status when a
description export is available.

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
  unknown. `identity_status` includes `unknown`, `reviewed`, `accepted_visual`,
  and `accepted_automatic`.
  Only accepted identities enter the manga-wide `identities` list.

With no review file or validated calibration, broad visual groups remain proposals
and are never copied wholesale into narration. A recurrent subset can now receive
`accepted_automatic` only after representation typing removes toys/images/statues,
at least three independent panels remain, grounded appearance traits agree, every
member has two independent visual neighbours, and DINO plus CCIP agree when both
encoders are available. Unsupported satellites remain unknown.
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

Calibration-based propagation additionally needs at least two clean reviewed references
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

Automatic OCR-derived naming is disabled. Each retained visual group receives a
deterministic pseudorandom display name from `story_entities/manga_names.json`.
The pool contains 200 common Japanese surnames, which are gender-neutral and
natural character references in manga. Ungrouped observations receive no display
name. These additive `display_name` fields do not replace reviewed canonical
entity names. When several retained visual groups are already assigned to the
same accepted entity, such as reviewed human and creature forms, they share one
entity-scoped display name. Pending transformation proposals never join forms or
share names by themselves.

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

Generic `narration_allowed` remains false because an ungrounded panel description
does not prove which visible character acted. New panel runs also create a
Qwen-only preview with request-local `C1`, `C2`, etc. labels. Qwen returns compact
actions using those labels; ReID maps their durable detection IDs to exact
instances and exports the result under `story_events`. Numerical model confidence
is not requested or accepted as evidence. An action receives
`status: accepted_automatic` and policy `grounded_visible_action_v1` only when
its source revisions are current, actor/target resolve exactly in the same panel,
both endpoints are narration-eligible, its phrase passes semantic validation,
and the panel description independently contains the same visible action verb.
Other proposals remain pending or explicitly ungrounded and are ignored by the
writer. Optional source-bound records in `character_reviews.json.reviewed_actions`
receive `status: accepted_reviewed` and policy `reviewed_visible_action_v1`. Unknown
identity still does not justify guessing gender; use the grounded stable
description or generic wording.

The description stage also exports exact-text proposals. Structurally valid text
from a complete, revision-bound page is accepted automatically under
`grounded_text_transcription_v1`; an attributed speaker additionally requires one
exact same-panel instance. The writer treats this as a transcription only, never
as proof that a spoken claim is objectively true. Invalid or incompletely grounded
records remain pending. A reviewed correction can still be supplied under
`character_reviews.json.reviewed_text_facts`:

```json
{
  "status": "accepted_reviewed",
  "fact_id": "text_ch1_p3_panel2_1",
  "panel_id": "panel_COPY_FROM_PROPOSAL",
  "kind": "dialogue",
  "text": "Exact wording checked against the page.",
  "speaker_instance_id": null,
  "truth_status": "statement_only",
  "source_hash": "COPY_FROM_PROPOSAL",
  "geometry_revision": "COPY_FROM_PROPOSAL",
  "grounding_revision": "COPY_FROM_PROPOSAL"
}
```

Allowed kinds are `dialogue`, `thought`, `narration`, `sign`, `sound_effect`, and
`publication` (series/chapter titles, author credits and promotional copy).
Publication text is retained for audit but omitted from story-generation input.
Use the proposed `candidate_speaker_instance_id` only after
visually checking the bubble tail; otherwise keep the speaker null. Changed page
pixels, panel geometry, or grounding invalidate the reviewed fact rather than
silently carrying dialogue into a different panel.

## 6. Transformation proposals from descriptions

If the combined description JSON contains `transformations`, its successfully
extracted events are attached as `transformation_proposals`. New runs resolve
source/target `C` labels through durable detection IDs; legacy exports fall back
to crop candidates based on panel numbers. A transformation, reversion, or disguise
reveal can automatically connect two forms only when Qwen reports exactly 100%
certainty and direct visible-change evidence, the panels are adjacent, hashes and
geometry are current, each labeled endpoint resolves to one high-quality instance,
both are story panels, and no accepted identity or reviewed-different constraint
conflicts. Accepted events use `status: accepted_automatic`; every failed gate
keeps `status: pending_review` and records `automatic_rejection_reason`.

Possession, summoning, splitting, and fusion never establish same identity.
Incomplete pages, text-only evidence, ambiguous sequences, non-adjacent panels,
and stale or legacy events cannot affect identities. The current description
pass still cannot infer cross-page transformations. Qwen's 100% value is a
strict application gate, not a guarantee that the model is objectively correct.

`test_description.py` writes transformations inside the same description JSON,
using one page-generation call. It checks expected panel coverage against
the manifest and preserves raw responses for inspection. New events include
`confidence_percent`; older exports without it remain pending until descriptions
are regenerated.

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

## 9. Story-evidence review extensions

The existing manga-scoped `output/<manga>/character_reviews.json` is also the
single lightweight story-evidence sidecar. Optional additive lists are:

- `eligibility`: exact `instance_id`, source/geometry/grounding revisions,
  `decision` (`eligible`, `uncertain`, or `rejected_non_character`), and reason;
- `appearance_corrections`: exact current instance/revisions, a reviewed short
  visible description, and `status: accepted_reviewed`;
- `reviewed_actions`: exact panel/actor/optional target, current revisions,
  visible action text, `fact_id`, and `status: accepted_reviewed`;
- `reviewed_transformations`: explicit source and target instance/panel
  endpoints with each endpoint's source hash and revisions, evidence, type,
  `fact_id`, and `status: accepted_reviewed`; cross-page endpoints are allowed;
- `panel_descriptions`, `reviewed_text_facts`, and `reviewed_narration`: exact
  `panel_id`, current source/geometry/grounding revisions, `fact_id`, text, and
  `status: accepted_reviewed`;
- `unresolved_critical_items`: records with `status: unresolved`; any such
  record blocks story publication.

Reviewed text kinds are `narration`, `dialogue`, `thought`, `sign`, `sound_effect`,
and `publication`. A speaker is optional, but when present must be a current,
narration-eligible instance in that panel. Dialogue remains a statement unless
its separately reviewed `truth_status` permits a stronger interpretation.

Panel-stage geometry/order/role corrections live beside each chapter as
`panel_review_corrections.json`. Each accepted entry is bound to `source_hash`
and the detector's `base_geometry_revision`, supplies ordered boxes (or an
explicit `reading_order`), and may supply a reviewed `page_role`. The detector
panels remain in the manifest for audit, while corrected geometry creates new
panel IDs/revisions and invalidates only affected downstream evidence.
