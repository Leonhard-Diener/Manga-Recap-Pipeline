# Character Identity Pipeline for Manga Recap Videos

Status: implementation specification. The standalone identity baseline now
implements cached visual proposals, reviewed entity/form assignments, conservative
propagation, recurrence/name/reveal gates, revision history, and review exports.
See CHARACTER_REID_USAGE.md for the supported commands and current limitations.
The broader plan below still includes future work: validated creature detection,
cross-page grounded actions, automatic-name benchmarks, and recap/video export.
`main.py` is intentionally outside the current identity work.

## 1. What success means

The recap should describe what happens without giving an action to the wrong
character. **An unknown identity is preferable to a wrong identity.**

There are two different priorities:

- Detection should find as many plausible characters as practical, including
  dragons, goblins, demons, animals, robots, and partial bodies.
- Identity assignment and naming should accept only well-supported matches.
  Uncertain observations remain anonymous.

The system must keep identities consistent across the available chapters of a
manga. Characters appearing in only one to three distinct story panels normally
receive no name. They still remain in the records because they may matter to a
transformation or become recurring characters later.

Names are optional even for main characters. Reaching four appearances makes a
character eligible for naming; it does not prove their identity or their name.

This is a conservative build plan, not a claim that any selected model achieves
the required accuracy. Models and thresholds must earn their use on held-out
manga examples.

## 2. Core concepts

| Concept | Meaning | Example |
| --- | --- | --- |
| Instance | One depicted individual in a panel, with a persistent ID and location | A dragon at the top of panel 4 |
| View | A crop of that instance | Whole body or visible head |
| Visual group | A provisional group of similar instances | Several views of a black dragon |
| Form | A recurring appearance of an entity | Human form or dragon form |
| Entity | The underlying story character | The character who has both forms |
| Claim | An observation or proposed relation, with its source evidence | Instance A may transform into B |
| Decision | Accepted, rejected, or pending interpretation of evidence | A and B belong to the same entity |
| Name record | A supported name, alias, and reveal position | A name introduced in chapter 3 |

A visual group is a hypothesis, not a verified identity. A form may have several
visual groups when poses or drawing styles differ substantially.

Keep these properties separate:

- Identity: unknown, proposed, or accepted.
- Recurrence: minor or recurring.
- Role: individual, group, decorative, or uncertain.
- Name: unknown, proposed, or accepted.

An individual can have an accepted identity but no name, or be recurring while
their identity is still unresolved.

Example:

```text
Entity E17
  Human form -> visual groups G2 and G6
  Dragon form -> visual group G9
  Accepted transformation -> evidence connecting a human instance to a dragon instance
```

A clothing change does not automatically require a new form. Keep a stable form
when the visual evidence supports it; create another when useful for matching.
Never force human and dragon images to share an embedding cluster.

## 3. End-to-end flow and model schedule

```text
Preserved source pages and chapter manifest
  -> panels, reading order, broad entity proposals, optional OCR
  -> clean crops, cached embeddings, provisional visual groups
  -> one description session with Qwen across all pages
       descriptions + grounded action records + event/name claims
  -> conservative entity resolution across all available chapters
  -> recurrence counts and supported names
  -> narration records with per-action identity checks
  -> recap script, source-panel selection, and video

Optional review decisions feed back into entity resolution.
Unreviewed claims remain pending; they do not block a generic recap.
```

Run detector/encoder work in batches before loading Qwen. Release those GPU
sessions before the description stage if memory is tight. Resolve identities
afterward using cached records and embeddings; do not reload Qwen for every
candidate pair, chapter, name change, or review decision.

Normal operation uses one Qwen request per page in one loaded session. The
request can include a small amount of adjacent-page visual context. Targeted
retries are exceptional, budgeted, and kept in the same session when possible.

The existing cleanup stage also loads Qwen separately. To achieve the proposed
single-session schedule, replace destructive cleanup with conservative page
selection and classify page role during the description request. Obvious extras
can be marked by manual rules before processing; uncertain pages remain included.
This may spend cheap detector work on a few extras, but avoids another mandatory
VLM load. Standalone cleanup can remain an explicitly separate legacy workflow.

## 4. Stage A: preserve pages and establish stable coordinates

Create a manifest before analysis:

- Manga ID, chapter IDs, source/edition, and explicit chapter order.
- Persistent page IDs, file hashes, dimensions, and source paths.
- Reading direction and whether the input is a page, spread, or vertical strip.
- Page role: story, reference/profile, extra, or uncertain.
- Panel IDs, original-page boxes, reading order, and ordering uncertainty.

Do not delete pages classified as extras. A false cleanup decision could remove
a transformation or name introduction. Character profiles may be useful naming
evidence even when excluded from recap scenes and appearance counts.

Preserve originals. All crops and overlays are derived assets. Record how resized,
tiled, or spread-split coordinates map back to the original page.

Do not silently discard pages with no detected panels. Supply a whole-page
fallback region marked as provisional. Uncertain reading order must weaken
continuity evidence, especially around scene cuts and spreads.

Use IDs assigned once and stored in the database. Filenames like 00001.jpg repeat
across chapters; array positions and cluster labels change on reruns. Neither is
a globally reliable identity.

## 5. Stage B: find beings, including non-human characters

### Detection strategy

Use a broad entity detector whose job is to locate depicted individuals,
including non-human beings. Species is optional descriptive metadata, never a
condition for inclusion. Animated objects can also be story characters.

Multiple detector proposal sources are allowed, but normalize them into one
instance registry. Descriptions, embeddings, reviews, and exports must use this
same registry.

Start by measuring the current panel detector's body proposals and the existing
person detector on annotated non-human examples. Retain only useful proposal
sources. Removing a human filter or renaming a detector class does not teach the
model to detect dragons.

When recall is inadequate, fine-tune a commercial-compatible detector with
explicit examples of creatures, partial bodies, profiles, chibi drawings, and
transformations. A practical training target is:

- Full visible individual: human, humanoid, creature, animal, robot, or other.
- Optional head/face region, attached to the same individual.

A head and its body must become two views of one instance, not two characters.
Do not infer that every head inside a body box belongs to that body when figures
overlap.

### Proposal processing

1. Run whole-page detection.
2. Run panel or tile fallback only where evaluation or low coverage indicates
   missing/small entities. Avoid running every detector on every crop.
3. Map all boxes to original-page coordinates.
4. Deduplicate overlapping proposals for the same depiction. Do not merge
   genuinely different overlapping beings simply because their boxes overlap.
5. Assign panel membership. A border-crossing depiction gets one instance and a
   primary panel; genuinely repeated drawings get separate instances.
6. Record crop quality and role uncertainty.

Remove invalid boxes and clearly duplicate detections. Keep plausible tiny,
occluded, or partial instances, but mark them ineligible to seed identity groups.
Replace the current fixed 5% whole-page area cutoff with quality checks based on
actual crop detail and panel context. A small figure in a small panel may matter.

Background is not a permanent identity class. Retain individual candidates where
feasible; use a group record for an unresolved crowd. A recurring creature must
not be discarded because it looks like an enemy or has not spoken.

### Detection failure fallback

Qwen may report an unmatched being, with a panel ID and local description.
Preserve that as an unlocalized observation. Do not fabricate a trusted box or
silently attach it to the nearest existing candidate.

Create a targeted redetection/manual-box task. Until resolved, its action can be
narrated generically. A Qwen warning is a recovery opportunity, not proof that the
detector has complete coverage.

## 6. Stage C: cache visual evidence and build provisional groups

Use original, unannotated pixels for embeddings. Do not embed ID labels, panel
numbers, decorative borders, or description text.

Begin with one encoder and these views:

- A tight whole-entity crop, when it contains one distinguishable individual.
- A head crop, when reliably detected and sufficiently detailed.

Keep panel context for Qwen and review; do not use shared scene backgrounds as
primary identity evidence. Additional torso crops, segmentation, or a second
encoder should be added only if measured errors justify their cost.

Mark quality per view. A partial body can have an excellent head crop. Missing
faces must not penalize dragons or other creatures simply for being non-human.

For a small dataset such as the current 128 detections, exact pairwise similarity
is simpler and cheap. Add approximate neighbour search only when dataset size
requires it. Search all available chapters, not just nearby pages.

Build conservative visual groups from eligible views using deterministic,
complete-linkage-style checks: every required cross-group comparison must pass
the calibrated visual threshold before a merge. This trades some fragmentation
for fewer mixed groups. Human constraints and competing matches can still veto
a merge.

A similarity score is not a probability. Calibrate thresholds on labeled manga
pairs, including similar-looking different characters. Do not copy a threshold
from anime illustrations or assume it transfers between checkpoints.

Do not automatically add weakly matched instances as prototypes. Preserve their
candidate matches separately so they cannot gradually corrupt a group. A
singleton without enough evidence remains unresolved.

Keep matching reproducible: compare like views (head-to-head, body-to-body), use
the checkpoint's documented preprocessing, and record which views supplied the
score. Do not average incomparable scores from different encoders or silently
substitute a missing face with a zero score. A body-only creature may qualify
under a separately validated body-only rule. When face and body evidence strongly
disagree, abstain rather than average away the disagreement.

## 7. Stage D: Qwen describes pages and records evidence

### Inputs

For each target page, supply:

- A readable page image with unobtrusive panel and short instance labels.
- A short mapping from prompt labels to stored instance IDs.
- Explicit panel order and page boundaries.
- Relevant previous-page boundary panels when needed for continuity.
- Optional OCR boxes/text, with uncertainty and source identifiers.

For the initial implementation, use a bounded previous-page overlap so a
transformation beginning on one page and ending on the next is visible in one
request. Emit new action records only for the target page; context panels may be
referenced as evidence. This prevents duplicated descriptions and counts.

Reset context across unrelated chapters or editions. Adjacent chapters with
continuous source order may share boundary context. If a transition begins but
does not finish, store a pending claim for subsequent evidence.

Start without a generated story-memory ledger. Raw neighboring images and
validated observations avoid a feedback loop where Qwen treats its own mistaken
identity guess as established truth. Later retrieval, if needed, must clearly
separate accepted facts from unresolved hypotheses and must not supply hidden
future names or identities.

Do not put entity names or guessed cluster identities in image labels. Use P1,
C1, C2, etc. Store the label mapping with the request. Keep labels legible without
covering faces or dialogue. Dense pages may need a bounded crop-based fallback.

### Output contract

Require compact JSON with:

- Target page ID and page role.
- One result per expected panel, including empty or unreadable panels.
- Brief setting and visible action records.
- Actor and target references to observed instance IDs, or null.
- Unmatched beings.
- Event claims with endpoint IDs, evidence references, and uncertainty.
- Name mentions, with readable source text and any uncertain speaker/addressee.

Use one description/action representation; do not ask for duplicate long prose,
a full transcript, identity biographies, and detailed attribute catalogs in every
request. Empty event/name arrays are valid and common.

Qwen should describe observed actions and propose relations. It does not assign
final canonical names.

Example response using request-local IDs; the adapter expands these through the
stored mapping:

```json
{
  "page_id": "ch02_p014",
  "page_role": "story",
  "panels": [
    {
      "panel_id": "P1",
      "setting": "A forest clearing",
      "actions": [
        {
          "actor": "C1",
          "target": null,
          "text": "is engulfed in smoke"
        }
      ]
    },
    {
      "panel_id": "P2",
      "setting": "The clearing",
      "actions": [
        {
          "actor": "C2",
          "target": null,
          "text": "spreads its wings"
        }
      ]
    }
  ],
  "event_claims": [
    {
      "type": "possible_transformation",
      "source": "C1",
      "target": "C2",
      "evidence_panels": ["P1", "P2"],
      "observation": "A dragon appears after smoke obscures the humanoid.",
      "alternative": "The dragon could be a separate arrival."
    }
  ],
  "name_mentions": [],
  "unmatched_beings": []
}
```

This example must remain pending: the pictured sequence alone does not establish
identity.

### Validation and generation limits

Validate JSON syntax, required fields, expected panel coverage, referenced IDs,
and whether referenced instances belong to the supplied images. Valid JSON does
not prove that the model chose the right actor. Ambiguous localization must leave
the actor unassigned or require review before named narration.

The current 48-token whole-page output limit is unsuitable for this schema.
Choose a bounded output budget based on panel and candidate count, measured on
representative pages. Detect output truncation; never treat a cut-off result as a
complete page or manufacture missing records.

Keep thinking disabled initially, consistent with the current runtime goal.
Schema-constrained decoding can help if the chosen runtime supports it; always
validate afterward. At most one targeted retry per failed page in the initial
design. If it fails, save raw output and mark the affected fields unavailable.
Only independently validated records may proceed.

Do not assume an invalid structured response contains a usable description.
Missing descriptions should be flagged for review or omitted from export, not
invented by a fallback template.

The extra context and output increase inference time even though model loading
is shared. Measure that increase; do not promise it is negligible.

## 8. Stage E: resolve identities conservatively

### Evidence rules

Store evidence and decisions separately. A model statement, its confidence
number, and another wording of that statement are one source, not three
independent confirmations. OCR and a VLM reading the same ambiguous bubble may
also be correlated.

| Evidence | Permitted use |
| --- | --- |
| Clean visual agreement in a validated domain | Propose or accept same-form links under calibrated rules |
| Nearest visual neighbour alone | Candidate retrieval only |
| Appearance in adjacent panels | Context hint, never sufficient identity proof |
| Qwen transformation statement | Pending story claim |
| Human-confirmed transformation with identified endpoints | Accept cross-form entity link after conflict checks |
| Explicit name text with supported referent | Proposed name attribution; use naming checks below |
| Human-confirmed different individuals | Hard separation constraint |
| Co-occurrence in one panel | Possible separation evidence, with exceptions |

Co-occurrence is not automatically a hard separation rule. One panel can contain
a reflection, inset flashback, motion sequence, or both ends of a transformation.
Likewise, two similar simultaneous figures can be separate clones. Require
depicted distinct individuals or human confirmation before creating a hard
separation constraint.

For version 1, automatic cross-form transformation merges are disabled.
Review can accept them; unattended runs keep the forms separate and use generic
narration. This is deliberate given the cost of falsely naming a protagonist.
A future automatic rule must first pass a held-out transformation benchmark.

A summon, possession, fusion, or split never automatically means equality of
identities. Preserve its event type and endpoints. Defer complex consciousness
tracking and fusion identity semantics to later versions.

### Implementable resolver

A database of records and typed edges is sufficient; no graph server or complex
global optimizer is required initially.

1. Load valid evidence from all available chapters and accepted human decisions.
2. Build/rebuild conservative visual groups from eligible instances.
3. For each group or instance, retrieve competing entity candidates, including
   candidates outside the recent scene. Include an explicit unknown option.
4. Accept a visual assignment only if absolute quality and similarity gates pass,
   the best candidate has a calibrated margin over competitors, enough validated
   support exists, and no separation/name conflict exists. No competitor does
   not mean automatic acceptance.
5. Keep low-quality and ambiguous attachments proposed. Proposed attachments
   cannot seed further assignments or contribute to accepted appearance counts.
6. Apply reviewed cross-form links to entities, preserving each form's visual
   groups.
7. Before each entity merge, check constraints across both complete memberships,
   not just the two examples that triggered the proposal. Distinct accepted names
   or conflicting evidence require review.
8. Persist a new resolution revision with the supporting decision IDs.
9. If evidence is later rejected, replay affected decisions and rebuild the
   affected groups; invalidate dependent assignments and names.

Do not implement identity resolution as unguarded union-find over positive
edges. Even an accepted merge must be reversible.

Most importantly, **a trusted entity does not automatically make every attached
instance trustworthy**. Track instance assignment status separately. A dubious
crop near a known protagonist must still be narrated anonymously.

### Configuration needed before automatic assignment

Store separate settings for view-quality gates, visual-group similarity,
entity-assignment similarity, and the best-versus-second-candidate margin. Select
them on development pairs at the desired accepted precision, then freeze them
for held-out evaluation. These thresholds are not interchangeable.

Use two clean supporting examples from distinct panels as the initial minimum
for an automatic assignment to an existing entity. Compare against all selected
representatives and apply the competing-candidate and conflict checks above;
two examples alone are not proof. A new visual group starts as provisional and
must pass the same purity checks before its members can be accepted. Human
confirmation can establish an identity from fewer examples with recorded evidence.

If a view/domain rule has not been validated (for example, body-only matching of
dragons), its matches remain proposals. Do not assign an arbitrary numeric
threshold to make the pipeline appear complete. Confirmed local exemplars can
support later calibration without treating unreviewed matches as training truth.

## 9. Stage F: recurrence and consistent naming

### Counting appearances

Use this exact count:

```text
appearance_count(entity) =
  number of distinct accepted story-panel IDs
  containing an accepted assignment to that entity
```

Consequences:

- Two boxes for one character in one panel count once.
- Two genuine panels on the same page count twice.
- Head/body views and duplicate detector boxes do not add appearances.
- Confirmed human and dragon forms contribute to the same entity's count.
- Proposed assignments, profile pages, covers, and repeated processing do not count.
- Duplicate editions/imported copies of the same page must be reconciled before
  counting. Separate intentional flashback panels can count.
- Uncertain page/panel membership does not increase the accepted count.

With 1-3 accepted panel appearances, use a local descriptor and no name by
default. At 4+, mark the entity recurring and eligible for naming. A review
override can promote a story-critical minor character, but is not mandatory.

Unknown observations are not all one unknown entity. Give them separate local
references; never aggregate unrelated unknowns until they qualify as recurring.

### Name attribution

Separate a name being mentioned from a character being named:

- A speaker saying "Rin!" may be addressing Rin, not introducing themselves.
- A bubble mentioning an absent character does not name someone visible.
- Speech bubble proximity alone does not establish speaker identity.
- Titles, aliases, disguises, and shared names do not prove entity equality.

Version 1 may store automatic name proposals, but release proper names through
reviewed, source-grounded name records. This one-time decision per recurring
character is much cheaper than reviewing every appearance. If there is no review,
use consistent descriptors. Future automatic naming requires its own precision
evaluation.

Use permanent manga-scoped entity IDs. Choose one accepted display name and
store aliases with evidence, language, and reveal position. Never generate names
such as Ren or Akira as placeholders. Internal labels such as E17 or Character 01
are suitable for review, not normal spoken narration.

When identities merge, preserve a canonical ID and redirect aliases to it.
When a bad merge is undone, restore original member records; do not leave all
their instances attached to the old canonical ID.

## 10. Stage G: safe narration and video grounding

Store actions against instance IDs, then resolve the permitted wording at export.
Do not put guessed names directly into stored descriptions and later use global
string replacement.

A proper name can be used for an action only if all checks pass:

1. The action's actor localization is sufficiently clear.
2. That particular instance has an accepted entity assignment.
3. The entity is naming-eligible or explicitly promoted.
4. The proper-name attribution is accepted.
5. No unresolved conflict affects this assignment or action.
6. The chosen recap policy permits revealing that name or identity at this point.

By default, preserve story reveal order. The database may know that an earlier
masked figure is the protagonist, while narration still says "the masked figure"
until the reveal. Store both the event's story position and the evidence/reveal
position; reading order is not always chronological time.

Safe alternatives:

| Available evidence | Wording |
| --- | --- |
| Accepted actor, entity, and name | The supported proper name |
| Accepted entity with no usable name | A stable, locally accurate descriptor |
| Uncertain entity but clear actor | "a goblin," "the winged creature," or "the masked figure" |
| Unclear actor | Describe the visible event without assigning an actor, or flag it |
| Pending transformation | Describe the observed change/arrival without asserting shared identity |

"The transformed creature" itself asserts a transformation: use it only when
that event is accepted. Pronouns need a clear antecedent and supported identity
or gender context; pronouns can misattribute an action just as names can.

Each recap claim must retain its source page/panel IDs and decision references.
The video editor uses those same panel IDs when selecting images. Reject or flag
script claims that introduce an unsupported name, actor, or transformation.
Name whitelisting alone is insufficient: a permitted name can still be attached
to the wrong action.

Freeze each exported script/video to a resolution revision. Later corrections
can generate a new export without silently altering an already approved one.

## 11. Persistence and data contracts

Use a manga-scoped SQLite database initially, plus files for images, embeddings,
raw Qwen responses, and JSON exports. This supports consistent IDs and reversible
decisions without introducing a separate database service.

Minimum records:

| Record | Required fields |
| --- | --- |
| Page/panel | Stable IDs, chapter/order, source hash, box, page role, geometry revision |
| Instance | ID, page/panel association, original-pixel box, proposal source, quality flags, depiction role |
| Embedding | Instance/view ID, model hash, preprocessing version, vector location |
| Action | ID, panel ID, actor/target instance IDs or null, text, grounding status, source response |
| Claim | ID, type, endpoint IDs or null, evidence refs, alternatives, proposed/rejected/accepted status |
| Assignment | Instance ID, entity/form ID, proposed/accepted status, supporting decision IDs |
| Entity/form | Stable IDs, membership revision, representative instances, aliases |
| Name | Entity ID, text, language, referent evidence, acceptance status, reveal position |
| Decision | ID, reviewer or calibrated rule version, inputs, outcome, superseded decision if any |
| Run/export | Input/model/prompt/schema versions, resolution revision, completion state |

Coordinates use original-page pixels in [x1, y1, x2, y2] order with positive area
and bounds validation. Valid records must not use zero-size example boxes.

Prompt-local labels are short aliases only. Responses are tied to the exact
page/instance mapping revision used for inference. If detection changes, reconcile
old and new instances conservatively using source page, geometry, and crop
evidence. Ambiguous remappings invalidate affected claims and require review;
do not silently transfer a name to a different new box.

Save validated results atomically after each page and decision batch. Resume
completed pages only when their cache key matches. Include source images,
candidate mapping, context images, model, prompt, schema, and preprocessing
versions in the appropriate cache keys.

Detection changes invalidate crops and grounding; embedding changes invalidate
visual assignments; a corrected name normally only requires resolution/export.
Use run-specific debug folders so contradictory sheets from different runs cannot
be mistaken for current results.

## 12. Runtime and model choices

Start with the smallest useful system:

- One broad detector/proposal adapter.
- One fast image encoder.
- Optional OCR when needed and validated for the manga's language.
- Existing Qwen description model, loaded once for that stage.
- CPU-side resolution, SQLite persistence, and a simple review interface.

DINOv2 is a candidate baseline, not a guaranteed manga identity model. CCIP or
the existing DINO export can remain comparison candidates only after checking
their exact checkpoint provenance and use rights. Do not mandate an ensemble
without demonstrating a meaningful accuracy gain.

Batch detector/embedding inference, decode each page once per stage where
practical, and cache reusable results. Record the actual ONNX execution provider:
an installed GPU package does not prove inference ran on GPU.

Measure cold model load, image preprocessing, inference, decoding, persistence,
peak memory, total chapter time, and median/slow-page latency separately.
Generation-only timing hides important costs.

Benchmark at least:

1. Existing descriptions only.
2. Descriptions plus the proposed compact event schema.
3. The same schema with bounded previous-page context.

Measure structured-output failure and missed-panel rates as well as time. Test
quantization or alternative runtimes only after a correctness baseline exists.
A 2B model may not reliably ground complex transformations; if it fails, keep
those events unresolved. A stronger model is a later measured option, not a
promise that loading a larger model fixes the architecture.

## 13. Commercial component selection

Magi is excluded under the stated commercial-use requirement.

Record exact code and weight licenses, revisions, and training-data provenance
for chosen components. A permissive repository does not automatically establish
the terms of a third-party fine-tuned checkpoint. An ONNX conversion does not
change those terms.

Useful starting references:

- [Official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR):
  Apache-2.0 code and custom-data training resources. Assess each chosen
  checkpoint separately, including the current third-party Manga109-s export.
- [Official DINOv2 repository](https://github.com/facebookresearch/dinov2):
  original DINOv2 code and model weights are published under Apache 2.0.
- Local Qwen license: models/Qwen3.5-2B/LICENSE and its model card.
- [Manga109/Manga109-s terms](https://manga109.github.io/manga109-project-website/en/):
  Manga109-s permits commercial use of machine-learning results under conditions;
  regular Manga109 is academic-use restricted. Observe dataset access,
  attribution, and redistribution terms.

These references identify candidates; they do not approve every derivative.
Training-data permission and permission to publish manga imagery in recap videos
are separate matters.

## 14. Evaluation and acceptance tests

Start with roughly 30 annotated pages as a development smoke test, including
the current failure examples. Expand beyond that before judging production
accuracy. Keep threshold tuning and final testing separate, preferably split
by manga title/artist and chapter, not random near-duplicate crops.

Include non-human leads, transformation sequences across page/chapter boundaries,
similar protagonists, summons, reflections, identical uniforms, tiny crops,
overlapping figures, unknown names, and deliberate false name references.
Do not train on final evaluation pages.

Evaluate detection and identity separately, then evaluate end-to-end narration:

| Measure | What it prevents |
| --- | --- |
| Detection recall by human/non-human/size category | Hiding missed dragon or goblin protagonists in an overall score |
| Accepted assignment precision and named-action accuracy | Giving a main character's action to someone else |
| Coverage/abstention rate | Claiming perfect precision by classifying nothing |
| False merges and false splits | Mixed characters versus harmless fragmentation |
| Accepted transformation/name precision | Turning unsupported story guesses into facts |
| Actor grounding accuracy | Correct identity attached to the wrong pictured action |
| Duplicate counting and ID consistency | Unstable recurrence thresholds and cross-chapter labels |
| Total latency, peak memory, retries | An accurate but unusably slow workflow |

Initial release rule: zero known wrong named-action attributions on the held-out
regression set, together with reported accepted coverage and sample counts.
Zero observed errors on a small set is not a guarantee of production accuracy.
Any named-main-character swap blocks release until fixed or safely abstained.

Mandatory scenario tests:

1. A dragon protagonist with no human face is retained and described.
2. A human followed by a dragon in smoke produces a pending event, not an auto-merge.
3. A reviewed transformation links the two forms without merging a different dragon.
4. Two similar main characters remain separate; ambiguous crops receive no name.
5. A reflection or depicted transformation inside one panel does not create a false hard separation.
6. An accepted entity has one dubious crop; that crop still receives generic wording.
7. "Rin!" in a speaker's bubble does not automatically name the speaker Rin.
8. Three accepted appearances remain unnamed; a fourth qualifies only the supported entity.
9. Head/body duplicates, duplicate imports, and reruns do not increase the count.
10. A mistaken bridge is revoked; dependent assignments/names disappear on replay.
11. A page-boundary transformation can reference prior context without double-counting it.
12. Truncated JSON, missing panels, or unknown IDs cannot silently enter named narration.
13. A rejected cleanup decision can be reversed because original pages still exist.
14. An accepted late identity reveal does not spoil earlier narration by default.
15. With no reviewer available, the run finishes with safe descriptors and pending claims.

## 15. Implementation order and repository mapping

Complete each phase before expanding its model complexity.

| Phase | Work | Done when |
| --- | --- | --- |
| 1. Foundation | Manifest, stable IDs, schemas, SQLite decisions, source preservation, development fixtures | The same inputs retain IDs and bad decisions can be replayed/revoked |
| 2. Detection | Normalize current proposals, evaluate creatures, train/fine-tune if necessary, deduplicate heads/bodies | Non-human recall and crop failures are measured and recoverable |
| 3. Visual baseline | One cached encoder, conservative grouping, unknown assignments | Similar protagonists do not merge on the regression cases |
| 4. Description | Compact Qwen schema, boundary context, ID validation, checkpoints | Actions/events are grounded and incomplete output is rejected |
| 5. Resolution | Conflict checks, reviewed form bridges, per-instance acceptance, recurrence/name records | Chapter consistency, minor-character policy, and bad-merge rollback pass |
| 6. Recap export | Evidence-linked wording, reveal policy, source-panel mapping | Every named action passes the export gate |
| 7. Improvement | Additional training, better recall, optional automation, latency tuning | Each change improves measured quality or cost without regressing named accuracy |

Suggested module responsibilities, not files to create automatically:

```text
story_entities/
  schema.py       record validation and enums
  storage.py      persistent IDs, revisions, cache/dependency metadata
  detection.py    proposal adapters, deduplication, candidate quality
  embeddings.py   clean views, batching, visual groups
  resolver.py     assignments, constraints, merges and replay
  naming.py       appearance counts, names, aliases and reveal positions
  review.py       pending claims and human decisions
  export.py       narration permissions and source references
description/
  describer.py    model lifecycle and grounded page requests
  validation.py   response schema, ID mapping and completeness checks
```

Changes required later in the existing repository:

- main.py: orchestrate stages for all selected chapters with explicit model
  lifetimes and resume behavior. "Manga-wide" means all available chapters,
  not waiting for an ongoing manga to finish publication.
- cleanup/classifier.py: replace deletion with reversible page-role metadata;
  avoid a mandatory separate Qwen load in the new normal workflow.
- panel_extractor/: persist panel geometry and reading order; preserve fallback
  pages and collect reusable entity proposals where useful.
- test_people_detection.py: replace the human-only output as the authority for
  ReID; reuse useful proposals through the shared adapter.
- test_character_reid*.py: keep comparative experiments if useful, but remove
  independent naming and Qwen pairwise verification from the production path.
- description/describer.py and config.py: replace plain-text-only output and
  the 48-token limit with validated, budgeted records.
- test_description.py: remain a thin entry point into the shared description
  implementation; avoid duplicating model logic in the runner.
- Add focused regression tests for the acceptance scenarios, separate from
  scripts whose test_ names merely launch real inference.

## 16. Deliberately deferred complexity

Do not require these to deliver the first useful version:

- A graph database, approximate search for tiny datasets, or a complex optimizer.
- A second mandatory embedding model or four embeddings for every crop.
- Automatic cross-form merging or automatic proper-name attribution.
- A Qwen comparison loop, unlimited retries, or unbounded chapter summaries.
- Face detection as a prerequisite for every character.
- Complete crowd tracking or automatic interpretation of fusion/possession.
- A large review application: initial contact sheets plus recorded decisions suffice.

The system remains useful without these features because unresolved identities
can still produce accurate, readable recap narration.
