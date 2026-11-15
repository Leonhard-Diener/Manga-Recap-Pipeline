# Make the manga recap usable: quality, completion, and speed

Date: 2026-09-26. Status: model-free mechanisms implemented 2026-09-27; crop/model rollout remains unvalidated.

Implementation update: deterministic placeholder/sentence handling, chapter
checkpoints, task metrics, bounded OOM fallback, reviewed panel corrections,
narration eligibility, revision-bound reviewed facts/events, description
fingerprints, past-only writer context, temporal validation, combined
input/output budgeting, sparse timing, recursive split recovery, and restricted
one-panel fallback are implemented. The clean-crop production rollout and its
dependent S1-S4 request redesign remain intentionally gated on the visual model
experiment required by this plan; no heavyweight model was run.

Audit update: 2026-09-26. The plan was checked again against the implementation for usefulness, dependencies, conflicting assumptions, and end-to-end escape paths. The deeper audit found four gaps in the earlier plan: page/panel selection was treated as trusted input, false body detections could still enter narration, every detected body was still assumed to need a Qwen appearance, and the writer received future-story context. Those gaps are now covered below. Low-value or speculative work was removed from the initial implementation. "Keep" below means the mechanism is justified, not that a model experiment has already succeeded.

## Goal and boundaries

Produce a faithful first usable recap, preserving panel coverage/order, stable IDs, descriptions, character records, actions, transformations, filtering metadata, and the three story outputs. Fix failures that prevent completion or corrupt important story facts. Ordinary wording, minor appearance detail, perfect brevity, and background-character completeness are not priorities.

This document records the agreed quality and performance fixes and the additional blockers found by inspecting the producer and consumer code. Only this document was changed for this task. No model, benchmark, or test was run. Any future heavyweight local model execution still requires explicit authorization under AGENTS.md. Proposed OCR/model alternatives are not installed or selected for execution by this plan.

There is a remedy for every identified blocker below, including a bounded review/correction route when automation cannot establish a fact. That is not a promise that Qwen3.5-2B can resolve every blocker automatically. Review-dependent remedies are explicitly identified; ignoring a critical fact is not counted as fixing it.

## Audit decisions: what is actually worth doing

| Proposal | Decision | Reason / condition |
| --- | --- | --- |
| Clean panel and character crops | Keep; test before full rollout | Directly removes observed panel-mapping and annotation confusion, but does not guarantee correct visual recognition |
| Page-role, panel-geometry, and reading-order gate | Keep; target only uncertain/critical pages | Crops cannot recover a missing/merged panel or repair wrong order; current heuristic order and model-only page exclusions are not proof |
| Panel-local actions with an actor-focused repair | Keep | Smaller set of actor choices; only repair ambiguous/failed records, not every action |
| Narration eligibility for detections | Keep | A clean crop can accurately describe a false body detection as an object; the writer must be able to exclude it without deleting provenance |
| Placeholder rejection and complete-sentence preservation | Keep | Addresses demonstrated data corruption/loss; preserve valid short facts and existing output coverage |
| Current classification/provenance checks | Keep; reuse existing preflight | Missing classification is a real blocker, but a new diagnostic framework is unnecessary |
| Reviewed critical corrections | Keep as an explicit fallback | Can resolve critical residual errors, but requires real review; it is not an automatic AI fix |
| Separate plot-text evidence | Keep the small reviewed-fact path; automate conditionally | Need accepted plot facts, not a full new OCR subsystem before the first usable result |
| Cross-page transformations | Keep reviewed explicit endpoints; defer automatic scanning | Covers critical transitions without adding another unvalidated pass to every page boundary |
| Output-aware writer budgets and chunk splitting | Keep together with sparse-segment handling | Concrete completion mechanism; splitting alone otherwise conflicts with average-word validation |
| Full evidence-ledger infrastructure | Simplify to records with source/status in existing exports and one review sidecar | Provenance is useful; a new database/workflow does not itself make prose true |
| Mandatory constrained decoding or short-ID rewrite in the writer | Defer unless the small writer trial shows format/ID failures | Additional grammar work can slow decoding and force bad responses; fix budgeting first |
| Separate character-retry scheduler | Remove as a separate project; fold into general batching | Same mechanism and benefit as S2/S3, not an additional speedup |
| Describe every detected body with Qwen | Remove | Many tiny/background/false detections are not needed by the story; retain their instance records but spend inference only on narration-relevant appearances |
| Image-preprocessing/vision-feature cache | Remove from initial plan | Preprocessing alone is likely small; feature reuse is unmeasured and model-specific, especially with mostly unique crops |
| Detailed GPU/kernel profiling and backend changes | Remove from initial plan | Record task wall time/tokens/retries first; investigate deeper only if runtime remains blocking |
| Visual-hold/audio/video changes | Defer | No need to expand downstream production for a text-generation fix; keep actual-word timing truthful |

Do not implement all conditional proposals before measuring the basic crop experiment. No rule-based validator can certify that a fluent visual claim is true, and fewer exported records must not be presented as improved accuracy.

## Evidence and corrections to the earlier assessment

- Latest analyzed export: `output/Maou-sama_descriptions.json`, written 2026-09-26 at 19:59. It includes 52 story pages and 159 panels; two pages were excluded as extras.
- Twelve panels contain exactly `Visible manga scene.` Another 30 start with generic wording; that wording alone does NOT prove failure.
- There are 215 exported character descriptions. Their presence or passing a text heuristic does NOT establish visual accuracy. The earlier approximate 48% usable-character figure was not a measured accuracy score and compared against all 319 manifest labels, including excluded pages.
- A keyword scan found much less dialogue/name contamination. Zero keyword matches is not proof of zero text leakage. Likewise, zero transformations does not establish good recall.
- Visual inspection confirmed wrong-panel content, invented objects, and missing prominent characters on known failure pages. Chapter 2 page 12 panel 3 is described as a sword scene despite showing a mechanical creature.
- Chapter 2 page 14 exports `, no label provided` for all nine characters and never retries them: the validator accepts these placeholders. Chapter 2 page 12 exports a crop retry describing a cyan box/red label.
- Chapter 2 page 24 panel 1's crop retry contains useful walking/pointing observations, but clipping and rejection turn the final result into a generic fallback.
- Accepted action proposals include appearance lists, not actions. Their model confidence of 100 is not proof of correctness.
- `output/Maou-sama/character_classification.json` is currently absent. The writer requires it before loading its model. This is a confirmed execution blocker, separate from quality.
- Writer execution on the new data has not been tested. Writer budgeting and validation concerns below are code-supported risks, not observed crashes in a completed story run.

## Blocker inventory and disposition

| ID | Blocking issue | Evidence/status | Remedy |
| --- | --- | --- | --- |
| B0 | Missing/currently incompatible downstream artifacts | Classification JSON missing; writer requires current instance mappings | Dependency preflight, current ReID export, precise invalidation |
| B1 | Wrong-panel descriptions and invented visual facts | Observed against source images | Clean per-panel extraction; targeted correction for residual critical errors |
| B2 | Unusable appearances and annotation descriptions | Observed in main and retry responses | Clean single-character extraction; explicit ambiguity/missing states |
| B3 | Wrong or meaningless actor-specific actions | Observed rejected mappings and accepted noun lists | Local-panel grounding; actor-focused clarification; semantic acceptance |
| B4 | Useful evidence lost; placeholders counted as success | Observed clipping/fallback behavior | Complete observations, bounded repair, honest per-field quality states |
| B5 | Bad evidence reused as stable character facts | Code copies earliest group description downstream | Validate before propagation; restrict identity-level reuse |
| B6 | Uncertain cross-page identity/name continuity | No verified end-to-end identity result; naming permits proposal-group aliases | Reviewed recurring-character anchors or neutral local references |
| B7 | Dialogue-dependent plot absent from visual evidence | Visual-only prompts/masking; no text-evidence channel | Separate OCR/reviewed story-text path with attribution and provenance |
| B8 | Critical transformations cannot be established across pages | Export/consumer restricted to within-page endpoints | Reviewed explicit endpoints first; optional adjacent-page proposals |
| B9 | Too little evidence to support required narration length | Writer enforces per-segment and average word limits | Scene-supported concise narration; explicit timing exceptions |
| B10 | Story generation truncates or repeatedly fails a large chunk | Input-only chunk sizing, heuristic output budget, stop after two failures | Output-aware budgeting, smaller fallback chunks, resumable assembly |
| B11 | Valid JSON can still narrate unsupported facts or early names | Validator checks IDs/names/basic length, not all claim support or temporal availability | Fact IDs, temporal checks, critical-claim review, conservative fallback text |
| B12 | A story page/panel is excluded, merged, missing, or read in the wrong order | Current page-role and geometry/order decisions can be automatic; clean crops inherit those mistakes | Conservative page-role policy, targeted geometry/order review, revision-bound corrections |

B6-B9 are conditional blockers: missing a background character's name or an optional event does not block a visual recap. They become blocking when a central action, identity change, or causal plot step cannot be narrated truthfully. Do not require perfect extraction of every decorative detail.

### B12: Validate the material that reaches description

- Treat page inclusion, panel geometry, and panel reading order as upstream evidence, not unquestionable truth. A clean crop cannot restore a panel that the detector omitted, split a merged panel, or undo an incorrect panel number.
- Do not let a model-only `extra`/`reference` answer silently remove a page from the final story. Automatic exclusion is allowed only for current deterministic layout metadata; otherwise retain the page as uncertain until reviewed. A reviewed role decision is bound to the source hash and layout revision.
- Add a small review/correction record for panel boxes and reading order on uncertain or plot-critical pages. Preserve the original detector output for audit. Applying a correction creates a new geometry/grounding revision and invalidates only affected descriptions, events, identity occurrences, and writer checkpoints.
- Check the current transitive row/column reading-order heuristic on representative normal, tall-spanning, inset, borderless, and merged layouts. Unit tests can prove deterministic behavior for fixtures, but the final plot-critical sequence still needs comparison to the drawings.
- The preflight must fail with an exact page/panel reason when a critical correction is unresolved. Noncritical uncertain geometry can remain included and be narrated neutrally; it must not be dropped merely to make coverage pass.

This gate is targeted rather than a mandatory re-run of all panel detection. Keep `test_panels.py` unchanged unless a confirmed detector/export defect requires a code fix; most corrections should be imported as revision-bound data.

## Core quality fixes retained from our discussion

### B1: Make clean panel crops the primary description input

- Build crops from original pixels and manifest boxes. Apply text masking in the correct coordinate space. Do not crop the already cyan/red-annotated preview for general descriptions.
- Clamp boxes to the source, translate text/character boxes into crop coordinates, and inspect representative masked inputs. Preserve art overlapping text; never erase a hand/face to satisfy an aggressive mask. Missing detail caused by masking must be repaired at input preparation, not by asking Qwen to guess it.
- Give each request one panel and one short task. Batch independent requests, not multiple panels in a single conversation. The program assigns panel IDs and order from request metadata.
- Replace the old whole-page description calls rather than running them in addition. Record total crop count and wall time: approximately three panels per page means more requests, so the old speed estimates do not transfer to this design.
- Preserve all visible beings, including creatures, without requiring a guessed species, gender, setting, motive, or object. Ask for visible facts only; an unknown setting is acceptable.
- Verify effective processor dimensions during the later experiment. The configured full-page cap is about 301,000 pixels, but actual resizing must be checked before claiming how much detail Qwen received.
- Keep page context for page classification and transformation review. Do not use surrounding panels to fill an unseen action into the current panel.
- For a splash/merged panel, cropping alone does not simplify it. Use overlapping detail views only when essential information is missing; merge observations under the existing panel ID and do not infer identity across views without evidence.
- Audit failed crop geometry against the original. Only correct panel/body geometry when it actually removes or misassigns critical content; this invalidates affected IDs/revisions and requires targeted downstream regeneration.

Why this is better than the old retry: the current retry still uses the full combined prompt and annotated image. This replacement changes both the task and the input. Expected improvement is high for panel mixing, but only moderate for hallucinations within a panel.

Failure exit: one alternative focused detail view, then an exact, source-linked correction for a critical unresolved panel. If many clean crops remain wrong, stop adding prompt rules; a stronger model or substantial review is required. Model suitability is an experiment, not an implementation guarantee.

### B2: Bind appearance requests to one character in code

- Use a clean, padded character crop from original art. Ask for one short appearance phrase or an explicit ambiguous/unavailable result. Do not ask it to locate `C17` or reproduce a dynamic dictionary of C keys.
- Associate the returned observation with the existing detection ID in code. Keep exported C labels and detection IDs for downstream compatibility; removing labels from appearance images does not remove identity fields.
- When overlapping people occupy the crop, use a wider panel view with one target marker, or mark it ambiguous. A body box is not proof that exactly one person is inside it.
- Do not infer brown/blonde hair from ambiguous monochrome shading. Prefer visible traits such as dark/light hair, horns, clothing, and silhouette.
- Reject annotations, missing-label messages, JSON fragments, and empty generic descriptions. Only strip stray punctuation when the remaining phrase is meaningful; punctuation repair cannot turn a box description into a person.
- Retain an instance and failure status when appearance is unknown. Every body label must still map to exactly one current instance even when it has no usable descriptor.
- Separate grounding from narration eligibility. Add states such as `eligible`, `uncertain`, and `rejected_non_character`, with source/review provenance. A red label, text box, crop artifact, or ambiguous multi-being proposal must not become a narrated character merely because it has a stable instance ID.
- Preserve every grounded label for coverage and audit, but pass only narration-eligible instances to character references, action endpoints, identity aliases, and writer prompts. A rejected occurrence stays addressable and can be restored by review; it is not deleted from classification.
- Do not request a Qwen appearance for every detection. Prioritize accepted/reviewed action endpoints, recurring or foreground proposal groups, and characters actually needed by the current scene. Tiny/background/low-quality instances keep explicit unavailable/unused status unless review promotes them. The lightweight visual matching stage remains independent of Qwen descriptions, so appearance work can be scheduled after likely recurring instances are known.

Expected improvement: strong for label/annotation failures; uncertain for tiny/occluded characters. Background omissions are nonblocking if they do not affect a narrated action. Recurring important characters have a reviewed-appearance fallback under B6.

### B3: Ground actions within the panel

- Replace the original full combined page request with an actions-only task; in the crop design, supply a single panel and only its local character labels.
- Draw fresh markers after masking so readable markers survive. Keep actor and target options restricted to that panel; retain exact detection/instance mappings.
- A proposed ambiguous actor gets a single focused panel request with that actor highlighted, preserving the surrounding interaction. Do not use a tight body crop to infer a target outside the crop.
- Trigger repairs for explicit uncertainty, failed local mappings, or inspected mistakes. A fluent wrong answer may produce no automatic flag; do not assume the repair trigger detects every factual error.
- Distinguish an actual action from an appearance list. `Standing` can be correct but low-value; `man in suit, woman in jacket` is not an action.
- Do not replace the current weak filters with a broad keyword blacklist that rejects valid phrases such as "the character with a hat raises a hand." Validate the action clause and its references, retain real short actions, and send unresolved critical attribution to the correction path.
- Do not assign an actor merely because only one character was detected: the detector may have missed someone. Do not loosen confidence thresholds just to raise action counts, and do not upgrade model confidence to certainty.
- Keep uncertain actor-specific claims out of named narration. Use supported neutral scene wording when attribution is not plot-critical. For a critical interaction, accept an explicit reviewed action with actor/target IDs and source panel evidence.
- Qwen's `confidence_percent == 100` is not an acceptance test. Until a held-out visual sample demonstrates adequate precision, model actions remain proposals. Only reviewed plot-critical actions, or actions admitted by a separately versioned and measured policy, enter `accepted_actions`. Any measured automatic policy must reject narration-ineligible endpoints and be easy to disable without changing the export shape.

Implementation must version the acceptance policy and update the resolver and writer together. Existing consumers accept only particular `accepted_automatic` policies; a reviewed action cannot be inserted as a fake automatic model result.

Expected improvement: moderate. Programmatic label validity does not prove visual attribution. A second model answer is not independent verification; the important test is against the panel.

### B4-B5: Preserve usable facts and stop invalid facts propagating

- Replace hard word slicing in `_finish_description` with complete-sentence handling. Allow sufficient schema/generation space to finish a short observation. Preserve a complete sentence preceding an unfinished tail; retry when no supported complete observation survives.
- Keep a bounded output ceiling. More space alone is not the fix, and adding a period to a clipped clause does not make it complete. Retained sentences must still pass contamination/placeholder checks; visually unsupported claims remain subject to the visual evaluation.
- Keep length a soft preference. Do not retry a correct 20-word description just because the prompt said 16 words.
- Use per-field states such as usable, ambiguous, repair_failed, and reviewed. Preserve failure reasons when a compatibility placeholder is exported. Structural completeness and factual readiness must be separate.
- Retain existing JSON sections/fields; add status/provenance in a versioned extension. Update all readers together so the writer does not silently accept old placeholders or reject legitimate partial-but-recoverable evidence.
- Validate appearances before `_apply_grounded_descriptions` and again at writer preparation. Do not choose the earliest group description blindly. Prefer a current reviewed/correctly sourced description; limit reuse to appropriate confirmed entity/form membership and story position.
- Preserve rejected raw responses for debugging. Successful repair should resolve the active error without erasing the audit history.
- Add a source-linked review sidecar for description/appearance/action corrections, keyed by hashes, panel/detection IDs, and revisions. Validate and import it into the same pipeline; never hand-edit generated output to hide failures.

These fixes have high confidence for the observed mechanical acceptance/data-loss bugs. They improve safety and recover some evidence; they are not a substitute for visual recognition.

## Remedies for the remaining story blockers

### B0: Generate and verify current inputs before loading the writer

1. Reuse `prepare_story`'s existing model-free preflight; add only missing provenance and per-field readiness checks plus specific repair instructions. Missing manifests/crops/classification already raise errors. Do not build a separate reporting framework just to reimplement those checks.
2. After corrected descriptions are available, regenerate the identity result from the current manifests and descriptions. Reuse valid encoder caches. Do not run Qwen from the ReID stage.
3. Preserve exactly one stable occurrence for each current grounded label, including uncertain/non-prototype observations. Unknown identity is not the same as missing instance.
4. Record the exact description/evidence fingerprint used by the identity export and compare it during writer preparation. Current geometry checks alone cannot detect classification derived from older descriptions with unchanged boxes.
5. Keep reviewed records but invalidate any whose evidence changed. Regenerate only affected artifacts. A description-only change does not require rerunning panel detection.

Missing classification is solved by the dependency workflow, not a new model prompt. No ReID or writer job was launched for this planning task.

### B6: Identity continuity without turning similarity into truth

- Start with the existing ReID proposals and review the few recurring characters/forms needed to narrate the core story. Seed exact instance/form assignments through the existing review system. Automatic propagation remains conditional on calibrated evidence; reviewing two seeds alone does not authorize all similar crops.
- Use neutral, local visual references for unresolved people. Two characters with similar descriptions must not silently become one entity. For essential continuity that neutral narration cannot express, review the specific unresolved appearances.
- Preserve the user's chosen fallback-name feature, but separate an editorial alias from a canonical manga name and from proof of identity. `assign_group_display_names` currently sets `display_name_allowed=True` for proposal-group members: the writer must not interpret that as confirmed cross-page identity.
- Keep alias information for galleries/metadata; use a recurring alias in factual narration only with adequate membership evidence. Until then, `display_name_allowed=True` from a proposal group is overridden at writer preparation and the local neutral reference is used. Do not infer a real name from appearance or OCR proximity. Preserve actual name reveal positions.
- Do not fix background identities or tune all clustering thresholds unless they block the current recap.

Practical fallback: exact reviewed assignments for critical appearances. This requires review effort, but provides a bounded route to a usable story without promising perfect automatic ReID.

### B7: Add a separate path for plot-bearing text

This is needed for a full plot recap when dialogue/signs/narration carry essential facts. It is optional for an explicitly visual-only recap. Do not restore uncontrolled OCR leakage into visual descriptions.

- For the first usable version, support reviewed text/story facts in the review sidecar. This is a workable fallback even before automatic OCR is reliable.
- Automate transcription only if the number of critical text corrections makes review impractical. Use existing text regions on the original unmasked page and a suitable local recognition model (a conventional PP-OCR recognition pipeline is a candidate, not a selected dependency). First test representative bubble/sign crops. Preserve raw text, region coordinates, page/panel IDs, language, reading order, extraction confidence, and review status. Retry only unreadable important regions; do not require every sound effect.
- Overlapping text boxes must be deduplicated without deleting distinct dialogue; ambiguous panel/bubble assignment stays unresolved. Detector regions may be incomplete: missing critical text gets a targeted region correction.
- Keep text detection/recognition confidence separate from truth. OCR confidence does not prove correct negation, speaker, intent, or that a character's statement describes a real event.
- Give text facts kinds: narration, dialogue, thought, sign, or sound effect. Dialogue supports an attributed statement, not automatic acceptance of its contents as reality. Do not infer an off-screen event solely because someone mentions it.
- Bind a speaker only with adequate source evidence or review; nearest character is not sufficient. Unknown speaker statements may be summarized without attribution only when that preserves meaning. Critical speaker/negation ambiguity goes to the review queue.
- Add accepted text evidence to the writer input and prompt, with explicit fact IDs. Extend `STORY_WRITER_SPEC.md` and relevant schemas during implementation. Existing visual fields remain visual.

Acceptance example: the ending's sale/sold-out notice can support the missed-purchase plot point through sourced text, while the visual description independently describes the sign and reactions.

### B8: Resolve critical transformations, including page boundaries

- Preserve the current within-page proposal pass. Zero events is not an error unless an important real transition is missed.
- The quickest reliable first-version remedy is a reviewed transformation record with source and target chapter/page/panel/detection/instance IDs, evidence, and acceptance status. Identity assignments alone do not create an accepted narrated event; add a validated event-review import path.
- Extend the event schema additively with explicit endpoints and both pages' hashes/revisions. Existing within-page records remain readable. Update response normalization, resolver, writer, event hashing, and tests together; do not put cross-page events into fields that assume one page.
- Current automatic acceptance requires adjacent panel numbers within a page and rejects same-panel endpoints. Do not remove that restriction globally. Route true same-panel, nonadjacent, and cross-page changes through reviewed endpoint evidence until an appropriate policy is tested.
- If automation is needed later, inspect the last story panel of one page with the first story panel of the next, and flagged transitions. Preserve page order. Candidate gating may miss transitions, so it cannot be the sole guarantee for critical changes.
- Keep summoning, possession, fusion, splitting, and a character changing form distinct. Only a justified identity bridge may merge forms; co-presence and resemblance are insufficient.

If the story can honestly describe separate appearances without asserting a transformation, no bridge is required. If the plot depends on their being the same person, an unresolved bridge is a real blocker requiring the reviewed record or better visual evidence.

### B9: Narration length must fit available facts

- Plan narration over a small scene using accepted evidence, while retaining one ordered primary-panel segment and citations for every panel. Do not restate the same event for each reaction shot.
- Keep approximately 25 words/10 seconds as a target and the existing output/timeline structure. Do not force filler to meet the current hard minimum/average when evidence is sparse.
- Permit documented shorter segments in sparse scenes and derive timing from actual words/audio. Keep the current timeline fields and target metadata; do not add a visual-hold/audio/video subsystem for this fix.
- This is a deliberate exception to the approximate ten-second target: update the spec and validator, expose the actual shorter estimated duration, and do not silently promise ten seconds. If fixed ten-second display is subsequently required, treat that as a separate timing decision.
- Use the same evidence-aware length policy for ordinary generation, split chunks, single-panel fallback, resume validation, and final assembly. The existing 8-word segment minimum and 16-word average would otherwise reject a truthful short fallback or make splitting counterproductive. Do not transfer an arbitrary chunk-average constraint to every split piece.
- A brief reaction panel with enough genuine visual evidence need not block the whole story. A panel with no usable evidence needs repair/review; transferring a neighboring panel's action into it is not a remedy.

### B10: Make generation failures recoverable

- Budget input AND expected output. `len(chunk) * 64 + 384` currently covers narration plus repeated long panel/instance IDs and JSON syntax with an unverified allowance. Tokenize representative serialized output structures and reserve narration, identifiers, syntax, and context headroom.
- Preserve the single-request path when the full input/output safely fits. Otherwise split on scene boundaries and record why. Extend the spec's current input-only condition rather than pretending the 48k input threshold guarantees completion.
- Initially keep the current output shape and ordinary generation, with corrected budgeting and bounded recovery. Do not add a writer grammar or short-ID rewrite by default. If the later trial shows recurrent JSON/ID failures, test one of those remedies on the failing cases before adoption; neither is a factual-quality fix.
- After one correction attempt, split a still-failing multi-panel chunk and process the pieces in order. Recalculate budgets for the correction prompt. Apply B9's compatible length rules and a finite retry allowance per split node; each split must reduce panel count. Preserve all raw attempts and checkpoint completed chunks exactly once.
- Retry recoverable memory failures only with smaller workloads; handle OOM consistently for main, focused, character, repair, and writer calls. Do not catch unrelated exceptions as memory problems.
- For a one-panel writer failure, produce concise narration from already accepted, complete facts using a restricted template, with exact references. If facts are insufficient, use a reviewed sentence through the same evidence checks. Never relax validation simply to publish a file.

This addresses completion failures without another huge repeated request. It cannot rescue incorrect source facts; B1-B8 must establish those first.

### B11: Prevent unsupported narration and future-name leakage

- Represent accepted facts as small records in the existing exports plus the review sidecar: source IDs/hashes, content, status, and extraction/review provenance. Reuse action/event IDs and stable panel/instance references. No separate evidence database or general-purpose workflow engine is required.
- Supply only usable or reviewed facts to narration and validate existing panel/instance/name references. For new text/reviewed-event inputs, retain their fact IDs as an additive extension. Avoid forcing the model to generate a large citation structure for every ordinary descriptive phrase; reference bookkeeping is not semantic verification.
- Enforce evidence availability at the primary panel's story position. Current validation can accept a supporting panel anywhere in the supplied chunk; a later named instance must not license that name earlier.
- Remove `complete_story_outline` built from future panel descriptions from ordinary generation. Replace it with an optional summary of already completed accepted facts. Filter the character guide for each request to current/prior eligible occurrences and names available at that story position. Future panel descriptions, identities, text facts, and name reveals are not prompt context at all, rather than merely being accompanied by an instruction not to use them.
- Do not rely on IDs to prove semantic entailment: a model can cite a real fact and still embellish it. For the first usable run, inspect plot-critical claims against the cited source and use a restricted sentence from accepted facts for unsupported segments. Broad narrative polishing is deferred.
- Keep ordinary valid visual prose usable. Review is targeted at central identities, actions, transformations, contradictory claims, and missing causal facts, not every adjective.
- Keep review lightweight: one validated JSON sidecar and a readable list linking the source panel/text region, current claim, failure reason, exact IDs and hashes. Reuse existing galleries where applicable; a new interactive review application is not required. Accept a corrected observation/statement/action or explicit unresolved decision. Changed sources invalidate prior approvals. Human corrections must also pass structural, temporal, and identity checks.
- Use the same accepted-policy representation in ReID and writer. Add support for reviewed records explicitly; do not label human decisions as `accepted_automatic`.

Important limit: schema checks, lexical filters, model confidence, and repeated Qwen agreement cannot guarantee factual narration. The combination of simpler extraction, traceable evidence, and targeted review provides a practical completion route. Fully unattended reliability remains unproven.

## Closure assessment after the deeper audit

The approach now has a completion route for every known blocker, but the route is not automatic in every case. “Covered” means either the automatic path passes a source-based gate or the exact critical fact is corrected/reviewed; it does not mean a generated answer is trusted because it is fluent.

| Area | Automatic-fix confidence | What closes the gap if automation fails |
| --- | --- | --- |
| Artifact/provenance compatibility, placeholders, clipping, invalid propagation | High | Deterministic preflight and targeted regeneration |
| Page inclusion, panel geometry, and reading order | Moderate | Hash/revision-bound role, box, and order correction |
| Panel descriptions | Moderate | One focused view, then source-linked correction for critical panels |
| Character detections and appearances | Moderate for good crops; low for tiny/false detections | Narration-eligibility decision and reviewed appearance only when needed |
| Actor/action attribution | Moderate at best until measured | Reviewed actor/target event or neutral narration |
| Recurring identity, aliases, and transformations | Low-to-moderate automatically | Exact reviewed instances/forms/endpoints |
| Dialogue-dependent plot facts | Low without a tested recognizer | Reviewed text/story fact with speaker/truth status |
| Writer completion, coverage, and resume | High after deterministic tests | Smaller chunks and restricted one-panel output from accepted facts |
| Semantic faithfulness of final narration | Not fully certifiable automatically | Source check of plot-critical claims and restricted corrected sentence |

Release condition: there must be no unresolved critical item in page order, panel content, identity, action, transformation, or plot text. An unresolved noncritical item may remain explicit and neutral. This is the part that prevents “all tests passed” from being mistaken for “the story is correct.”

## Speed fixes to retain, ranked and qualified

These are the retained low-risk ideas. The earlier 15-30% and 10-25% figures were unmeasured estimates for the old whole-page design; they are withdrawn as predictions for the crop redesign. Measure actual runtime before assigning percentages. S5 is removed from initial scope; IDs are retained for continuity with the earlier plan.

| Priority | Fix | Expected benefit | Constraint |
| --- | --- | --- | --- |
| S1 | Replace original combined generation with actions-only / focused local tasks | Removes output that is currently discarded; crop workload determines net saving | Keep description, appearance, and transformation functionality through their replacement tasks |
| S4 | Skip unnecessary appearance/action inference by explicit downstream need | Potentially substantial because there are hundreds of detections; exact saving is unmeasured | Never delete labels; always process reviewed/recurring/action-critical characters, and skip action work only where no narration-eligible endpoint exists or the scene is explicitly reviewed as not needing attribution |
| S2 | Batch independent requests using a per-request schema and group similar pixel/output workloads | Fewer underfilled batches; old 42-vs-27/28 page-batch calculation does not describe the new crop workload | Preserve batch size initially, correct padding/mapping, output order, and bounded OOM fallback; include character repairs in this scheduler |
| S3 | Batch panel repairs and generate only repaired fields | Most useful on retry-heavy pages | Preserve one meaningful repair and all quality checks; no full character/event generation for discarded repair fields |
| S6 | Checkpoint completed extraction units and reuse unchanged results | No substantial first-run gain; major interruption/rerun savings | Fingerprint source pixels, boxes, masks, task/prompt/schema/model settings and preserve raw provenance |

The description runner currently writes its combined JSON only after `describe_manga` returns, not after chapter 1. A future checkpoint sidecar should preserve completed work and assemble the same final export; do not recommend interrupting a current run assuming chapter output has already been saved.

Initially record end-to-end wall time, per-task elapsed time, tokens, batch sizes, and ALL retries. The current reported generation time omits character work and repairs, and generation-attempt counts omit focused passes. Detailed encoder/kernel/grammar profiling is unnecessary unless these measurements identify a remaining blocking slowdown. Timing fixes are instrumentation, not speedups.

Do NOT prioritize: another tokenizer-tree cache (already cached), enabling use_cache/inference mode/thinking-off (already set), tiny prompt wording edits, lower global resolution, truncating token limits, deleting validation, one giant all-purpose prompt, or speculative backend migrations. Kernel/compilation changes are optional profiling-led work after readiness; no speed guarantee.

## Implementation order and likely files

1. Freeze representative baseline examples, add the demonstrated placeholder/sentence-loss regressions, and identify the current artifact dependency failures. Audit page roles, geometry, and order for those cases first (B12); otherwise the crop experiment can test the wrong material. Reuse existing model-free preflight code.
2. Implement a small clean-crop panel/selected-character experiment with simple tasks (B1-B2). Include a false detection, overlapping characters, a tiny character, and ordinary controls. Before replacing all production passes, compare against the source images on the known failures, with explicit model-run authorization. If it does not materially improve factual accuracy, stop the crop rollout and use the correction/stronger-model decision path.
3. After that gate passes, integrate panel-local actions, replacements for redundant passes, selective narration eligibility, and S1-S4 batching. Add truthful field statuses and B4-B5 validation. Model actions remain proposals until the visual acceptance sample justifies an automatic policy. Main areas: `description/describer.py`, `images.py`, `schema.py`, `response.py`, `quality.py`.
4. Add atomic extraction checkpoints (S6) and the small source-linked correction sidecar. Preserve the returned/final combined export. Extend ReID and writer for the new status/reviewed-fact fields. Keep every grounded occurrence, but exclude narration-ineligible rows from aliases, event endpoints, and prompt characters.
5. Resolve the critical recurring identities, text facts, and transformation bridges (B6-B8). Use reviewed facts for the first run; add OCR or automatic boundary inspection only if the practical workload justifies it. Generate the missing current classification and validate exact label coverage/provenance (B0).
6. Implement output-aware writer budgets, finite smaller-chunk recovery, temporal/evidence checks, past-only context, and compatible sparse-segment rules together (B9-B11). Remove the future outline and filter the guide by story position. Keep ordinary writer decoding initially. Files: `story_writer/writer.py`, `story_entities/pipeline.py`, `naming.py`, and related specs/tests. Keep Qwen out of ReID.
7. Produce and inspect one short contiguous scene end to end, then the chapter and complete story. Measure useful fact coverage, factual errors, review effort, and runtime. The final release gate is the absence of unresolved plot-critical items, not merely successful JSON generation. Adopt conditional additions only when this exposes a real unresolved blocker.

Possible new files: `description/tasks.py`, `description/scheduler.py`, and one correction-schema/import module. Extract modules only where needed to keep implementation readable; a new evidence architecture is not a prerequisite. Keep `test_panels.py` untouched unless a confirmed geometry or provenance defect requires it.

## Validation and stop conditions

- Unit/contract tests: clean crop/mask coordinates, split/overlapping detections, local-to-external label maps, batch result isolation/order, placeholder rejection, sentence preservation, partial failures, OOM fallback, hashes, reviewed-evidence ingestion, and backward compatibility.
- Upstream-selection tests: deterministic/reviewed page roles, model-only exclusion held for review, missing/merged/corrected panel revisions, and manga order fixtures covering tall-spanning, stacked, inset, and borderless panels.
- Eligibility tests: a stable false/body label remains in classification but cannot enter the writer, receive an alias, or become an action endpoint; restoring it by a current reviewed decision works. Verify skipped optional appearance jobs remain explicit rather than looking complete.
- Regressions: chapter 1 pages 2, 5; chapter 2 pages 1, 7, 12, 14, 18, 22, 23, 24, plus ordinary pages. Assess against drawings, not only text heuristics. Include actual transform/no-transform examples; do not fabricate evidence just to get nonzero events.
- Writer tests: complete ordered coverage, exact current eligible instances, reviewed and measured automatic policies, rejection of self-reported-confidence-only events, no unsupported/early names, absence of future outline/guide entries from prompts, future-support rejection, output-aware splitting, two-failure recovery, one-panel fallback, resume without duplicate segments, sparse-scene timing, all three outputs.
- Recover materially more usable evidence on the existing failures. A smaller error count caused by deleting records or replacing them with generic placeholders is NOT improvement. A few rough sentences, incidental descriptor omissions, or short segments are acceptable.
- Confirm all plot-critical facts and identities used in the trial scene are sourced; do not label the full manga ready based on structural tests or a sample alone. Residual critical errors need exact corrections before final publication. Noncritical uncertainty may stay neutral.
- Track quality and actual total runtime together. The new crop workload may cost more than the previous estimate even with less duplication; do not claim a speedup until measured.
- If corrected crops still repeatedly invent major content, invoke the explicit correction route or reconsider model capability. No indefinite retry loop, weaker acceptance threshold, or untested promise of 8/10 quality.

## External references checked during planning

- [Qwen visual input processing](https://github.com/QwenLM/Qwen3-VL#image-resolution-for-performance-boost): pixel-budget resizing and per-input dimensions. Supports investigating effective crop detail, not a guarantee of manga accuracy.
- [Transformers generation configuration](https://huggingface.co/docs/transformers/main_classes/text_generation): generated-token limits are separate from input length; validate both when sizing writer jobs.
- [PaddleOCR pipeline documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html): text-region/recognition outputs and confidence filtering. Candidate transcription component only; manga reading order, speaker identity, and semantic truth require separate handling.
- [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer): independent per-sequence constraints for batched generation. Format validity does not establish semantic correctness.

## What remains intentionally unfixed

Minor wording, exact description word counts, perfect background-character naming, clustering-score tuning without a demonstrated critical error, sophisticated style polishing, TTS/video production, and speculative model/backend upgrades. All identified execution and critical evidence blockers have a remedy above; success of the automatic visual remedies still needs empirical verification, and the review fallbacks require actual reviewed evidence.
