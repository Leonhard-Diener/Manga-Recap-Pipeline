# Model output policy

## HIGH restored for five-page run (current)

User requested restoring HIGH thinking and processing the first five pages.
HIGH/includeThoughts is restored; unlimited transient retries remain. The
five-page run uses separate resumable outputs in `high_five_pages/` under the
existing comparison repository. Prior MEDIUM findings below are historical.
The run completed all first five pages with five HTTP200 responses and nonempty
exposed thought summaries for every page. Final answers/full API replies are
archived separately; production final outputs were not overwritten. No factual
quality improvement is inferred from successful completion.

## MEDIUM thinking and unlimited transient retries (historical trial)

Gemini used MEDIUM thinking during this trial at the user's request, still with
`includeThoughts: true` and separate response archives. Its cache fingerprint
changes with the thinking level, so HIGH results are not silently reused.
Both the runner and direct Google clients default to no transient-retry count
limit. A caller can still explicitly request finite retries. The isolated live
test driver's four-request cap and ten-minute child timeout were removed.
Quota/authentication errors still stop, and Ctrl+C can interrupt waiting.

MEDIUM may reduce token use/latency, but no evidence establishes less server
demand or fewer503s. The isolated `live_medium_resume_check/` run passed real
wait/hard-exit/resume checks: three real requests (200/503/200), plus the synthetic
503. Both successful MEDIUM responses returned no readable thought summaries
despite includeThoughts; opaque signatures and complete API replies were saved.
Page5 still described lightning. Retry/resume success is not factual accuracy
or a guarantee that MEDIUM exposes summaries. Historical reports/settings below
remain as records of those earlier runs, not claims about current defaults.

## Durable progress and transient-error waiting

`test_description_google.py` now waits and retries temporary HTTP408/500/502/503/504
and network/timeout failures instead of exhausting three retries and abandoning
the page. The same policy applies to initial model availability checks. Waits
start at 45 seconds and grow to five minutes; Retry-After seconds/HTTP-date values
can extend that wait (absurd values are capped at one day). Ctrl+C interrupts the
wait. `--max-retries N` retains bounded operation; `0` makes no retries. Direct
library clients also default to unlimited retries unless given an explicit cap.

Authentication, request, model-not-found and quota/access HTTP400/401/402/403/404/429
failures stop rather than indefinitely retrying or spending on later pages.
CLI interruptions and expected operational errors report how to resume without
a traceback. Disk errors and code defects are not hidden as transient API errors.

Completed pages already publish atomic chapter checkpoints before the combined
export. Shared atomic JSON writes now flush/fsync temporary files before rename,
so archives/checkpoints are synced before replacement. Rerun the same command
to reuse current completed pages and retry the unfinished page. Changed source,
geometry, prompt or inference fingerprints still intentionally invalidate stale
results. Raw JSON API replies remain independently archived before parsing.

This is crash-resumable saved progress, not a promise against hardware failure,
disk corruption, lost in-flight server replies or every possible software defect.
The unfinished page may need another API request after interruption. No API calls
or full extraction runs were launched to test retry behavior; regression tests
simulate repeated503s, timeouts, limits, quota stops, interrupted writes and resume.

## Gemini HIGH thinking and response archives

Gemini requests now use HIGH thinking and request exposed thought summaries via
`includeThoughts`. Previous LOW settings were held constant for comparative
experiments, not a demonstrated quality optimum. Gemma's existing MINIMAL
setting remains unchanged. HIGH does not establish more accurate extraction.

The Google runner saves each JSON API response under
`output/<manga>/google_responses/<image stem>_<unique id>.json`, with separate
`final_answer`, `thought_summary` and `raw_api_response` fields, plus the model,
image hash, prompt and generation settings. Thought-marked text is excluded from
the final answer/JSON parser. Empty summaries stay empty. Original malformed,
truncated, thought-only and JSON HTTP-error replies are archived before handling;
retries/format repairs get separate files. Network failures/non-JSON HTTP bodies
cannot supply a raw JSON API response. No credentials or image bytes are archived.

Programmatic Google clients return the summary, raw API response, thinking token
count and optional archive path. Set `response_dir` for on-disk archiving outside
the runner. Final public extraction fields are unchanged. A v2 Google cache key
includes thinking settings so LOW checkpoints are not silently reused.

Gemini gets 4096 additional tokens of shared output headroom for thinking; the
archive records the effective total cap. This is not a guarantee against
truncation. HIGH can increase latency and billed tokens/quota use. Exposed
summaries are not complete internal reasoning or causal proof of an error.
No model fallback or change in grounding rules is made.

The initial live page5 HIGH/summary probe returned HTTP503 high demand; its raw
error/settings were archived in `high_thinking_archive_check/`. A subsequent
isolated two-page operational-runner test in `live_resume_check/` succeeded:
both Gemini 3.5 Flash Lite HIGH-thinking responses supplied exposed summaries
(2180/3319 characters). The runner used its configured medium image resolution.
After an injected503 it really waited 45.13s; after a hard process exit it reused
the saved first page and requested only the unfinished one. This validates
archive/retry/resume behavior on those two pages, not factual story accuracy or
recovery from every possible failure. Existing production outputs were untouched.

## Rejected DeepSeek follow-up candidates (2026-11-13)

DeepSeek V4.1 Flash Free (low thinking) diagnosed the remaining force/effects
and warning-attribution failures from images, relevant prompt excerpts and the
failed replies. Four isolated candidates tested expanded lettering criteria,
tail-to-body criteria, effect-outline/illumination criteria and explicit
box-overlap non-evidence. Failed results were fed back after each round.
14 Gemini attempts produced 11 responses and three HTTP503s. None met the
predefined cross-case criteria: energy/lightning/explosion remained, or a warning
was still attributed to the girl. Useful links survived some cases but were not
reliable. DeepSeek recommended stopping both lines after repeated failures.

No candidate is adopted. Production remains v17_lettering_before_effects; no
schemas, source images, generated answers or final outputs are rewritten. These
failures do not establish correctness of existing production. Full evidence is
in `output/model_comparisons/Maou-sama_input_fix_20261112_214237_017992/deepseek_followup/`.
DeepSeek's final unverified speculation that page5 might contain a genuine tail
is not accepted as ground truth or used to change the evaluation labels.

## Lettering-first effect check (v17_lettering_before_effects)

The shared visual rules now first check whether apparent effects may be Japanese
or Chinese characters. Reading the glyphs is unnecessary; an ambiguous effect
is omitted. The exact combined prompt was tested twice on page 5 with otherwise
unchanged input/schema/settings. Neither response mentioned lightning or energy.
The first retained flight and unknown warning attribution; the repeat invented
an attack/blast cause and assigned that warning to C4. Retained as a narrow
lettering-versus-effect improvement only, not a fix for story-critical grounding.
Two responses do not establish general reliability. Other pages/local inference
were not live-tested. No output filtering, rewriting or schema change is added.
The version invalidates old checkpoints on future runs; existing outputs remain.

## Rejected v17 trial and assessment correction

The user-requested explicit Japanese/Chinese lettering and caption rules were
live-tested on pages 5/1 with the exact production prompt. Page 5 still added
lightning and falling; page 1 recovered captions but mistyped the one-shot notice
as narration and FLAWLESS as a character thought. Those additions were reverted
through Git, returning production to v16 at that point. Raw trial evidence is preserved in
v17_lettering_check/. No claim of a successful story-critical fix is made.

The user's correction of page-7 C7 as a winged person remains accepted. The
assistant's earlier "hand" identification and nonexistent-speaker allegation
are withdrawn, not reinstated by reverting the prompt. Existing outputs remain
untouched.

## Panel scope versus occurrence scope (v16)

The shared visual rules formerly said "Describe only what is inside each box"
without restricting this to character appearances. They now explicitly limit
only appearances to C boxes, identify green/purple panel annotations separately
from cyan/red occurrence annotations, and require full-region descriptions.
Combined and focused-text prompts read text outside C boxes as well. This is
a prompt-scope correction, not a geometry, parser or identity correction.

A two-request combined-prompt experiment on pages 1/4 restored page 1's upper
figure/skulls and omitted captions. It also included a border-crossing-bubble
instruction that failed to fix page 4's line placement; that trial was reverted
through Git, then only the useful scope clarification was retained. The exact
reduced prompt was not separately live-tested. Page 1 still acquired unsupported
fire/destruction and mistyped the one-shot notice as narration. These are remaining
errors, not repaired output. No automatic corrections or filters are added.
The version invalidates stale checkpoints on future runs. Existing evidence,
raw trial responses and source pixels remain unchanged.

## Outside-bubble text and conservative extra classification (v15)

Combined and focused-text prompts skip incidental text outside bubbles, retain
story-relevant captions/names/ranks/reveals as narration, scene-object lettering
as signs and publication material separately. Importance alone does not justify
classifying text as a character's thought; speech/thought requires a visible cue.
Narration, signs and publication retain null speakers. No new text kind or output
filter is added; raw wording remains unchanged.

Combined and standalone role prompts reserve extra for non-story material or
clearly unrelated standalone content. Title-like styling alone cannot exclude a
scene, and a new scene/unfamiliar character is not evidence of unrelatedness.
Mixed story pages stay story; unclear distinctions use uncertain rather than
extra. The standalone classifier sees one page, so it must not invent continuity
evidence from unseen neighbors. Existing output schemas and artifacts are intact;
the version invalidates older checkpoints on future runs.

User correction: flight on page 5 is valid. Earlier reports treating flight itself
as an error are superseded by this correction; unsupported energy effects and
crashing remain errors. This change adds text/role rules, not an anti-flight rule.
V15 has model-free regression coverage but has not been live-tested.

## Small tails on round bubbles (v14)

Both speaker prompts now explain that a lone point on an otherwise round bubble
may be a tail, even when very small. Shape alone is not proof of a speaker:
the extension must still clearly point to one supplied occurrence. With many
sharp corners, prefer unknown over guessing which point is a tail.

Six live Flash-Lite responses tested pages 2/4/5 twice. Both page-2 speaker links
returned correctly on the first run but were unknown on the repeat. Both warning
bubbles stayed unknown in both runs. The page-5 council line chose C2 rather than
the pointing C1 occurrence twice, and the page-4 close-up line was placed in the
wrong panel twice. The small-tail clarification is retained, but does not reliably
restore clear links or fix occurrence/panel grounding. Model-free tests also pass.
The version invalidates older checkpoints on future runs without altering existing
generated evidence.

## Decorative bubble borders versus speaker tails (v13)

Combined and focused-text prompts distinguish jagged edges, sharp corners and
repeated emphasis spikes from a distinct directional speaker tail. Without a
clear tail to one supplied occurrence in the same panel, retain dialogue with
`s:null`. No speaker postprocessing, schema changes or blanket-null rule is added.

Eight live Flash-Lite page responses tested the exact combined-prompt paragraph:
the first five pages, then repeats of pages 2, 4 and 5. Both warning bubbles stayed
unknown in both runs; valid links survived on pages 4/5. Both clear page-2 lines
also became unknown in both runs, and a name caption acquired a speaker on the
repeat. This is a targeted warning-case improvement, not a general grounding fix;
invented actions and coverage errors remain. Focused-text wording is synchronized
but was not separately live-tested. The version invalidates old checkpoints for
future runs; existing generated evidence is unchanged.

## Simplified requests without sound effects (v12)

The combined prompt puts uncertainty, local-label grounding and the bubble-tail
rule before field instructions. It replaces repeated checklists with short field
definitions. New text requests exclude `sound_effect` and skip decorative
lettering. Dialogue, narration, thoughts, signs and publication remain distinct;
existing parsers/ReID/writer still understand historical sound-effect records.
There is no dedicated visual-effects analysis requirement. Ordinary descriptions
can still mention a clearly visible flash or lightning without inferring cause.
The JSON field layout remains compatible; required panel and label coverage is
unchanged. Existing Google panel slots and panel-local label enums already enforce
coverage. Changing the public layout would add migration work without proving
better perception. The new description version invalidates older checkpoints
for future runs, without modifying existing generated artifacts.

## Uncertainty-first prompts (v11)

Inspect posture, held objects, then supported actions; omit uncertain claims
and allow empty action lists. Speech attribution requires a clear bubble tail
pointing to a supplied character label; otherwise use null. Appearance phrases
have no minimum length: an explicit empty string is retained with its detection
mapping and `appearance_status: "unknown"`, not rejected or replaced. Missing or
malformed descriptors still fail validation. ReID does not use an empty phrase
as appearance evidence. Page-role rules and model selection are unchanged.
The description version invalidates old checkpoints for future runs; existing
generated outputs are not modified by this change.

Descriptions, character appearances, action text, transformation evidence, and
transcriptions retain the model's exact strings. Short sentences, repeated
observations, quotes, annotation mentions, and fragments are not rewritten,
keyword-filtered, replaced with placeholders, or sent for content repair.

Whole-page images, the combined API request, and normal token budgets are
unchanged. The local text task sees the existing grounded
preview so its speaker-label options are visible. API page roles are used at
every page position; local interior pages still have no classification request.
An uncertain role is retained and included, not rewritten as `story`.

JSON/schema failures (including truncation, missing fields, wrong panel coverage,
and incompatible endpoint mappings) permit one additional generation of that
task with the same token budget. No content-based inference is added. Local
format failures still stop after that repair; unresolved remote pages are reported
without stopping later pages, never replaced by invented observations.
Successful repairs retain the failed response in `format_repairs`; failed
repairs are recorded beside the chapter checkpoint in a `.failure.json` file.

The description pipeline version invalidates older rewritten checkpoints without
deleting them. Rerun descriptions and then ReID when adopting the policy; do not
manually change generated artifacts. No model run is part of this code change.

ReID exports versioned format-only policies:
`grounded_visible_action_format_v3`, `grounded_text_transcription_format_v2`,
and `body_observation_format_v2`. The writer supports the new policies and
older v1 records, preserves input wording and generated segment strings, and
does not remove repeated cross-chunk sentences. Existing stable-ID, provenance,
identity, review, and temporal checks remain in place.

Format validity does not establish visual or transcription accuracy. Model
transcriptions remain statements, not proof that the stated events happened.
Normal request counts are unchanged. Format repairs and bounded HTTP retries
can incur additional calls; no paid fallback or model switching is performed.

## Google panel slots and bounded failure handling

The Google client sends required `panel_N` object keys instead of a panel array,
including for transcriptions. Each slot permits only that panel's actor/target/
speaker labels or null. Required slots prevent missing/repeated panel numbers;
small string enums are retained in the Google schema to constrain labels.
The client validates the wire response, then restores the public arrays and
numbers without changing observation strings. Original wire responses are kept
for audit. Full public schema and grounding validation still runs afterward.

If a remote page exhausts its one immediate format repair, or its client exhausts
bounded HTTP retries, save an attempt diagnostic under the chapter's
`description_attempts/` and process the remaining pages. There is no deferred retry
loop or automatic budget escalation. The export lists unresolved pages under
`pending_pages`, and runners print an incomplete-run summary rather than success.
Successful pages are not regenerated. No failed-page prose or page-filter
exclusion is invented. The chapter checkpoint remains incomplete; rerunning with
unchanged settings processes only unfinished pages. The writer's existing
required-page checks reject an incomplete export.

Source/manifest validation, invalid configuration and disk failures are not
silenced. Existing compatible checkpoints remain reusable: the transport changes
representation only, not the public evidence contract or acceptance policy.

## Visible-evidence completeness (v9)

The combined Google/OpenRouter prompt requests concise complete accounts rather
than a mandatory 20-word sentence. Its schema allows up to 600 characters per
panel description and 240 per action, without changing output-token budgets,
adding normal inference passes, or rewriting returned strings. Local decoding
limits remain unchanged. Both paths use visible-evidence instructions covering
posture, motion, effects, partial views, repeated depictions, and unknown actors.
Missing labels do not justify inventing an actor or omitting visible effects.

Mixed title/story pages should be classified as story when they contain scene
content, dialogue, narrative captions, or reveals. A cover/title layout hint alone
no longer forces exclusion. Explicit reviewed roles still override models;
unreviewed title/cover hints defer to the existing role-classification path (this
can require a local edge-page role request previously skipped by that hint).

Text requests distinguish narrator captions from character thoughts and assign
speakers only through a clear visual link, not proximity or a named addressee.
Remote combined responses now undergo the same panel-local endpoint validation
as local results before being cached; incompatible mappings get one format
repair, then fail closed with both attempts retained.

ReID no longer requires an action's leading verb to recur in the short summary:
two fields from one model are not independent visual proof, and lexical overlap
discarded separately described actions. Current hashes/revisions, exact eligible
actor/target occurrences, and nonempty action text are still required. A proposed
interaction between distinct occurrences already assigned the same entity stays
pending review rather than treating repeated drawings as separate people. The
resolver revision is `resolver_v10_grounded_actions`; the writer recognizes the
new action policy alongside v1/v2.

These changes require rerunning descriptions and then ReID. Original detections,
source pixels, saved outputs, and review history are not edited by the code change.
Prompt instructions and valid mappings are not independent verification of visual
truth: ambiguous interpretation may still require human review.

## Story text scope (v10)

Text evidence adds the kind `publication` for series/chapter titles, author
credits and promotional copy. Its exact wording remains in description/ReID
exports; no keyword matching or rewriting assigns/removes text. Story-relevant
editorial captions and explanations of in-story character titles remain narration.
Normal speech bubbles remain dialogue even when the speaker is unboxed or unknown.
Local and combined remote text instructions share this distinction.

Writer prompt records exclude only text explicitly typed as publication, put
unknown-speaker dialogue/thought in `unattributed_speech`, and put sound effects
in `sound_effects_context_only`. All accepted text remains in the prepared evidence.
The prompt forbids guessing speakers from proximity/adjacent bubbles, moving text
to a neighboring panel, or turning narrator captions into a character's thoughts.
Appearance aliases are hints rather than mandatory prose. These are evidence
scopes and generation instructions, not a word blacklist or semantic verifier.

Visual instructions distinguish each subject's pose from impact/motion effects,
avoid inventing body poses from comic reaction heads, and request visible actions
for unboxed subjects with null endpoints. This does not create missing detector
boxes or confirm identities. Source-bound panel/appearance reviews remain the
appropriate route for corrections the models cannot reliably make.

Description v10, resolver v11 and writer v10 fingerprints invalidate affected
downstream caches/progress. Regenerate descriptions and ReID before writing;
panel detection need not be rerun unless correcting its boxes/order. No generated
artifacts are hand-edited by this change, and older misclassified text is not
silently reinterpreted as publication text.

Cloud model switching is configured through `OPENROUTER_MODEL` in `.env` or a
one-run `--model` override. Both accept a model ID or OpenRouter model-page link.
The default is `google/gemma-4-26b-a4b-it:free`.

Once per run, the client checks endpoint metadata (not model inference). It
chooses strict JSON Schema, JSON-object mode, or prompted JSON text according to
an actual endpoint's capabilities, and omits unsupported optional parameters.
For modes without strict schema enforcement the schema is included in the
prompt. All modes use the same local validation and bounded format repair.
Models without image input or available endpoints fail before inference.
No model-specific code edit, paid fallback, or extra routine generation is needed.
Custom compatible APIs without metadata discovery can use `--response-format
json_schema`, `json_object`, or `text` explicitly.
