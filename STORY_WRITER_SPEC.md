# Story Writer Specification

## Goal

Create `test_write_storry.py`. It uses the information produced by the existing manga-analysis scripts to write an English YouTube recap.

The narration must:

- use third person;
- use present tense;
- sound like one uninterrupted story;
- stay faithful to the supplied information;
- narrate the story directly, without mentioning captions, panels, text extraction,
  or attribution status; unknown speakers remain anonymous ("Someone warns ..."),
  while narrator-caption content is told directly rather than announced as a caption;
- give each displayed panel about 10 seconds of spoken narration when read by text-to-speech at normal speed.

At a default narration speed of 150 words per minute, 10 seconds is about 25 spoken words per panel. This word count is only an estimate used before audio exists. The script may adjust sentence boundaries slightly so the narration sounds natural, while keeping the average close to 10 seconds of speech per panel.

## Inputs

### Display layout from saved narration

The panel timeline retains one timing entry per narration segment and adds
`display_policy` and `display_groups` for a 16:9 renderer. Adjacent panels from
the same source page can share a frame if side-by-side (right-to-left) or stacked
placement meaningfully improves artwork area. Artwork is fitted without cropping
or stretching; well-fitting panels stay single. Groups never change story text,
evidence links, panel coverage, or individual timing entries.

Each display group contains ordered `panel_ids`, normalized `placements`, and
the combined word/time span of its panels. Both panels remain visible throughout
their combined narration. Estimated seconds are not actual TTS timing: a future
renderer must align group boundaries to the corresponding audio segments.

Refresh an existing story without model inference:
`python test_write_storry.py "Manga title" --refresh-timeline`.
This validates matching story/timeline provenance and archives the old timeline
before updating it. It does not create a video or audio.

Use the available outputs from the earlier scripts:

- panel descriptions and their panel IDs;
- panel order and source image paths;
- accepted character identities and names;
- safe visual descriptions for unnamed characters;
- accepted character transformations or form changes;
- accepted story events and other confirmed information.

Only accepted information may be stated as fact. Pending or uncertain identity matches, names, events, and transformations must not be presented as confirmed. Every story section must keep the panel IDs that support it.

## Multiple Characters and Action Grounding

Before story generation, every detected character in a panel must have a stable instance ID. The description stage should receive a copy of the panel with visible markers such as `C1`, `C2`, and `C3` placed beside the corresponding detection boxes. Qwen must return actions and events using those labels instead of guessing character names.

Each grounded action should contain:

- chapter, page, and panel ID;
- the acting character's instance ID;
- a short description of the supported action;
- a versioned automatic or reviewed acceptance policy. Model-generated numeric
  confidence is not evidence.

A transformation should contain both the source instance ID and target instance ID, together with their panel IDs. The two appearances remain separate visual groups. Only an accepted transformation decision may connect them as different forms of the same underlying entity.

ReID later maps each instance ID to a visual group, accepted entity, and generated
alias. Retained ReID groups receive aliases without canonical-name extraction or
manual confirmation. Alias use does not confirm identity or event attribution.
The writer's reference policy may withhold a permitted alias temporarily to avoid
repetition; then it uses clear pronouns or a visual reference such as `the horned girl`.

This creates the required evidence chain:

```text
panel -> detected instance -> grounded action or event -> ReID entity -> permitted name -> narration
```

If a panel contains several characters and the acting instance is uncertain, the writer must use neutral wording or omit the uncertain attribution. It must never select a name based only on which known characters occur elsewhere on the page.

Human review is optional for publication. Revision-bound text transcriptions may
be accepted automatically, but dialogue/thought claims remain statements rather
than objective truth. Recurrent identity cores may be accepted automatically only
after representation separation, multi-panel recurrence, compatible grounded
traits, and multi-encoder visual support. Pending actions, text, identities, and
other unresolved proposals are retained for audit but ignored. If every clean
description retry for a panel fails, the panel remains in the timeline with an
explicit uncertainty record rather than being dropped or blocking the run.

## Qwen / llama.cpp Usage

Load the installed text model once when `test_write_storry.py` starts. Pass text and structured JSON to it, without images. This must remain separate from `test_character_reid.py`; the character ReID script must never launch Qwen.

The writer uses `models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf` (Unsloth's
Qwen3.5-4B Q8_0, revision `e87f176479d0855a907a41277aca2f8ee7a09523`).
Its runtime is `models/llama.cpp-b12292/llama-server.exe`, from the llama.cpp
`b12292` Windows CUDA 12.4 release, with that release's matching CUDA runtime
DLLs extracted alongside it. CUDA 12 is intentional: CUDA 13 drops Pascal
support, including the project's GTX 1080. These are ignored local artifacts.
Paths are controlled by `STORY_MODEL_PATH` and `STORY_LLAMA_SERVER` in `config.py`.

Each writer invocation starts its own localhost-only server on a free port,
with an 8192-token context, one slot, GPU offload, and thinking disabled. It stops
only its own server in `finally`, including when validation fails. Server logs
are retained under `output/llama_logs/`. There is no external API, automatic
paid fallback, PyTorch model load, or Python per-token grammar callback.
The description model and its loading behavior remain unchanged.

The local writer uses at most `STORY_CHUNK_MAX_PANELS` panels per request
(default four), even when more panels fit the context window. A context window
does not establish a small model's ability to finish a long structured response.
Requests also obey the shared input/output token budget. Output estimates reserve
45 narration words and every current character reference per panel.

llama.cpp's native schema grammar constrains JSON keys, segment count, current primary
panel IDs, and permitted reference/name values during decoding. It does not
rewrite narration or prove factual correctness. Post-generation validation still
checks order, attribution, and temporal/name permissions.

Each generated segment cites its primary panel and uses only that panel's
instances/names. Previous narration provides continuity, not additional evidence
for new claims. Empty name/instance lists remain empty rather than placeholder
objects; post-generation validation is still required.

Each later request receives:

- the count of panels already covered, not an ever-growing list of old IDs;
- the last 250 to 400 words of narration for style and continuity;
- the next ordered group of panels to write.

Previous narration is context only and must not be repeated. Published model
sentences are preserved; no final boundary-editing model pass or sentence removal
is performed.

## Writing Rules

- Follow the original panel and chapter order.
- Write in natural English, third person, and present tense.
- Use an energetic but clear YouTube manga-recap style.
- Make consecutive sections flow together without announcing chunks or prompts.
- Do not invent dialogue, actions, motives, relationships, names, or transformations.
- Use the supplied generated gender-neutral alias, not canonical names from dialogue or reviews.
- Aliases are automatic for retained ReID groups only. Fully assigned groups of the same accepted entity share an alias; one accepted member cannot rename unresolved group neighbours. Unmatched/filtered-out detections stay unnamed; naming adds no separate recurrence cutoff. Aliases do not prove identity or permit new events.
- Establish or disambiguate a subject with its alias, then prefer clear pronouns. Do not repeatedly introduce characters with "a boy named", narrate multiple views as "another [alias]", or describe distant people as small/tiny characters.
- Omit drawing/impact/lightning effects as presentation, without inventing a physical cause or removing accepted events. Gender-neutral aliases do not establish he/she; use clear current evidence or neutral wording.
- Use a clear visual description only when no permitted alias is available.
- Avoid unclear pronouns when several characters are present.
- Do not describe the same event twice merely because it appears in nearby panels.

## Panel Timing

The narration is divided into ordered segments. Each segment has one primary
panel and targets about 10 seconds of speech, normally around 25 words at 150
words per minute. Sparse panels may use shorter, explicitly truthful segments;
their estimated duration comes from their actual word count rather than filler.

The displayed panel changes when the next segment's spoken audio begins. The story should continue naturally across these switches. The timeline stores word positions and estimated start and end times so a later video script knows when to change panels.

Estimated timing is sufficient before voice generation. When real text-to-speech timestamps become available, the video stage replaces the estimates with measured audio timestamps. Each panel then remains visible for the actual duration of its assigned narration segment, with a target of about 10 seconds.

## Outputs

Write these files:

- `<manga>_story.txt`: the uninterrupted narration;
- `<manga>_story.json`: narration segments with text, chapter, primary panel ID, and supporting panel IDs;
- `<manga>_panel_timeline.json`: ordered panel switches with word ranges and estimated start and end times.

The JSON output must also record the input file hashes, identity revision, prompt version, model version, narration speed, and generation settings. This makes runs reproducible and prevents resuming from stale data.

## Reliability

Save progress after every completed chunk so generation can resume after an
interruption. Validate that all references are current and are not later than
the primary panel. Each chunk gets at most one correction, including the failed
response and its errors so the model can repair the actual mistake. A truncated
response gets a larger output budget for that correction if context/output caps
permit. After two unsuccessful attempts, save diagnostics and completed progress,
then split a multi-panel chunk into smaller ordered chunks. A failed single panel
waits five seconds and retries, carrying its latest repair feedback when it fits.
After `config.STORY_MAX_PANEL_FAILURES` failed single-panel cycles per invocation
(default four, at most eight generation requests), stop with a diagnostic error;
keep completed checkpoints and raw failures. This operational limit does not
change the prompt fingerprint or invalidate already accepted segments.
Never replace failures with automatic fallback text or
publish an incomplete story. Ctrl+C can stop the run; saved completed panels resume.
Transient local-server HTTP/network errors wait ten seconds and retry, up to three
HTTP attempts per request; an exited owned server fails immediately. Invalid
configuration, permanent HTTP errors, and disk failures still require intervention.
Input-sized chunk planning still happens before generation.
Standalone progress bars show completed story panels. `main.py` also displays
stage bars and retry messages from unbuffered logs, with a 30-second heartbeat
and a configurable no-progress watchdog (`--stall-timeout`, default 1800 seconds).
Repeated retry output cannot keep the story/description watchdog alive. It stops
only the owned stage process tree, retains saved checkpoints, and exits 124.
This cannot certify freedom from OS/hardware hangs; very slow valid work may
require a longer timeout. The interactive MangaDex download remains outside that
watchdog; its network/rate-limit retries are independently bounded.
Short observations of three or more words are valid regardless of
the density label; do not demand invented padding to satisfy a timing target.
Third-person narration and concise indirect speech are prompt-level style guidance,
not a hard pronoun filter. Brief attributed quotations of supplied dialogue may
contain "I" or "you" without changing the narrator's perspective. The validator
does not reject or rewrite such strings. Grounding, name permissions, temporal
references, coverage, nonempty text, and a relaxed 150-word acceptance ceiling remain
validated. The prompt still asks for under 50 words; accepted text is not truncated.
No special perspective-correction path is used. This acceptance relaxation retains
compatible existing progress rather than regenerating already accepted panels.
Existing source-linked fallback records remain validated and resumable.

The v36 writer prompt uses `story_context` for accepted narrator content and
`anonymous_dialogue` for speech with no known speaker; unknown-speaker thoughts
and context-only sound-effect transcriptions are omitted from the model prompt,
not from the saved evidence. Signs and other accepted text kinds remain available.
The public input
and story JSON contracts are unchanged. Validation rejects source commentary
such as caption announcements, unattributed-speech labels, camera-shot reports,
and explicit sound-effect inventories, feeding failures back for repair. Long
paragraphs copied verbatim from completed chunks are also rejected, not silently
deleted, unless the matching earlier panel has the same nonempty source
description (case/whitespace normalized). Repeated source observations may
legitimately produce repeated narration; different or missing descriptions do
not receive this exemption. This acceptance relaxation preserves existing
compatible checkpoints and does not change the prompt fingerprint.
Previous-chunk prose is not supplied to the model because live probes
copied it into unrelated panels. A compact `reference_context` carries previously
used aliases and the last unambiguous named subject from committed instance IDs,
scoped to aliases permitted in the current request, without carrying earlier
actions or prose. Unnamed panels receive no earlier aliases. Current evidence remains authoritative.
Validation also rejects clone-like "another [alias]", observed drawing-scale/effects wording, and repeated
"boy/girl named" introductions; rejected responses are retried, never text-stripped.
Alias frequency is now enforced: each alias may appear once per segment and must
then be absent for two segments. Recently used aliases are withheld from temporary
prompt facts, action references, reference context, and the output schema, replaced
with evidence-supported pronouns or visual references. This preserves IDs and saved
upstream evidence. A name-frequency rejection, or a declared name absent from the
narration, rebuilds the retry without any current
aliases or the named failed response; it does not deterministically replay the same
name-heavy input. The changed prompt policy invalidates older story checkpoints.
Once name suppression is activated for a chunk, it stays active through its
remaining retry cycles. Later validation feedback is retained with proper names
masked, without replaying the named failed response; fixing name metadata must
not erase corrections for missing dialogue. This retry-only repair preserves
already accepted compatible checkpoints and does not alter saved narration.
If a retry repeats optional sound-effect commentary, a terminal `and/with [a]
sound effect(s)` clause is removed only from the temporary retry observation.
The remaining observed subject and panel stay covered; source descriptions, raw
failed responses, accepted prose, and the ban on effect commentary are unchanged.
Readable entity/form IDs are replaced with stable opaque reference IDs only in
the prompt, preserving equality without leaking canonical names embedded in IDs.
Existing canonical names are reserved for validation, not supplied as narrator
aliases; their ungrounded appearance in output is rejected. Public stable IDs and
all upstream artifacts remain unchanged.
Explicit `I am [canonical name]` dialogue is adapted only in temporary writer
input: use that exact speaker's permitted alias, or retain the introduction's
title without its canonical name when the alias is withheld. Name-only statements
become an introduction statement. Saved transcriptions remain verbatim.
Known dialogue/thought endpoints receive a prompt-only `speaker_reference` from
their exact current occurrence, with the same drawing-scale cleanup as other
writer references. The writer must use that speaker with clear
he/she or neutral references, not anonymize them. When all dialogue endpoints in
a panel are known, anonymous speech-attribution sentence openings are rejected.
Mixed/unknown-speaker panels retain anonymous speech without guessed attribution;
an unknown endpoint does not establish that the speaker is someone *else*.
Meaningful unknown dialogue receives `speaker_reference: null` in temporary writer
input. Prefer a standalone exact double-quoted line without a repetitive
"someone says/asks" tag. Its normalized words must match the supplied unknown
dialogue, and it must stand alone rather than follow a known speaker's attribution.
A source-matching quote may be shorter than three words (e.g. `"Pardon?"`).
Explicit anonymous attribution remains an accepted fallback. This does not permit
guessing a speaker and is a targeted check, not full semantic proof.
Anonymous-attribution repair does not replay the failed guessed speaker. When a
single panel has only unknown dialogue and no accepted actions, transformations,
narration, or other meaningful text facts, a retry may constrain its output text
to validated standalone source quotations, with empty instance/name-use lists.
This is a decoding constraint (`source_quotation_only_v1` in attempt diagnostics),
not post-generation rewriting or a bypass of response validation. Panels with
accepted actions or known dialogue retain free narration after a required
standalone-quotation prefix (`source_quotation_prefix_v1`), so the speech repair
does not remove their other facts. The candidate quotation is checked by the
normal validator before adding either decoding constraint.
The prose target is one or two complete connected sentences for the main story
beat, not a list of visible figures and objects. Roughly 95% he/she/permitted-name
references is a prompt-level goal when grounded, not a numerical acceptance gate.
Temporary character hints are shortened to a role and at most one visual trait;
known-speaker/action references prefer evidence-supported pronouns. Long outfits,
hair descriptions and camera poses remain in source evidence, not prompt references.
Groups remain plural/neutral; aliases and job titles do not establish gender.
The v34 prompt consolidates repeated style/evidence instructions; its prose goal
does not establish that every sparse observation can become a meaningful event.
This v36 input/prompt policy changes the fingerprint. Trials use isolated output;
older production narration is archived only on an explicit production rerun.

### Story beats and visual holds (v35/v36)

The prose target is continuous spoken storytelling: establish supplied situation
and stakes, follow meaningful action/reaction, retain dialogue/reveals, and use
transitions only when the evidence supports the relationship. Reference transcripts
guide these general style characteristics; their characters, facts and scenes are
not supplied as evidence or copied into generated stories. Earlier long-prose
context is still withheld because it caused scene copying.

Every panel still has a segment and timeline row, but a panel without accepted
actions, transformations, meaningful text, reviewed narration, or uncertain evidence
may optionally use `text: ""`, `narration_policy: "visual_hold_v1"`, and empty
instance/name lists. This is an explicit silent visual hold, not a missing panel
or an automatic fallback for failed generation. The model decides whether the
visual-only panel adds a meaningful beat; this remains subject to prose review.
Blank text without the policy is invalid. Holds never excuse omitting accepted
speech, context, actions or transformations. The upstream evidence is unchanged.

The renderer retains these panels for 0.75 seconds, writing silence in the actual
speech WAV's audio format rather than synthesizing an empty sentence. Their word
span is empty; their visual/audio duration remains positive. All-silent stories
cannot be rendered. Narrated segments continue to use measured TTS durations.
Multi-panel chunks are bounded to one page; experiments can test up to four
panels together without changing panel ordering or identity/speaker permissions.
This is not whole-scene free rewriting or permission to invent causal links.
The initial v35 ten-page trial completed but still copied first-person narration
and camera commentary. V36 adds targeted rejection of those observed failures and
requires holds for three explicitly static description forms (legs standing,
standing in a doorway, and a hair close-up), only when the normal no-event/text
eligibility gate passes. Other visual-only panels retain an optional hold; action
descriptions with hair/body words are not automatically silenced. This deliberately
narrow rule is not a general semantic detector of incidental panels.

### Page-level scene prose

The operational story CLI now defaults to `--style scene`; `--style panel` keeps
the legacy per-panel mode. Direct library callers retain the panel-mode default
unless they pass `narration_style="scene"`. `story_writer/scenes.py` generates one
paragraph for each source page, not one sentence for every panel. Its separate
`scene_prompt_version` participates in input fingerprints and generation settings.
Cross-page scenes are not inferred. Over-budget pages stop rather than silently
dropping evidence. Existing checkpoints/finals with incompatible fingerprints
are archived; failed scenes preserve a completed prefix and raw attempts.

The prompt prioritizes accepted narration, speech, and meaningful actions over
appearance observations. Routine poses, gaze and grooming are omitted from
temporary prose facts; targeted interactions and standing guard are retained.
This is a narrow syntactic selection rule, not a semantic action-importance model.
The original proposals and manifests remain unchanged. Only actual speech or
retained event endpoints receive character reference hints, so merely being
visible does not become an instruction to narrate that person. Reviewed narration from
every panel remains in scene input. Pronoun hints use the original occurrence's
gender evidence, not an alias or a guessed speaker. Unknown dialogue remains
unassigned, and actual character claims are not promoted into story facts.

First-person repair may constrain output to source quotations. Known thoughts
retain their grounded thinker and must declare that occurrence ID. First-person
narration with no known identity can remain a quotation rather than being assigned
to a nearby character. This is a safe fallback, not a claim that the model has
learned to retell every passage fluently. Repairs are bounded to four attempts per
page; raw prompts, schemas, responses and metrics are kept for audit.

The scene's final panel is its narration anchor, with `scene_panel_ids` and all
current scene panels in `supporting_panel_ids`. Earlier panels have empty text,
`narration_policy: "scene_visual_v1"`, and `narrated_by_panel_id` linking that
anchor. These are coverage rows, not discarded panels or independent silence gaps.
The renderer validates contiguous same-page coverage, synthesizes the paragraph
once, and divides its measured duration evenly across those panels. It does not
claim word-level/event-exact alignment within the scene. The existing aspect-aware
grouping, borders and backgrounds still apply. All three story outputs remain.

The first two per-panel style trials still failed prose review despite passing
structural tests. A page-level trial improved some passages but also produced
"the character stands while he stands"; this motivated filtering routine poses.
Missing identities/speakers and incomplete source text remain upstream limits.
Neither these changes nor retry counts establish a perfect match to reference
transcripts or cross-manga factual/prose reliability.
Embedded newlines in source quotations are normalized only for temporary decoding
constraints; unchanged source words still pass through the ordinary quote validator.
Physical `sign reads [exact transcribed text]` is accepted only when that exact
sign fact is current and the literal text immediately follows the attribution.
Captions, ungrounded sign readings, and other presentation wording remain rejected.
This acceptance relaxation preserves compatible checkpoints, whose segments are
revalidated on resume; it does not change the generation prompt fingerprint.
Temporary writer observations omit explicit drawing-effect clauses, duplicate
"another view of the same character" sentences, and size adjectives on background
people. Original descriptions and character evidence remain unchanged; real
lightning events and small objects are not removed by this presentation filter.
The default is one panel per writer request, independently of
multi-panel video display groups, to limit cross-panel speech/content leakage.
Single-word anonymous dialogue ending in ! or ? must survive as a brief quotation
in its own panel; missing interjections are sent back for repair.
This is
a targeted wording check, not a complete semantic guarantee. The prompt revision
invalidates older story checkpoints through the existing fingerprint mechanism.

The v10 fingerprint includes backend, GGUF path, model/runtime artifact size and
modification time, settings, source hashes, and identity revision. Changing from
the 2B PyTorch writer to the 4B GGUF writer archives old progress and published
files on the next production run; it does not delete them or silently relabel old
generations as outputs from the new model. Prompt/policy changes also archive
older progress on the next run; matching v10 runs resume normally.

The v10 writer scopes accepted text before prompting: publication metadata is
omitted, null-speaker dialogue/thought is separated as `unattributed_speech`, and
sound effects are context only, not lettering to read aloud. Other accepted text
retains its type and exact wording. The prepared evidence still retains every
accepted text fact. Narration must not invent speakers for unattributed lines or
assign a whole exchange to the only visible character. Prefer a flowing scene
with its setup/reveal over an OCR or panel-layout inventory. `allowed_reference`
is an appearance hint, not mandatory narration wording. These instructions are
not an independent factual check on the model's final prose.

## Implementation status

Implemented by `test_write_storry.py` and the reusable `story_writer` package.
Prompts contain no future-story outline or future character guide. The complete
ordered coverage list remains in the validated progress file, not the generation
prompt. Per-attempt output token counts, elapsed seconds, truncation, and failure
reasons are logged. Cloud description model selection does not change this local
text-only writer.
