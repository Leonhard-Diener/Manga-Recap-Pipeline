# Story Writer Specification

## Goal

Create `test_write_storry.py`. It uses the information produced by the existing manga-analysis scripts to write an English YouTube recap.

The narration must:

- use third person;
- use present tense;
- sound like one uninterrupted story;
- stay faithful to the supplied information;
- give each displayed panel about 10 seconds of spoken narration when read by text-to-speech at normal speed.

At a default narration speed of 150 words per minute, 10 seconds is about 25 spoken words per panel. This word count is only an estimate used before audio exists. The script may adjust sentence boundaries slightly so the narration sounds natural, while keeping the average close to 10 seconds of speech per panel.

## Inputs

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

ReID later maps each instance ID to a visual group, accepted entity, and permitted name. The story writer may use a proper name only when that exact instance is confirmed as the accepted entity and the name has already been revealed at that point in the story. Otherwise, it must use a stable visual description such as `the horned girl`.

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
- Introduce a name only when the accepted data permits it at that point in the story.
- Use the same clear visual description for an unnamed character until a name is safely available.
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
permit. A failed chunk is never recursively split or replaced by automatic
fallback text: save diagnostics and completed progress, then stop without
publishing an incomplete story. Input-sized chunk planning still happens before
generation. Short observations of three or more words are valid regardless of
the density label; do not demand invented padding to satisfy a timing target.
Third-person narration and concise indirect speech are prompt-level style guidance,
not a hard pronoun filter. Brief attributed quotations of supplied dialogue may
contain "I" or "you" without changing the narrator's perspective. The validator
does not reject or rewrite such strings. Grounding, name permissions, temporal
references, coverage, nonempty text, and the existing segment-length limit remain
validated. No special perspective-correction path or extra retries are used.
Existing source-linked fallback records remain validated and resumable.

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
