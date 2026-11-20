1. Reliable panel extraction and reading order  
Irregular layouts, merged panels, borderless art, and right-to-left ordering made extraction difficult. Mistakes here affected descriptions, character matching, and narration.
 2. Keeping every stage grounded in current evidence  
Descriptions, text, characters, and events needed stable source references. Changes upstream had to invalidate outdated downstream results without unnecessarily rerunning everything.
 3. Maintaining character continuity  
Matching characters across poses, clothing changes, partial views, and art variations required conservative matching. Similar appearance could suggest an identity, but could not safely confirm it.
 4. Reading text and attributing it correctly  
Stylized lettering, sound effects, and ambiguous speakers complicated OCR. Correctly reading a sentence did not establish who said it—or whether its claim was true.
 5. Recognizing actions and transformations without inventing connections  
Nearby panels and similar appearances were insufficient evidence. Character endpoints and visible events needed grounding, with uncertain proposals kept separate from accepted facts.
 6. Preserving model output without letting cleanup damage it  
Attempts to improve responses through rewriting, keyword filtering, duplicate rejection, and placeholders sometimes removed valid information or changed meaning. Separating format validation from content alteration became essential.
 7. Making structured generation reliable across models and providers  
Malformed JSON, truncation, incorrect labels, and differing API capabilities caused failures. Strict schemas were not universally supported, and even local decoding constraints introduced bugs involving sentence-ending tokens, empty arrays, and spaces inside narration.
 8. Producing coherent narration without repeated failure loops  
The writer had to preserve panel order, identity permissions, and factual uncertainty while sounding natural. Conflicting dialogue rules, rigid length checks, and retries followed by recursive splitting made processing extremely slow.
 9. Resuming expensive runs safely  
Checkpoints needed to preserve completed work while rejecting stale evidence. Compatibility changes also exposed cases where the writer rejected fallback sentences it had previously saved.
10. Keeping processing practical on available hardware  
Local inference speed, GPU memory, prompt size, and output length constrained the workflow. Larger context windows did not guarantee reliable long responses; batching and retry policies needed real-world testing.

## Narration prompt experiments and remaining limitations (November 2026)

These are observed engineering results, not evidence of cross-manga reliability.
The usable chapter-one video completed 104 panels in 89 display groups (783.7
seconds). Completion and structural validation did not establish factual or prose
quality; no complete independent visual/factual review was performed.

### Problems diagnosed and mitigated

- **Copied context instead of new narration.** Supplying previous prose led the
  local writer to copy old scenes into unrelated panels. Previous prose was
  removed; compact alias/subject context remains, without earlier events.
- **Names repeated or reconstructed.** The writer repeated aliases, introduced
  them as "a boy named", or recovered names withheld for cadence. Names are now
  limited to once per segment followed by two segments without that alias.
  Name-related retries mask aliases in both prompt input and feedback, persistently
  across retry cycles. Aliases are not canonical names or gender evidence.
- **Retry feedback reinforced the mistake.** Replaying a failed sentence repeatedly
  reproduced guessed speakers. Unknown-speech repair now supplies the source lines
  rather than the failed attribution. Name repairs similarly avoid leaking names.
- **Unknown dialogue was dropped or guessed.** Being the only visible character
  did not establish authorship. Exact unassigned quotations, including short
  interjections, avoid repetitive "someone says" without inventing he/she/name.
  Known-speaker lines retain their exact grounded occurrence. Unknown thoughts
  are omitted from temporary prompts rather than converted to audible speech.
- **Conflicting instructions and sparse panels.** Third-person rules conflicted
  with first-person source quotations; minimum word counts encouraged padding.
  Source quotations are distinguished from narrator perspective, and short
  supported segments are allowed. Repeated instructions were consolidated in v34.
- **Appearance inventories.** Long character hints encouraged lists of clothes,
  hair, body parts, and poses. Temporary references now prefer supported pronouns
  or a role with at most one trait; saved evidence remains unchanged. Groups
  remain neutral/plural and job titles do not prove gender.
- **Source presentation became story content.** Captions, sound-effect inventories,
  camera reports, and drawing-scale descriptions leaked into narration. Targeted
  rejection and temporary-input repairs reduce these failures without deleting
  panels, editing accepted source evidence, or inventing in-world causes.
- **Constrained decoding had runtime-specific bugs.** Quote-prefix patterns caused
  unescaped ASCII quotes; a wildcard could consume JSON field boundaries. Curly
  quotes and a restricted continuation fixed the observed native-decoding failures.
  Speech-only retries can constrain text to validated source quotations; mixed
  panels retain free continuation for accepted actions/known speech. Consecutive
  quotations also exposed a validator boundary-consumption bug, now regression-tested.
- **Retries were not a guarantee of success.** Unlimited repeated failures could
  stall generation. HTTP and single-panel failure cycles are bounded and preserve
  checkpoints/raw attempts. A no-progress watchdog stops owned stage processes.
  These limits bound failure cost; they do not prove successful completion.

### What the isolated first-ten-page experiment actually showed

The v34 trial covered all 22 included panels from the first ten source pages,
retaining existing cover/publication exclusions. Segments containing "someone"
dropped from 5 to 0. A narrow appearance-keyword counter dropped from 5 to 2 but
missed hyphenated appearances and framing/body language. These counts do not
establish the requested roughly 95% pronoun/name reference rate or natural flow.
All 295 model-free tests passed at that checkpoint; the prose review still failed.

Remaining examples included "A person stands with legs visible", "The blonde
character is shown in a close-up view", "touches his hair in profile", "as a
caption box declares", and an unquoted first-person narrator sentence. Replacing
"someone" with "a voice" also improved the counter without solving the style.
Lexical validators missed some source commentary; structural JSON validity is
not semantic validity. The current chapter video was accepted as usable, not ideal.

### Unresolved or deliberately deferred

- One narration segment per panel encourages disconnected captions for static
  fragments (legs, hair, doorways). Scene/page-level prose with retained panel
  grounding was proposed but **not implemented**. Further prompt tuning was stopped
  after the user judged the result usable on the available small models.
- Missing/wrong speaker evidence and split or incorrect ReID groups still constrain
  safe references. A prose prompt cannot fix them by guessing continuity or gender.
- Canonical manga-name extraction, training-data export, fine-tuning, and replacing
  Gemini with a specialist local model remain future work, not implemented features.
- No measured 95% reference compliance, general cross-manga quality guarantee,
  exhaustive factual/video review, or larger-model comparison is claimed.
- Completed historical narration is not silently treated as regenerated under a
  new prompt. Trials preserve production artifacts and archive raw failures.

## Follow-up: reference-transcript style trials (18 November 2026)

The user supplied two recap transcripts emphasizing continuous exposition,
action/reaction, clear stakes and natural transitions. Their story facts were not
used as evidence for this manga. Further local-model trials were explicitly requested.

- Removing the demand to speak over every static panel helped avoid some forced
  captions, but optional silence alone did not create coherent prose. V35 and v36
  ten-page trials still described camera views and bodies, or used filler such as
  "a thought surfaces" and "the narrator". These trials failed prose review.
- Page-level prose replaced individual panel captions in the operational CLI.
  Source pages remain ordered; all panels remain in the video/grounding timeline.
  Shared paragraph audio is divided across its same-page panels using measured
  TTS duration, not synthesized repeatedly or padded with per-panel silence. This
  is scene-level timing, not word-/event-exact visual synchronization.
- Model inputs containing every visible occurrence still encouraged a roll call,
  even when the instructions said not to enumerate people. The latest scene input
  includes only known speech and retained event endpoints as character hints.
  Routine pose/gaze/grooming actions are omitted from temporary prose facts while
  source evidence remains unchanged. This heuristic can miss subtle importance;
  it is not a learned or reviewed scene-importance classifier.
- Strict third-person checking initially rejected single-quoted dialogue containing
  contractions such as "I'll". The quote scanner was fixed and regression-tested.
  Known first-person thoughts that resisted paraphrasing can retain exact quotes
  with their grounded thinker. Unidentified first-person narrator context is not
  assigned to a nearby person just to produce he/she wording.
- A source line containing an embedded newline broke native pattern decoding.
  Whitespace normalization in temporary quotation constraints repaired it without
  editing source evidence or discarding completed checkpoints.
- A subsequent chapter run reached 76 of 104 panels before a CUDA out-of-memory
  failure. GPU offloading was reduced to leave desktop headroom on the 8GB card;
  the same model and acceptance rules were retained, and its checkpoint was resumed.
  More conservative hardware settings do not improve prose quality by themselves.
- Page-level trials still produced omissions, repetitive clauses ("the character
  stands while he stands"), generic role labels, and occasional unsupported-looking
  attribution. The validators check structural IDs, allowed names they know about,
  lexical style and some dialogue constraints; they do not prove complete semantic
  coverage, correct pronoun antecedents, or canonical name-to-speaker links.
  Unknown source names are not comprehensively recognized by the name validator.
- The first complete scene-based chapter retained 104 panels in 37 page passages.
  A further pass was requested after review, reducing irrelevant actor/pose hints.
  The earlier complete video, YouTube upload and full-manga artifacts were retained.

No perfect transcript-style match, robust semantic coverage scorer, automatic
cross-page scene segmentation, canonical identity repair or larger-model upgrade
was completed as part of these experiments. More retries alone are not evidence
that the remaining factual and writing problems are solved.
