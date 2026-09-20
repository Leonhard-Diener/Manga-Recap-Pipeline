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
- evidence or confidence information.

A transformation should contain both the source instance ID and target instance ID, together with their panel IDs. The two appearances remain separate visual groups. Only an accepted transformation decision may connect them as different forms of the same underlying entity.

ReID later maps each instance ID to a visual group, accepted entity, and permitted name. The story writer may use a proper name only when that exact instance is confirmed as the accepted entity and the name has already been revealed at that point in the story. Otherwise, it must use a stable visual description such as `the horned girl`.

This creates the required evidence chain:

```text
panel -> detected instance -> grounded action or event -> ReID entity -> permitted name -> narration
```

If a panel contains several characters and the acting instance is uncertain, the writer must use neutral wording or omit the uncertain attribution. It must never select a name based only on which known characters occur elsewhere on the page.

## Qwen Usage

Load the installed text model once when `test_write_storry.py` starts. Pass text and structured JSON to it, without images. This must remain separate from `test_character_reid.py`; the character ReID script must never launch Qwen.

If the prepared manga information fits within about 48,000 input tokens, send it in one request. Otherwise, split it into scenes or chapters of about 10,000 to 12,000 input tokens, with a hard target below 16,000 tokens per request.

Each later request receives:

- a short outline of the complete story;
- the fixed character and name list;
- the current story state and unresolved events;
- the IDs of panels already covered;
- the last 250 to 400 words of narration for style and continuity;
- the next ordered group of panels to write.

The previous narration is context only and must not be repeated. After all parts are written, run a small boundary-editing pass to remove repetition and awkward joins without changing facts, panel order, names, or panel references.

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

The narration must be divided into ordered segments. Each segment has one primary panel and enough text to produce about 10 seconds of speech at normal text-to-speech speed, normally around 25 words at 150 words per minute.

The displayed panel changes when the next segment's spoken audio begins. The story should continue naturally across these switches. The timeline stores word positions and estimated start and end times so a later video script knows when to change panels.

Estimated timing is sufficient before voice generation. When real text-to-speech timestamps become available, the video stage replaces the estimates with measured audio timestamps. Each panel then remains visible for the actual duration of its assigned narration segment, with a target of about 10 seconds.

## Outputs

Write these files:

- `<manga>_story.txt`: the uninterrupted narration;
- `<manga>_story.json`: narration segments with text, chapter, primary panel ID, and supporting panel IDs;
- `<manga>_panel_timeline.json`: ordered panel switches with word ranges and estimated start and end times.

The JSON output must also record the input file hashes, identity revision, prompt version, model version, narration speed, and generation settings. This makes runs reproducible and prevents resuming from stale data.

## Reliability

Save progress after every completed chunk so generation can resume after an interruption. Validate that all referenced panel and instance IDs exist, the named entity is confirmed for the cited instance, all panels remain in order, names respect their reveal points, and no output chunk repeats already covered panels. If Qwen returns invalid JSON, retry that chunk once with a correction prompt and preserve the original response for debugging.
