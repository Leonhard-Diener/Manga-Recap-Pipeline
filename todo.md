# Planned training and identity work (not implemented)

- [ ] Automatically collect training candidates on every run for character ReID: source/page/panel/detection IDs, original and contextual crops, representations, matching evidence, proposed identities, and corrections. Keep predictions separate from verified labels; do not train on unreviewed groups as ground truth.
- [ ] Automatically collect Gemini input images/prompts, model/settings, raw responses, accepted structured outputs, failures, and corrections on every run for future Qwen fine-tuning. Preserve provenance and uncertainty; model acceptance is not a verified training label.
- [ ] Fine-tune/evaluate a manga ReID model using the collected data, including chibi views, back views, lookalikes, toys, and illustrations. Split evaluation by manga to avoid train/test leakage.
- [ ] Replace Gemini with a fine-tuned Qwen vision model once held-out evaluation supports it; no provider change yet.
- [ ] Extract character names from the manga and ground each name to the correct character, retaining uncertainty. Current generated gender-neutral aliases remain the fallback until this is implemented.

## Historical notes

* Earlier naming solution: random gender-neutral aliases (superseded as the long-term goal by grounded manga-name extraction above).
* fix how files select on which manga to run the script if multiple manga exist inside of input. SOLUTION: Takes the name inside of config (downloader also)
* create the test_write_storry.py based on the .md file  DONE
- optimize the result
- add text to speech
- combine thtat into a video

  Good:

  - All 54 pages returned valid complete JSON—no truncations.
  - Every expected panel was returned.
  - Character-label coverage technically worked: only 1 grounding warning total.
  - It did not invent transformations: 0 transformation events, which is better than fabricating them.

  Mid:

  - It often recognizes the broad scene correctly: people, wings, suits, phones, city/school settings, etc.
  - Basic visible-action records sometimes work, e.g. “girl holding phone.”
  - The page 18 dense scene is at least broadly understood as a hooded figure, a suited man, and phone-related
    interaction.

  Bad:

  - 310 of 319 character descriptors are literally "unclear labeled figure"—about 97%. So the C1/C2 grounding data
    exists but is barely useful.

  - Of 183 actions, only 9 have an actor label and only 6 a target label. 174 actions have neither. This means Qwen
    mostly avoids connecting actions to characters.

  - 12 pages reuse an identical description for multiple distinct panels. Chapter 2 page 18 is especially bad: it
    repeats one description across panels 1–3 and another across panels 4–5.

  - 22 panel descriptions end in obvious fragments such as "The.", "A.", or "says."

  Horrible:

  - Qwen repeatedly uses dialogue, narration, titles, and names even though the prompt explicitly forbids it. For
    example:
      - Chapter 1 page 2 says the Demon King is kneeling at the capsule machine. That is copied from the large narration
        box; the image itself mainly shows a boy and a close-up girl.

      - Chapter 1 page 5 describes dialogue and even infers an “underwear show,” rather than sticking to visible
        actions.

      - Chapter 2 page 1 names “Nukui Teppei” from printed text.

  - It classifies every page as story. Chapter 1 page 3 is plainly a chapter/title illustration and should be extra, not
    story.

  - On dense pages, it can confidently copy the wrong scene into several panels. That is worse than simply saying
    “unclear.”

  One part is also the detector’s fault: chapter 2 page 18 still has 12 labels, including some duplicate/partial boxes
  on the same people. But Qwen is the bigger problem here: even on simple pages with clean boxes, it chooses the safe
  fallback descriptor and reads dialogue instead of following the visual-only instruction.

  My honest rating:

  JSON / pipeline structure:      good
  Basic scene recognition:        mid
  Panel-by-panel accuracy:        bad
  Character-action grounding:     bad
  Following visual-only rules:    horrible
  Safe for story writer as-is:    no

  The output is usable as a rough visual reference, but not yet as trustworthy story evidence.

----------------------------------------------------------------------------------------------------

Cross-page transformation detection; transformations currently need to appear on the same page
