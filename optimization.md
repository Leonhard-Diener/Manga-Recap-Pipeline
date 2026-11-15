For test_description
• 
# Plan

  Keep Qwen3.5-2B for basic visual descriptions, but stop asking one generation to classify pages, describe panels,
  describe every character, extract actions, ignore text, and detect transformations simultaneously. Prevent the
  horrible failures using text masking, deterministic validation, fail-closed filtering, and selective panel retries.

  ## Scope

  - In: page-role mistakes, dialogue contamination, copied OCR text, repeated/wrong panel descriptions, fragmented
    sentences, unusable character grounding, and unsafe downstream data.

  - Out: replacing Qwen, improving already-acceptable broad scene recognition, ReID tuning, naming, and performance
    optimization.

  ## Action items

  [ ] Build an automated evaluation fixture from the observed failures: chapter 1 pages 2–5 and chapter 2 pages 1, 7,
  12, 18, 22, and 24. Record expected page roles, prohibited dialogue copying, unique panel descriptions, and minimum
  grounding requirements.

  [ ] Separate page-role classification from description generation. Use original unmasked pages and a tiny dedicated
  classification request, rather than making Qwen decide the role while producing a large JSON response.

  [ ] Add a high-confidence non-Qwen filter for obvious extras: use panel count, full-page illustration layout, detected
  text regions, and lightweight OCR only on likely title/credit pages. Recognized phrases such as “chapter,” “one-shot,”
  “credits,” and scanlation messages can override Qwen only when evidence is strong; uncertain pages remain included.

  [ ] Produce temporary text-masked previews from the text boxes RT-DETR already detects. Give these masked images to
  the description request so Qwen cannot copy dialogue, narration, titles, or character names. Preserve the original and
  grounded previews unchanged.

  [ ] Reduce the main Qwen task to panel descriptions and visible actions. Move character descriptors and transformation
  analysis out of the already overloaded page response, while preserving the same compact JSON and schema enforcement.

  [ ] Add deterministic semantic validation after generation: reject descriptions containing strong dialogue markers,
  OCR overlap, character labels such as C12, dangling fragments such as “The.”, and exact descriptions reused across
  visibly different panels. Structural JSON success must no longer count as semantic success.

  [ ] Retry only failed panels using their individual panel crops. A focused crop request should describe one panel
  without seeing neighboring dialogue or confusing panel mappings. Permit one retry; if it still fails, emit a safe
  generic description and no actions instead of contaminated evidence.

  [ ] Make actions fail closed. Drop dialogue-like actions, actions without an actor label, and actions whose actor box
  belongs to another panel. For panels containing exactly one reliable character label, allow deterministic actor
  assignment; for multi-character panels, retain an action only when a focused highlighted-character request identifies
  the actor.

  [ ] Handle character appearance descriptions separately using character crops or one highlighted character at a time.
  Do not allow "unclear labeled figure" as a successful descriptor; treat it as missing evidence. Limit retries to
  characters that ReID or recurrence data shows are important enough to reach the story writer.

  [ ] Add quality gates before writing the combined JSON: cap duplicate-description rate, OCR-overlap rate, fragment
  rate, unclear-descriptor rate, and ungrounded-action rate. Store rejected records with explicit reasons but prevent
  them from reaching the story writer as accepted evidence.

  [ ] Validate the revised design first on the failure fixture, then on both chapters. Compare against the current
  baseline of 310/319 unclear descriptors, 174/183 ungrounded actions, 12 pages with repeated panel descriptions, and
  one known title page misclassified as story.

  ## Open questions

  - None. Recommended defaults are one selective retry, fail closed for actions, and exclude pages only when the non-
    Qwen title/extra evidence is strong.