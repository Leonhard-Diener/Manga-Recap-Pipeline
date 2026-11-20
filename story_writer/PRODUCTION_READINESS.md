# Story-writer production gates

Scope: `story_writer/` and writer-specific tests only. Upstream evidence,
identities, shared configuration, video rendering and uploading are out of scope.
The existing public story/text/timeline contract and ordered panel coverage remain.

## Definition of done

There are two different claims: a writer can enforce its contracts, and a
particular generated story can pass factual/prose review. Neither implies that a
small model will produce a perfect story for every manga. A gate is not complete
merely because its instruction appears in a prompt or its output is valid JSON.

| Gate | Required result | Verification |
| --- | --- | --- |
| Current evidence | Only accepted, current evidence; no pending actions or inferred identities | Existing input validation plus regression tests |
| Plot coverage | Every required background, dialogue, thought, action and transformation is accounted for; optional visual detail omissions are explicit | Immutable fact ledger, sentence-to-fact coverage, separate semantic review, manual evaluation against source facts |
| Supported claims | No invented action, motive, causal connection, recipient, relationship or identity; transcribed claims remain claims | Sentence-level citations, role/modality checks, separate model review and independent manual review; adversarial regressions |
| Attribution | Known speakers retain their exact occurrence; unknown speech is never assigned to a visible person | Explicit speaker bindings, mechanical checks and semantic review |
| Names and references | Only permitted aliases identify body occurrences; source labels do not establish identity; pronouns have supported gender and clear antecedents | Occurrence-bound references, ambiguity regressions and prose review |
| Storytelling | Connected complete sentences about meaningful beats, not a roll call of bodies, poses, metadata or camera views | Evaluated paragraphs including unseen examples; no automatic approval from word counters |
| Dialogue and perspective | No quote dumps used to evade summarization, no unbalanced quotations, no first-person narrator leakage; unknown attribution can be preserved without inventing a speaker | Source-grounded speech paraphrase/quotation policy and regressions |
| Panel/timeline compatibility | Every original panel remains once, in order, with valid existing scene links and unchanged consumer contracts | Publication, resume and existing renderer-input compatibility tests; no renderer edits |
| Failure safety | Invalid/truncated/unreviewed candidates do not become final story facts; retries are bounded and all failures remain auditable | Failure injection, raw-attempt retention and publication tests |
| Resume correctness | Compatible accepted scenes are reused; source/prompt/schema/review changes invalidate stale work; saved text and evidence bindings revalidate | Fingerprint and checkpoint-corruption tests |
| Size and performance | Oversized inputs never cause silent evidence loss; normal scenes fit configured budgets; no identical failed-response loops | Token-budget tests and measured live runs using the existing runtime |
| Honest readiness | Evaluation states the corpus, failures, review limits and unresolved upstream blockers | Versioned evaluation report; never equate model self-review with proof |

## Required evaluation cases

### Release thresholds

- 100% of current source panels retained once and in order.
- 100% of required facts cited **and** preserved in source-based manual review;
  zero critical omissions or distortions. Optional omissions have recorded reasons.
- Zero unsupported events, causes, identities, recipients, gender assignments or
  speech attributions in the evaluated release; one failure blocks approval.
- Zero malformed sentences, broken quotes, perspective leaks, source framing or
  contradictory references in the reviewed release. No paragraph may pass merely
  by reciting quotes while dropping exposition.
- At least 95% of references eligible for grounded he/she or a permitted name use
  those forms. Unknown gender/identity is excluded from this denominator, reported
  separately, and never guessed to improve the percentage.
- Every saved scene revalidates bindings, source fingerprint, actual public text
  and exhaustive review records on resume; truncation/rejection/transport failure
  publishes no new final story. Failure tests must cover each path.
- All model-free tests pass. Review all 37 source pages of the target chapter,
  not only the four known regression pages, before approving that chapter. A
  broader production claim requires at least two additional independently reviewed
  chapters, including another manga; report corpus and failures rather than infer
  general reliability from chapter one.
- Record wall time, retries, input/output tokens and failures. There is no invented
  speed target: factual correctness outranks runtime, with existing bounded retries.

Audio/event synchronization and video quality are explicitly **not** writer-only
release gates: the writer preserves the consumer contract but cannot certify a
renderer that this task forbids changing.

- The previous chapter-one game exposition must not vanish behind spell quotes.
- The popularity-poll fact must survive first-person narrator repair.
- A vow, condition, threat, question or character belief must not become an
  unconditional world fact.
- A question about being an heir must not invent the recipient or reverse roles.
- A line with no speaker must not be assigned to the only visible character.
- A source caption containing a name must not become a confirmed body identity.
- Adjacent occurrences and visual similarity must not establish continuity.
- Routine poses must not become mandatory narration; meaningful quiet actions
  must not be silently removed by broad keyword filters.
- Newlines, contractions, curly/straight quotes, short interjections and sparse
  scenes must not break decoding or produce malformed prose.
- Saved progress with changed prose, citations, required facts or review results
  must not be accepted on a complete flag alone.
- A scene with only static optional evidence may be a hold; a scene with required
  story evidence may not.

## Implementation order

1. Reproduce known omissions, invented attribution and unsafe acceptance as tests.
2. Replace implicit evidence coverage with a compact source-linked fact ledger.
3. Generate sentences with explicit evidence/role bindings; compile existing
   public segments from validated sentences rather than trusting free-form IDs.
4. Review meaning against the original ledger separately from prose generation;
   reject omissions, unsupported additions, role swaps and invented antecedents.
5. Feed specific source-based failures into bounded repairs, not repeated bad prose
   or increasingly long lists of banned synonyms.
6. Revalidate checkpoints and add auditable quality reports before publication.
7. Run model-free regressions, then isolated live writer evaluations. Review actual
   source facts and prose, retry failures, and document anything still unachieved.

## Limits that must remain explicit

Writer-only changes cannot repair an incorrect upstream speaker/identity decision,
recover missing source text or invent missing story context. Model-generated
citations and a second model verdict are checks, not factual proof. Production
readiness must not be declared while the evaluated stories still have known
critical omissions, unsupported claims or attribution errors. Any unresolved gate
must be reported as a blocker rather than waived to finish the task.

## Current status

**Not production-ready.** The previous 104-panel draft completed structurally but
omitted meaningful exposition and contained role/attribution and prose errors.

Implemented in the scene path:

- Immutable source-linked facts, required/optional/excluded distinctions and exact
  occurrence endpoints. A broad pose-prefix filter no longer discards quiet actions.
- Sentence-level evidence bindings with required-fact coverage and speaker checks.
- Exhaustive, separate semantic-review requests and bounded source-based repairs.
- Raw requests/responses, validation diagnostics, ledger fingerprints and saved
  review records; resume recompiles and rechecks actual public narration.
- Existing story/text/timeline outputs remain; additive metadata explicitly says
  automated checks are not production approval. No renderer changes.
- Source quotes containing names are distinguished from narrator identity claims;
  quotation marks do not permit invented canonical-name dialogue.

### Live regression trial 1

Artifacts: `output/model_comparisons/production_writer_1795032998739496300/`.
This is the first evidence-ledger policy snapshot, before compact inputs and
same-scene candidate repairs. All four cases failed closed after four attempts;
no story was published. Total generation time was approximately 420 seconds.

| Source page | Observed failure |
| --- | --- |
| `00008.jpg` | Narration facts assigned to visible occurrences; optional poses still recited; source commentary and literal C keys in prose |
| `00010.jpg` | Popularity fact retained but the two other required narration facts omitted |
| `00028.jpg` | Unknown speech attached to a visible body; source labels treated as identity/role permission |
| `00039.jpg` | First-person fragments or malformed quotation punctuation |

These failures establish that the stricter gate catches defects, **not** that the
writer can produce acceptable prose. They must not be counted as quality passes.

Trial 2 (`output/model_comparisons/production_writer_1795033623134953200/`)
used compact inputs and same-scene draft repairs. It also failed all four pages:
role bindings/perspective still failed, and `00010.jpg` still omitted two required
facts. Approximately 354 seconds; no final story published. Repeating prose prompts
alone did not solve the problem. The third trial binds legal evidence/occurrence/
speaker combinations mechanically instead of asking the model to choose freely.

Trial 3 (`output/model_comparisons/production_writer_1795034111125679800/`)
also failed all four pages, approximately 495 seconds. Schema-level role choices
were constrained, but first-person copying, missing required facts and ungrounded
name use still blocked publication. Across the three policies, **0/12 page
evaluations passed**. These are repeated evaluations of four pages, not 12 unique
pages. No existing production story/video was replaced.

The third trial also tested the semantic reviewer separately on five synthetic
calibration cases. It rejected four deliberately bad candidates (omitted meaning
behind complete citations, a contradicted event, invented attribution/recipient,
and invented cause), but **also rejected the valid paraphrase control**. Its control
explanation incorrectly demanded a causal connection from a temporal "after".
In another explanation it confused invented candidate wording with original
evidence. Correct rejection labels alone do not establish sound review reasoning.
The current reviewer is **not calibrated for release approval**. Full records are
in that trial's `calibration_results.json` and raw request/response files.

Model-free tests cover failure safety; prose quality, same-model reviewer accuracy,
the full target chapter, the 95% grounded-reference target and unseen-manga
robustness remain unapproved. Oversized inputs fail explicitly without dropping
evidence; automatic scene splitting is not implemented.

Latest model-free verification: **361 tests passed**. Log:
`C:/Users/diene/AppData/Local/Temp/opencode/production-writer-suite-20261118-10.log`.
This verifies contracts and regressions, not factual/prose release approval.

### Release blockers

1. The tested local 4B writer does not reliably satisfy coverage, perspective and
   name permissions, even with explicit evidence and same-scene repairs.
2. The same-model reviewer rejects a supported paraphrase and can misread its own
   inputs. It must not be represented as an independent factual approval service.
3. Canonical-looking source labels do not establish body/speaker links. Unknown
   speakers and first-person context remain unknown; no identities were guessed.
4. A full accepted chapter, independent manual fact/prose review, the grounded
   pronoun/name percentage and unseen-manga evaluation have not been achieved.
5. Oversized scenes stop without evidence loss, but automatic splitting and the
   no-identical-failed-response-loop goal are not yet implemented beyond four
   bounded attempts. Retry bounds prevent infinite generation, not wasted retries.

The scene writer now fails closed on these evaluated defects. Consequently, a
normal scene run can stop where the previous writer published a defective draft.
Do not remove checks to recover a superficial completion rate. A stronger and
measured writer/reviewer backend or explicitly reviewed writer evidence/corrections
is needed before further release claims; neither was silently substituted here.
