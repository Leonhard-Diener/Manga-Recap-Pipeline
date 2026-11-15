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
