import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from story_entities.manifest import file_hash
from story_entities.matching import stable_id
from story_writer.writer import (
    _boundary_edit,
    _timeline,
    _initial_chunks,
    _load_progress,
    _restricted_fallback,
    _story_response_schema,
    _panel_prompt_record,
    build_story_prompt,
    generate_story,
    generate_response,
    prepare_story,
    validate_story_response,
)


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, list[Path], Path]:
    manga = root / "input" / "Example Manga"
    chapter = manga / "chapter_001"
    chapter.mkdir(parents=True)
    source = chapter / "00001.jpg"
    source.write_bytes(b"source page")

    output = root / "output"
    chapter_output = output / manga.name / chapter.name
    chapter_output.mkdir(parents=True)
    Image.new("RGB", (640, 480), "white").save(chapter_output / "panel_0001.jpg")
    source_hash = file_hash(source)
    geometry = "geometry_test"
    grounding = "grounding_test"
    panel_id = "panel_test_1"
    _write_json(chapter_output / "panel_manifest.json", {
        "schema_version": 2,
        "pages": [{
            "page": str(source.resolve()),
            "page_order": 0,
            "source_hash": source_hash,
            "geometry_revision": geometry,
            "grounding_revision": grounding,
            "character_labels": [{
                "label": "C1",
                "detection_id": "detection_1",
                "panel_number": 1,
            }],
            "panels": [{"panel_number": 1, "panel_id": panel_id}],
        }],
    })
    _write_json(output / f"{manga.name}_descriptions.json", {
        "descriptions": {chapter.name: {source.name: {
            "extraction_status": "complete",
            "page_role": "story",
            "source_hash": source_hash,
            "geometry_revision": geometry,
            "grounding_revision": grounding,
            "panels": [{
                "panel_number": 1,
                "description": "A horned student runs through the school corridor.",
                "actions": [{"actor_label": "C1", "text": "runs through the corridor"}],
            }],
        }}},
        "page_filter": {
            "schema_version": 1,
            "policy": "story_only_v1",
            "chapters": {chapter.name: {source.name: {
                "included_in_story": True,
                "source_hash": source_hash,
                "geometry_revision": geometry,
                "grounding_revision": grounding,
                "extraction_status": "complete",
            }}},
        },
    })
    description_export = json.loads(
        (output / f"{manga.name}_descriptions.json").read_text(encoding="utf-8")
    )
    _write_json(chapter_output.parent / "character_classification.json", {
        "schema_version": 2,
        "revision": "revision_test",
        "description_fingerprint": stable_id(
            "descriptions_", description_export["descriptions"]
        ),
        "characters": [{
            "instance_id": "instance_1",
            "panel_id": panel_id,
            "chapter": chapter.name,
            "page": str(source.resolve()),
            "page_hash": source_hash,
            "geometry_revision": geometry,
            "grounding_revision": grounding,
            "entity_id": "entity_1",
            "form_id": "form_1",
            "visual_group_id": "group_1",
            "identity_status": "accepted",
            "name": None,
            "naming_allowed": False,
            "display_name": "Akira",
            "display_name_allowed": True,
            "stable_description": "horned student",
            "grounding_ids": ["detection_1"],
        }],
        "story_events": [{
            "event_id": "event_1",
            "panel_id": panel_id,
            "text": "runs through the corridor",
            "actor_instance_id": "instance_1",
            "target_instance_id": None,
            "actor_display_name": "Akira",
            "actor_description": "horned student",
            "confidence_percent": 100,
            "acceptance_policy": "reviewed_visible_action_v1",
            "status": "accepted_reviewed",
        }],
        "transformation_proposals": [],
    })
    return manga, [chapter], output


class StoryWriterTests(unittest.TestCase):
    def test_prepare_story_joins_current_grounded_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)

            self.assertEqual([row["panel_id"] for row in prepared["panels"]], ["panel_test_1"])
            panel = prepared["panels"][0]
            self.assertEqual(panel["characters"][0]["allowed_name"], "Akira")
            self.assertEqual(panel["characters"][0]["allowed_reference"], "Akira")
            self.assertEqual(panel["accepted_actions"][0]["actor_instance_id"], "instance_1")
            self.assertEqual(prepared["identity_revision"], "revision_test")

    def test_prepare_story_accepts_only_revision_bound_reviewed_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            classification_path = output / manga.name / "character_classification.json"
            classification = json.loads(classification_path.read_text(encoding="utf-8"))
            classification["reviewed_evidence"] = {"reviewed_text_facts": [{
                "status": "accepted_reviewed", "fact_id": "fact_1",
                "panel_id": "panel_test_1", "kind": "dialogue",
                "text": "We have to leave now.", "speaker_instance_id": "instance_1",
                "source_hash": file_hash(chapters[0] / "00001.jpg"),
                "geometry_revision": "geometry_test",
                "grounding_revision": "grounding_test",
            }]}
            _write_json(classification_path, classification)
            prepared = prepare_story(manga, chapters, output)
            fact = prepared["panels"][0]["accepted_text_facts"][0]
            self.assertEqual(fact["text"], "We have to leave now.")
            self.assertEqual(fact["acceptance_policy"], "reviewed_text_fact_v1")

    def test_prepare_story_accepts_revision_bound_automatic_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            classification_path = output / manga.name / "character_classification.json"
            classification = json.loads(classification_path.read_text(encoding="utf-8"))
            classification["text_proposals"] = [{
                "status": "accepted_automatic", "proposal_id": "text_1",
                "acceptance_policy": "grounded_text_transcription_v1",
                "panel_id": "panel_test_1", "kind": "dialogue",
                "text": "We have to leave now.", "speaker_instance_id": "instance_1",
                "chapter": chapters[0].name, "page": "00001.jpg",
                "source_hash": file_hash(chapters[0] / "00001.jpg"),
                "geometry_revision": "geometry_test",
                "grounding_revision": "grounding_test",
            }]
            _write_json(classification_path, classification)
            prepared = prepare_story(manga, chapters, output)
            fact = prepared["panels"][0]["accepted_text_facts"][0]
            self.assertEqual(fact["text"], "We have to leave now.")
            self.assertEqual(fact["acceptance_policy"], "grounded_text_transcription_v1")
            self.assertEqual(fact["truth_status"], "transcribed_statement_only")

    def test_writer_scopes_unknown_speech_and_excludes_only_typed_publication_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            classification_path = output / manga.name / "character_classification.json"
            classification = json.loads(classification_path.read_text(encoding="utf-8"))
            rows = [
                ("publication", "TWO-PART SPECIAL ONE-SHOT", None),
                ("narration", "A special one-shot attack breaks the wall.", None),
                ("dialogue", "Unauthorized flying is banned!", None),
                ("thought", "What happened?", None),
                ("dialogue", "Anyone who sees it dies.", "instance_1"),
                ("sound_effect", "BAM", None),
                ("sign", "School entrance", None),
            ]
            classification["text_proposals"] = [{
                "status": "accepted_automatic", "proposal_id": f"text_{index}",
                "acceptance_policy": "grounded_text_transcription_format_v2",
                "panel_id": "panel_test_1", "kind": kind, "text": text,
                "speaker_instance_id": speaker, "chapter": chapters[0].name, "page": "00001.jpg",
                "source_hash": file_hash(chapters[0] / "00001.jpg"),
                "geometry_revision": "geometry_test", "grounding_revision": "grounding_test",
            } for index, (kind, text, speaker) in enumerate(rows)]
            _write_json(classification_path, classification)
            panel = prepare_story(manga, chapters, output)["panels"][0]
            self.assertEqual(len(panel["accepted_text_facts"]), 7)
            record = _panel_prompt_record(panel)
            self.assertEqual([fact["text"] for fact in record["accepted_text_facts"]],
                             [rows[4][1], rows[6][1]])
            self.assertEqual(record["story_context"], [rows[1][1]])
            self.assertEqual([fact["text"] for fact in record["anonymous_dialogue"]],
                             [rows[2][1]])
            self.assertNotIn(rows[3][1], json.dumps(record))
            self.assertNotIn("BAM", json.dumps(record))
            self.assertNotIn(rows[0][1], json.dumps(record))
            prompt = build_story_prompt([panel], completed_panels=[], previous_narration="",
                                        tokenizer=FakeTokenizer())
            self.assertIn("anonymous_dialogue has NO known", prompt)
            self.assertIn("not mandatory", prompt)
            self.assertIn("then prefer pronouns", prompt)
            self.assertIn("story_context supplies narration", prompt)
            self.assertIn("prefer a standalone exact quotation", prompt)
            self.assertIn("Never guess a visible character", prompt)
            self.assertIn("an audible voice from a thought", prompt)
            self.assertNotIn("retain meaningful content as an unattributed warning/comment", prompt)
            self.assertNotIn(rows[0][1], prompt)

    def test_source_commentary_is_rejected_but_direct_narration_is_accepted(self):
        panel = {"panel_id": "p", "characters": []}
        bad = ["Narration captions hail this figure as the strongest.",
               "An unattributed warning states that flying is banned.",
               "An exclamation appears above the building.",
               "A close-up shows a character staring forward.",
               "A shot displays the school building.",
               "A shot of the school appears.",
               "Signs declare him the Demon King.",
               "Signs above read Demon King.",
               "Another view of the same student stands in the background.",
               "A title declares the demon king the strongest.",
               "Someone shouts as the scene shifts to the school.",
               "Someone shouts as the scene shows a building.",
               "An anonymous voice warns that flying is banned.",
               "Someone shouts with a sound effect.",
               "The caption reads that he has never lost."]
        good = ["The strongest warrior has never been defeated.",
                "Someone warns that flying is banned.",
                "Someone shouts, 'Hey!'", "He reads a warning on the door."]
        for text in bad + good:
            with self.subTest(text=text):
                raw = json.dumps({"segments": [{
                    "primary_panel_id": "p", "supporting_panel_ids": ["p"],
                    "text": text, "used_instance_ids": [], "name_uses": [],
                }]})
                _, errors = validate_story_response(raw, [panel])
                self.assertEqual(bool(errors), text in bad)

    def test_long_previous_chunk_paragraph_cannot_be_reused_for_a_different_panel(self):
        text = "A student kneels in front of the toy machine while another watches from nearby."
        panel = {"panel_id": "new", "characters": []}
        raw = json.dumps({"segments": [{
            "primary_panel_id": "new", "supporting_panel_ids": ["new"],
            "text": text, "used_instance_ids": [], "name_uses": [],
        }]})
        _, errors = validate_story_response(raw, [panel], previous_segments=[{"text": text}])
        self.assertTrue(any("repeats previously completed narration" in error for error in errors))

    def test_reference_style_rejects_clones_name_spam_and_drawing_language(self):
        panel = {"panel_id": "p", "characters": [{"instance_id": "i", "allowed_name": "Yamazaki"}]}
        bad = ["Yamazaki points while another Yamazaki stands behind him.",
               "Yamazaki looks down. Yamazaki frowns. Yamazaki turns away.",
               "Yamazaki recoils. Someone shouts a warning. Yamazaki threatens to fight.",
               "A boy named Yamazaki looks down.",
               "She stands amidst lightning effects.",
               "A group of small characters stands nearby."]
        good = ["Yamazaki looks down. He frowns and turns away.",
                "She stands near the people in the background."]
        for text in bad + good:
            uses_name = "Yamazaki" in text
            raw = json.dumps({"segments": [{
                "primary_panel_id": "p", "supporting_panel_ids": ["p"], "text": text,
                "used_instance_ids": ["i"] if uses_name else [],
                "name_uses": [{"name": "Yamazaki", "instance_id": "i"}] if uses_name else [],
            }]})
            with self.subTest(text=text):
                _, errors = validate_story_response(raw, [panel])
                self.assertEqual(bool(errors), text in bad, errors)

    def test_reference_context_carries_aliases_not_previous_scene_prose(self):
        base = {"chapter": "chapter", "page": "page", "panel_number": 1,
                "description": "A student waits.", "accepted_actions": [],
                "accepted_transformations": [], "characters": [{"instance_id": "i", "allowed_name": "Yamazaki"}]}
        previous = [{"panel_id": "old", **base}]
        segments = [{"used_instance_ids": ["i"], "name_uses": [{"name": "Yamazaki", "instance_id": "i"}]}]
        prompt = build_story_prompt([{**base, "panel_id": "new"}], completed_panels=previous,
                                    previous_narration="An unrelated dragon breathes fire.",
                                    previous_segments=segments, tokenizer=FakeTokenizer())
        context = json.loads(prompt.split("Grounded input:\n")[1].split("\nGenerate exactly")[0])
        self.assertIsNone(context["reference_context"]["last_unambiguous_subject"])
        self.assertEqual(context["reference_context"]["previously_used_aliases"], [])
        self.assertNotIn("Yamazaki", prompt)
        self.assertNotIn("An unrelated dragon", prompt)
        unnamed = {**base, "panel_id": "new", "characters": [{"instance_id": "unknown", "allowed_name": None}]}
        prompt = build_story_prompt([unnamed], completed_panels=previous,
                                    previous_narration="", previous_segments=segments, tokenizer=FakeTokenizer())
        context = json.loads(prompt.split("Grounded input:\n")[1].split("\nGenerate exactly")[0])
        self.assertIsNone(context["reference_context"]["last_unambiguous_subject"])
        self.assertEqual(context["reference_context"]["previously_used_aliases"], [])
        self.assertNotIn("Yamazaki", prompt)

    def test_writer_input_omits_drawing_conventions_without_mutating_evidence(self):
        panel = {"chapter": "chapter", "page": "page", "panel_number": 1, "panel_id": "p",
                 "description": "A horned female character stands amidst electrical lightning and impact effects near a smaller group of characters.",
                 "characters": [{"instance_id": "i", "allowed_reference": "the group of small characters",
                                 "visual_description": "Small characters standing nearby"}],
                 "accepted_actions": [], "accepted_transformations": []}
        before = json.dumps(panel)
        record = _panel_prompt_record(panel)
        self.assertEqual(record["description"], "A horned female character stands near a group of characters.")
        self.assertEqual(record["characters"][0]["allowed_reference"], "the group")
        self.assertEqual(json.dumps(panel), before)
        panel["description"] = ("A student shouts and points forward. Another full-body view of the same character stands in the background.")
        self.assertEqual(_panel_prompt_record(panel)["description"], "A student shouts and points forward.")
        panel["description"] = "Lightning strikes a tree. A small toy falls."
        self.assertEqual(_panel_prompt_record(panel)["description"], panel["description"])
        panel["description"] = "A miniature version of a character stands nearby."
        panel["characters"][0]["allowed_reference"] = "a miniature character standing upright"
        panel["characters"][0]["visual_description"] = "a miniature character standing upright"
        panel["accepted_text_facts"] = [{"kind": "dialogue", "speaker_instance_id": "i", "text": "Hello."}]
        record = _panel_prompt_record(panel)
        self.assertEqual(record["description"], "A character stands nearby.")
        self.assertEqual(record["accepted_text_facts"][0]["speaker_reference"], "the character")

    def test_recent_alias_is_hidden_from_prompt_schema_and_action_references(self):
        panel = {"chapter": "chapter", "page": "page", "panel_number": 1, "panel_id": "new",
                 "description": "A girl looks down.", "accepted_actions": [{"actor_reference": "Yamazaki"}],
                 "accepted_transformations": [], "characters": [{"instance_id": "i", "allowed_name": "Yamazaki",
                    "allowed_reference": "Yamazaki", "visual_description": "a girl with horns"}]}
        previous = [{"text": "Yamazaki looks away.", "name_uses": [{"name": "Yamazaki", "instance_id": "old"}]}]
        before = json.dumps(panel)
        prompt = build_story_prompt([panel], completed_panels=[], previous_narration="",
                                    previous_segments=previous, tokenizer=FakeTokenizer())
        schema = _story_response_schema([panel], previous)
        self.assertNotIn("Yamazaki", prompt)
        self.assertNotIn("Yamazaki", json.dumps(schema))
        self.assertIn('"allowed_reference":"she"', prompt)
        self.assertEqual(json.dumps(panel), before)
        raw = json.dumps({"segments": [{"primary_panel_id": "new", "supporting_panel_ids": ["new"],
            "text": "Yamazaki looks down.", "used_instance_ids": ["i"],
            "name_uses": [{"name": "Yamazaki", "instance_id": "i"}]}]})
        for history in (previous, previous + [{"text": "The door opens.", "name_uses": []}]):
            _, errors = validate_story_response(raw, [panel], previous_segments=history)
            self.assertTrue(any("name-frequency" in e for e in errors), errors)
        _, errors = validate_story_response(raw, [panel], previous_segments=previous + [
            {"text": "The door opens.", "name_uses": []}, {"text": "She walks inside.", "name_uses": []}])
        self.assertFalse(errors, errors)

    def test_prompt_identity_ids_do_not_leak_names_and_canonical_names_are_rejected(self):
        panel = {"chapter": "chapter", "page": "page", "panel_number": 1, "panel_id": "p",
                 "description": "A girl looks down.", "accepted_actions": [], "accepted_transformations": [],
                 "characters": [{"instance_id": "i", "entity_id": "entity_lucia", "form_id": "form_lucia_school",
                                 "allowed_name": "Yamazaki", "allowed_reference": "Yamazaki"}]}
        prompt = build_story_prompt([panel], completed_panels=[], previous_narration="", tokenizer=FakeTokenizer())
        self.assertNotIn("lucia", prompt.lower())
        self.assertIn('"instance_id":"i"', prompt)
        raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
            "text": "Lucia looks down.", "used_instance_ids": ["i"], "name_uses": []}]})
        _, errors = validate_story_response(raw, [panel], known_names={"Lucia"})
        self.assertTrue(any("does not ground the name" in e for e in errors), errors)

    def test_self_introduction_uses_alias_or_title_without_changing_saved_speech(self):
        panel = {"chapter": "chapter", "page": "page", "panel_number": 1, "panel_id": "p",
                 "description": "A girl introduces herself.", "accepted_actions": [], "accepted_transformations": [],
                 "characters": [{"instance_id": "i", "allowed_name": "Yamazaki", "allowed_reference": "Yamazaki"}],
                 "accepted_text_facts": [{"kind": "dialogue", "speaker_instance_id": "i",
                                          "text": "I AM LUCIA, THE DEMON KING."}]}
        before = json.dumps(panel)
        record = _panel_prompt_record(panel, ["Lucia"])
        self.assertEqual(record["accepted_text_facts"][0]["text"], "I AM Yamazaki, THE DEMON KING.")
        prompt = build_story_prompt([panel], completed_panels=[], previous_narration="", tokenizer=FakeTokenizer(),
                                    canonical_names=["Lucia"], suppress_names=True)
        self.assertNotIn("LUCIA", prompt)
        self.assertNotIn("Yamazaki", prompt)
        self.assertIn("I am THE DEMON KING.", prompt)
        self.assertEqual(json.dumps(panel), before)
        panel["accepted_text_facts"][0]["text"] = "I AM LUCIA."
        panel["characters"][0]["allowed_name"] = None
        self.assertEqual(_panel_prompt_record(panel, ["Lucia"])["accepted_text_facts"][0]["text"],
                         "The speaker introduces themself.")

    def test_known_speech_uses_its_reference_while_unknown_speech_stays_anonymous(self):
        panel = {"chapter": "chapter", "page": "page", "panel_number": 1, "panel_id": "p",
                 "description": "A boy looks surprised.", "accepted_actions": [], "accepted_transformations": [],
                 "characters": [{"instance_id": "i", "allowed_name": None, "allowed_reference": "he"}],
                 "accepted_text_facts": [{"kind": "dialogue", "speaker_instance_id": "i", "text": "Is that really true?"}]}
        record = _panel_prompt_record(panel)
        self.assertEqual(record["accepted_text_facts"][0]["speaker_reference"], "he")
        for text, invalid in [("He asks whether it is really true.", False),
                              ("Someone else asks whether it is really true.", True),
                              ("He looks surprised. Someone asks whether it is true.", True),
                              ("He says that someone warns her about the danger.", False)]:
            raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
                "text": text, "used_instance_ids": ["i"], "name_uses": []}]})
            _, errors = validate_story_response(raw, [panel])
            self.assertEqual(bool(errors), invalid, errors)
        panel["accepted_text_facts"][0]["speaker_instance_id"] = None
        record = _panel_prompt_record(panel)
        self.assertFalse(record["accepted_text_facts"])
        self.assertIsNone(record["anonymous_dialogue"][0]["speaker_reference"])
        raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
            "text": "Someone asks whether it is really true.", "used_instance_ids": [], "name_uses": []}]})
        _, errors = validate_story_response(raw, [panel])
        self.assertFalse(errors, errors)
        panel["accepted_text_facts"].append({"kind": "dialogue", "speaker_instance_id": "i", "text": "It is difficult."})
        raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
            "text": "He says it is difficult while asking whether it is really true.",
            "used_instance_ids": ["i"], "name_uses": []}]})
        _, errors = validate_story_response(raw, [panel])
        self.assertTrue(any("unknown dialogue explicitly anonymous" in error for error in errors), errors)

    def test_exact_grounded_physical_sign_does_not_block_narration(self):
        panel = {"panel_id": "p", "characters": [],
                 "accepted_text_facts": [{"kind": "sign", "text": "TOY"}]}
        for text, invalid in [("The sign reads TOY.", False),
                              ("The sign reads 'TOY'.", False),
                              ("The sign reads DANGER near a toy machine.", True),
                              ("The sign reads TOY. A caption declares him the strongest.", True),
                              ("A caption reads TOY.", True)]:
            raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
                "text": text, "used_instance_ids": [], "name_uses": []}]})
            with self.subTest(text=text):
                _, errors = validate_story_response(raw, [panel])
                self.assertEqual(bool(errors), invalid, errors)

    def test_name_frequency_retry_changes_input_instead_of_replaying_named_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            named = json.dumps({"segments": [{"primary_panel_id": "panel_test_1", "supporting_panel_ids": ["panel_test_1"],
                "text": "Akira turns away. Akira looks down.", "used_instance_ids": ["instance_1"],
                "name_uses": [{"name": "Akira", "instance_id": "instance_1"}]}]})
            corrected = json.dumps({"segments": [{"primary_panel_id": "panel_test_1", "supporting_panel_ids": ["panel_test_1"],
                "text": "She turns away and looks down.", "used_instance_ids": ["instance_1"], "name_uses": []}]})
            with patch("story_writer.writer.generate_response", side_effect=[
                (named, {"truncated": False}), (corrected, {"truncated": False}),
            ]) as calls:
                result = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(result["segments"][0]["text"], "She turns away and looks down.")
            self.assertNotIn("Akira", calls.call_args_list[1].args[2])
            self.assertNotIn("Akira", json.dumps(calls.call_args_list[1].kwargs["response_schema"]))

    def test_repeated_narration_is_allowed_for_identical_current_descriptions(self):
        description = "A steaming food steamer basket is lifted to reveal a head with horns inside."
        text = "A basket is lifted through rising steam to reveal a horned head resting inside."
        previous = {"panel_id": "old", "characters": [], "description": description}
        current = {"panel_id": "new", "characters": [], "description": description}
        for narration in (description, text):
            raw = json.dumps({"segments": [{
                "primary_panel_id": "new", "supporting_panel_ids": ["new"],
                "text": narration, "used_instance_ids": [], "name_uses": [],
            }]})
            prior = [{"primary_panel_id": "old", "text": narration}]
            _, errors = validate_story_response(raw, [current], supporting_panels=[previous, current],
                                                 previous_segments=prior)
            self.assertFalse(errors, errors)
            current["description"] = "A student runs down the school corridor."
            _, errors = validate_story_response(raw, [current], supporting_panels=[previous, current],
                                                 previous_segments=prior)
            self.assertTrue(any("repeats previously completed narration" in error for error in errors))
            current["description"] = description

    def test_unused_name_declaration_retries_without_replaying_names_or_weakening_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            segment = {"primary_panel_id": "panel_test_1", "supporting_panel_ids": ["panel_test_1"],
                       "text": "She turns away and looks down.", "used_instance_ids": ["instance_1"],
                       "name_uses": [{"name": "Akira", "instance_id": "instance_1"}]}
            invalid = json.dumps({"segments": [segment]})
            _, errors = validate_story_response(invalid, prepared["panels"])
            self.assertTrue(any("declares a name that is not in its text" in error for error in errors))
            corrected = json.dumps({"segments": [dict(segment, name_uses=[])]})
            with patch("story_writer.writer.generate_response", side_effect=[
                (invalid, {"truncated": False}), (corrected, {"truncated": False}),
            ]) as calls:
                result = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(calls.call_count, 2)
            self.assertNotIn("Akira", calls.call_args_list[1].args[2])
            schema = calls.call_args_list[1].kwargs["response_schema"]
            self.assertNotIn("Akira", json.dumps(schema))
            self.assertEqual(schema["properties"]["segments"]["items"]["anyOf"][0]["properties"]["name_uses"]["maxItems"], 0)
            self.assertEqual(result["segments"][0]["text"], segment["text"])
            self.assertEqual(result["segments"][0]["name_uses"], [])
            progress = json.loads((output / "Example Manga_story_progress.json").read_text(encoding="utf-8"))
            attempt = progress["chunks"][0]["attempts"][0]
            self.assertEqual(Path(attempt["raw_response"]).read_text(encoding="utf-8"), invalid)

    def test_name_suppression_survives_retry_cycles_and_keeps_dialogue_repair_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            prepared["panels"][0]["accepted_text_facts"] = [
                {"kind": "dialogue", "text": "Pardon?", "speaker_instance_id": None}]
            segment = {"primary_panel_id": "panel_test_1", "supporting_panel_ids": ["panel_test_1"],
                       "text": "She turns away. Someone asks, 'Pardon?'", "used_instance_ids": ["instance_1"],
                       "name_uses": [{"name": "Akira", "instance_id": "instance_1"}]}
            phantom = json.dumps({"segments": [segment]})
            omitted = json.dumps({"segments": [dict(segment, text="She turns away and looks down.", name_uses=[])]})
            corrected = json.dumps({"segments": [dict(segment, name_uses=[])]})
            with patch("story_writer.writer.generate_response", side_effect=[
                (phantom, {"truncated": False}), (omitted, {"truncated": False}),
                (corrected, {"truncated": False}),
            ]) as calls, patch("story_writer.writer.time.sleep"):
                result = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(calls.call_count, 3)
            for call in calls.call_args_list[1:]:
                self.assertNotIn("Akira", call.args[2])
                self.assertNotIn("Akira", json.dumps(call.kwargs["response_schema"]))
            self.assertIn("omits brief anonymous dialogue", calls.call_args_list[2].args[2])
            self.assertIn("Pardon?", calls.call_args_list[2].args[2])
            self.assertNotIn("Previous failed response", calls.call_args_list[2].args[2])
            self.assertEqual(result["segments"][0]["text"], segment["text"])

    def test_brief_anonymous_interjection_cannot_be_omitted(self):
        panel = {"panel_id": "p", "characters": [], "accepted_text_facts": [{
            "kind": "dialogue", "text": "HEY!!", "speaker_instance_id": None,
        }]}
        for text, invalid in [("A school stands under a cloudy sky.", True),
                              ("Someone shouts, 'Hey!' outside the school.", False)]:
            raw = json.dumps({"segments": [{
                "primary_panel_id": "p", "supporting_panel_ids": ["p"],
                "text": text, "used_instance_ids": [], "name_uses": [],
            }]})
            _, errors = validate_story_response(raw, [panel])
            self.assertEqual(bool(errors), invalid)

    def test_unknown_dialogue_can_be_a_standalone_quote_without_someone_or_a_guessed_speaker(self):
        panel = {"panel_id": "p", "characters": [], "accepted_text_facts": [
            {"kind": "dialogue", "text": "Pardon?", "speaker_instance_id": None}]}
        cases = [('"Pardon?"', False), ('“Pardon?”', False),
                 ('She turns away. "Pardon?"', False), ('"Pardon?" She turns away.', False),
                 ('He asks, "Pardon?"', True), ('"Pardon?", he asks.', True),
                 ('"Pardon?" he asks.', True), ('"I will attack."', True),
                 ('He turns away.', True)]
        for text, invalid in cases:
            raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
                "text": text, "used_instance_ids": [], "name_uses": []}]})
            with self.subTest(text=text):
                _, errors = validate_story_response(raw, [panel])
                self.assertEqual(bool(errors), invalid, errors)

    def test_unknown_speech_retry_does_not_replay_the_guessed_speaker(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            panel = prepared["panels"][0]
            panel["accepted_text_facts"] = [{"kind": "dialogue", "text": "...I've been waiting for you",
                                             "speaker_instance_id": None}]
            segment = {"primary_panel_id": panel["panel_id"], "supporting_panel_ids": [panel["panel_id"]],
                       "text": 'The character says, "I\'ve been waiting for you."',
                       "used_instance_ids": [], "name_uses": []}
            invalid = json.dumps({"segments": [segment]})
            corrected_text = '“...I\'ve been waiting for you.” The student runs through the corridor.'
            corrected = json.dumps({"segments": [dict(segment, text=corrected_text, used_instance_ids=["instance_1"])]})
            with patch("story_writer.writer.generate_response", side_effect=[
                (invalid, {"truncated": False}), (corrected, {"truncated": False}),
            ]) as calls:
                story = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertNotIn("Previous failed response", calls.call_args_list[1].args[2])
            self.assertNotIn(segment["text"], calls.call_args_list[1].args[2])
            self.assertIn("UNASSIGNED QUOTATIONS", calls.call_args_list[1].args[2])
            self.assertIn("...I've been waiting for you", calls.call_args_list[1].args[2])
            properties = calls.call_args_list[1].kwargs["response_schema"]["properties"]["segments"]["items"]["anyOf"][0]["properties"]
            self.assertNotIn("enum", properties["text"])  # Accepted actions must remain narratable.
            self.assertRegex(corrected_text, properties["text"]["pattern"])
            self.assertTrue(properties["text"]["pattern"].startswith("^“"))
            self.assertEqual(json.loads(json.dumps(corrected_text, ensure_ascii=False)), corrected_text)
            self.assertNotRegex(segment["text"], properties["text"]["pattern"])
            self.assertNotRegex(corrected_text + '\",\"used_instance_ids\":[\"wrong\"]', properties["text"]["pattern"])
            self.assertNotRegex(corrected_text + '\\nMore text', properties["text"]["pattern"])
            self.assertEqual(story["segments"][0]["text"], corrected_text)

    def test_consecutive_unknown_quotations_are_checked_without_consuming_each_others_boundaries(self):
        panel = {"panel_id": "p", "characters": [], "accepted_text_facts": [
            {"kind": "dialogue", "text": "PYRO BURGUNDY!", "speaker_instance_id": None},
            {"kind": "dialogue", "text": "LET THE BRILLIANCE OF YOUR LIFE BURN BRIGHT!", "speaker_instance_id": None}]}
        quotes = '"PYRO BURGUNDY!" "LET THE BRILLIANCE OF YOUR LIFE BURN BRIGHT!"'
        cases = [(quotes, False), ("He points. " + quotes, False),
                 ('“PYRO BURGUNDY!” “LET THE BRILLIANCE OF YOUR LIFE BURN BRIGHT!”', False),
                 ("He says, " + quotes, True), ('"PYRO BURGUNDY!"', True),
                 ('"LET THE BRILLIANCE OF YOUR LIFE BURN BRIGHT!"', True)]
        for text, invalid in cases:
            raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
                "text": text, "used_instance_ids": [], "name_uses": []}]})
            with self.subTest(text=text):
                _, errors = validate_story_response(raw, [panel])
                self.assertEqual(bool(errors), invalid, errors)

    def test_speech_only_unknown_retry_schema_forbids_invented_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            panel = prepared["panels"][0]
            panel.update(accepted_actions=[], accepted_transformations=[], accepted_text_facts=[
                {"kind": "dialogue", "text": "...I've been waiting for you", "speaker_instance_id": None}])
            base = {"primary_panel_id": panel["panel_id"], "supporting_panel_ids": [panel["panel_id"]],
                    "used_instance_ids": [], "name_uses": []}
            invalid = json.dumps({"segments": [dict(base, text='The character says, "I\'ve been waiting for you."')]})
            corrected = json.dumps({"segments": [dict(base, text='"...I\'ve been waiting for you."')]})
            with patch("story_writer.writer.generate_response", side_effect=[
                (invalid, {"truncated": False}), (corrected, {"truncated": False}),
            ]) as calls:
                story = generate_story(prepared, FakeProcessor(), object(), output)
            properties = calls.call_args_list[1].kwargs["response_schema"]["properties"]["segments"]["items"]["anyOf"][0]["properties"]
            self.assertEqual(properties["text"]["enum"], ['"...I\'ve been waiting for you."'])
            self.assertEqual(properties["used_instance_ids"]["maxItems"], 0)
            self.assertEqual(properties["name_uses"]["maxItems"], 0)
            self.assertEqual(story["segments"][0]["text"], '"...I\'ve been waiting for you."')

    def test_short_reference_uses_evidence_not_outfits_alias_gender_or_group_members(self):
        panel = {"chapter": "chapter", "page": "page", "panel_number": 1, "panel_id": "p",
                 "description": "A girl screams while a boy turns away.",
                 "characters": [
                     {"instance_id": "girl", "allowed_name": None,
                      "allowed_reference": "the blonde girl with a long skirt and boots",
                      "visual_description": "blonde girl with a long skirt and boots"},
                     {"instance_id": "boy", "allowed_name": None,
                      "allowed_reference": "the dark-haired boy in a long coat",
                      "visual_description": "dark-haired boy in a long coat"}],
                 "accepted_actions": [{"actor_instance_id": "girl", "actor_reference": "the blonde girl with a long skirt",
                                       "text": "screams"}], "accepted_transformations": [],
                 "accepted_text_facts": [{"kind": "dialogue", "text": "Run!", "speaker_instance_id": "girl"}]}
        before = json.dumps(panel)
        record = _panel_prompt_record(panel)
        self.assertEqual([c["allowed_reference"] for c in record["characters"]], ["she", "he"])
        self.assertEqual(record["characters"][0]["visual_description"], "the blonde girl")
        self.assertEqual(record["accepted_text_facts"][0]["speaker_reference"], "she")
        self.assertEqual(record["accepted_actions"][0]["actor_reference"], "she")
        self.assertNotIn("skirt", json.dumps(record))
        self.assertEqual(json.dumps(panel), before)
        panel["characters"] = [{"instance_id": "unknown", "allowed_name": "Akira",
                                "visual_description": "a demon king with horns"}]
        panel["description"] = "A horned character looks forward."
        panel["accepted_text_facts"] = [{"kind": "dialogue", "text": "Run!", "speaker_instance_id": "unknown"}]
        record = _panel_prompt_record(panel)
        self.assertEqual(record["accepted_text_facts"][0]["speaker_reference"], "the horned character")

    def test_anonymous_declaration_is_valid_but_must_not_replace_a_known_speaker(self):
        panel = {"panel_id": "p", "characters": [], "description": "A hand holding an ornate dagger",
                 "accepted_text_facts": [{"kind": "dialogue", "text": "Then I'll take her",
                                          "speaker_instance_id": None}]}
        for text in ["A hand grips an ornate dagger as someone declares, 'Then I'll take her.'",
                     "A hand holds an ornate dagger while someone declares, 'Then I'll take her.'",
                     "Someone declares, 'Then I'll take her.'"]:
            raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
                "text": text, "used_instance_ids": [], "name_uses": []}]})
            with self.subTest(text=text):
                _, errors = validate_story_response(raw, [panel])
                self.assertFalse(errors, errors)
        panel["accepted_text_facts"][0]["speaker_instance_id"] = "i"
        _, errors = validate_story_response(raw, [panel])
        self.assertTrue(any("anonymizes known speech" in error for error in errors), errors)

    def test_common_reporting_verbs_keep_unknown_speech_anonymous(self):
        panel = {"panel_id": "p", "characters": [], "accepted_text_facts": [
            {"kind": "dialogue", "text": "I swear to protect Celestine for the rest of my life",
             "speaker_instance_id": None}]}
        verbs = ("swears", "vows", "promises", "announces", "explains", "insists", "admits", "agrees",
                 "answers", "demands", "calls", "cries", "screams", "tells", "retorts", "protests",
                 "reassures", "repeats", "recounts", "continues", "pleads", "begs", "replies", "declares")
        for verb in verbs:
            raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
                "text": f"Someone {verb} to protect Celestine for the rest of his life.",
                "used_instance_ids": [], "name_uses": []}]})
            with self.subTest(verb=verb):
                panel["accepted_text_facts"][0]["speaker_instance_id"] = None
                _, errors = validate_story_response(raw, [panel])
                self.assertFalse(errors, errors)
                panel["accepted_text_facts"][0]["speaker_instance_id"] = "i"
                _, errors = validate_story_response(raw, [panel])
                self.assertTrue(any("anonymizes known speech" in error for error in errors), errors)

    def test_anonymous_speech_allows_adverbs_but_not_a_different_named_subject(self):
        panel = {"panel_id": "p", "characters": [], "accepted_text_facts": [
            {"kind": "dialogue", "text": "Has anyone been bullying you?", "speaker_instance_id": None}]}
        for subject in ("Someone nearby", "Someone quietly", "Someone off-screen", "Someone off screen",
                        "Somebody nearby quietly", "A voice elsewhere", "Someone then", "Someone still"):
            text = f"{subject} warns, 'Has anyone been bullying you?'"
            raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
                "text": text, "used_instance_ids": [], "name_uses": []}]})
            with self.subTest(subject=subject):
                panel["accepted_text_facts"][0]["speaker_instance_id"] = None
                _, errors = validate_story_response(raw, [panel])
                self.assertFalse(errors, errors)
                panel["accepted_text_facts"][0]["speaker_instance_id"] = "i"
                _, errors = validate_story_response(raw, [panel])
                self.assertTrue(any("anonymizes known speech" in error for error in errors), errors)
        panel["accepted_text_facts"][0]["speaker_instance_id"] = None
        raw = json.dumps({"segments": [{"primary_panel_id": "p", "supporting_panel_ids": ["p"],
            "text": "Someone stands nearby while she warns, 'Has anyone been bullying you?'",
            "used_instance_ids": [], "name_uses": []}]})
        _, errors = validate_story_response(raw, [panel])
        self.assertTrue(any("unknown dialogue explicitly anonymous" in error for error in errors), errors)

    def test_source_commentary_is_repaired_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            def response(text):
                return json.dumps({"segments": [{
                    "primary_panel_id": "panel_test_1", "supporting_panel_ids": ["panel_test_1"],
                    "text": text, "used_instance_ids": [], "name_uses": [],
                }]})
            raw = response("An unattributed warning states that flying is banned.")
            repaired = response("Someone warns that flying is banned.")
            with patch("story_writer.writer.generate_response", side_effect=[
                (raw, {"truncated": False}), (repaired, {"truncated": False}),
            ]) as call:
                story = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(call.call_count, 2)
            self.assertIn("'unattributed' reports source presentation", call.call_args_list[1].args[2])
            self.assertEqual(story["segments"][0]["text"], "Someone warns that flying is banned.")

    def test_sound_effect_suffix_is_removed_only_from_retry_input_without_dropping_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            panel = prepared["panels"][0]
            panel.update(description="A banner at the bottom with text and a sound effect.",
                         characters=[], accepted_text_facts=[], accepted_actions=[])
            def response(text):
                return json.dumps({"segments": [{"primary_panel_id": panel["panel_id"],
                    "supporting_panel_ids": [panel["panel_id"]], "text": text,
                    "used_instance_ids": [], "name_uses": []}]})
            invalid = response("A banner displays text and a sound effect at the bottom.")
            corrected = response("A banner with text sits at the bottom.")
            with patch("story_writer.writer.generate_response", side_effect=[
                (invalid, {"truncated": False}), (corrected, {"truncated": False}),
            ]) as calls:
                story = generate_story(prepared, FakeProcessor(), object(), output)
            context = json.loads(calls.call_args_list[1].args[2].split("\nGrounded input:\n")[1].split("\nGenerate exactly")[0])
            self.assertEqual(context["next_panels"][0]["description"], "A banner at the bottom with text.")
            self.assertNotIn("Previous failed response", calls.call_args_list[1].args[2])
            self.assertEqual(panel["description"], "A banner at the bottom with text and a sound effect.")
            self.assertEqual(len(story["segments"]), 1)
            self.assertEqual(story["segments"][0]["text"], "A banner with text sits at the bottom.")

    def test_unresolved_optional_review_does_not_block_story(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            path = output / manga.name / "character_classification.json"
            classification = json.loads(path.read_text(encoding="utf-8"))
            classification["reviewed_evidence"] = {
                "unresolved_critical_items": [{
                    "status": "unresolved", "reason": "speaker is unknown",
                }],
            }
            _write_json(path, classification)
            prepared = prepare_story(manga, chapters, output)
            self.assertTrue(any("speaker is unknown" in row for row in prepared["warnings"]))

    def test_unknown_identity_without_reviews_uses_visual_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            path = output / manga.name / "character_classification.json"
            classification = json.loads(path.read_text(encoding="utf-8"))
            character = classification["characters"][0]
            character.update(
                entity_id=None, form_id=None, identity_status="unknown",
                name=None, naming_allowed=False, display_name_allowed=False,
            )
            classification["story_events"] = []
            classification["reviewed_evidence"] = {}
            _write_json(path, classification)
            prepared = prepare_story(manga, chapters, output)
            fact = prepared["panels"][0]["characters"][0]
            self.assertIsNone(fact["allowed_name"])
            self.assertEqual(fact["allowed_reference"], "the horned student")
            self.assertEqual(prepared["panels"][0]["accepted_actions"], [])

    def test_unknown_identity_uses_generated_alias_without_a_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            path = output / manga.name / "character_classification.json"
            classification = json.loads(path.read_text(encoding="utf-8"))
            classification["characters"][0].update(
                entity_id=None, form_id=None, identity_status="unknown",
                name="Canonical name", naming_allowed=True,
                display_name="Yamazaki", display_name_allowed=True,
            )
            _write_json(path, classification)
            panel = prepare_story(manga, chapters, output)["panels"][0]
            self.assertEqual(panel["characters"][0]["allowed_name"], "Yamazaki")
            self.assertEqual(panel["accepted_actions"][0]["actor_reference"], "Yamazaki")

    def test_ambiguous_panel_keeps_model_text_without_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            description_path = output / f"{manga.name}_descriptions.json"
            data = json.loads(description_path.read_text(encoding="utf-8"))
            page = data["descriptions"]["chapter_001"]["00001.jpg"]
            page["extraction_status"] = "complete_with_uncertainty"
            page["panels"][0].update(description="The.", status="ambiguous")
            data["page_filter"]["chapters"]["chapter_001"]["00001.jpg"][
                "extraction_status"
            ] = "complete_with_uncertainty"
            _write_json(description_path, data)
            classification_path = output / manga.name / "character_classification.json"
            classification = json.loads(classification_path.read_text(encoding="utf-8"))
            classification["description_fingerprint"] = stable_id(
                "descriptions_", data["descriptions"],
            )
            _write_json(classification_path, classification)

            prepared = prepare_story(manga, chapters, output)
            panel = prepared["panels"][0]
            self.assertEqual(panel["evidence_density"], "uncertain")
            self.assertEqual(
                panel["description"], "The.",
            )

    def test_automatic_grounded_action_is_available_without_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            path = output / manga.name / "character_classification.json"
            classification = json.loads(path.read_text(encoding="utf-8"))
            event = classification["story_events"][0]
            event.update(
                status="accepted_automatic",
                acceptance_policy="grounded_visible_action_v1",
            )
            event.pop("confidence_percent", None)
            for policy in ("grounded_visible_action_v1", "grounded_visible_action_format_v2", "grounded_visible_action_format_v3"):
                with self.subTest(policy=policy):
                    event["acceptance_policy"] = policy
                    _write_json(path, classification)
                    prepared = prepare_story(manga, chapters, output)
                    action = prepared["panels"][0]["accepted_actions"][0]
                    self.assertEqual(action["acceptance_policy"], policy)
                    self.assertNotIn("confidence_percent", action)

    def test_prepare_story_rejects_stale_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            (chapters[0] / "00001.jpg").write_bytes(b"changed page")
            with self.assertRaisesRegex(ValueError, "Source page changed"):
                prepare_story(manga, chapters, output)

    def test_prepare_story_does_not_silently_skip_failed_page_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            path = output / "Example Manga_descriptions.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            decision = data["page_filter"]["chapters"]["chapter_001"]["00001.jpg"]
            decision.update(included_in_story=False, extraction_status="invalid_output")
            data["descriptions"]["chapter_001"].pop("00001.jpg")
            _write_json(path, data)
            classification_path = output / "Example Manga" / "character_classification.json"
            classification = json.loads(classification_path.read_text(encoding="utf-8"))
            classification["description_fingerprint"] = stable_id(
                "descriptions_", data["descriptions"]
            )
            _write_json(classification_path, classification)
            with self.assertRaisesRegex(ValueError, "Page analysis is incomplete"):
                prepare_story(manga, chapters, output)

    def test_response_validation_enforces_panel_instance_and_name_grounding(self):
        panel = {
            "panel_id": "panel_1",
            "characters": [{"instance_id": "instance_1", "allowed_name": "Akira"}],
        }
        valid = json.dumps({"segments": [{
            "primary_panel_id": "panel_1",
            "supporting_panel_ids": ["panel_1"],
            "text": "Akira races through the crowded school corridor as distant footsteps echo behind the determined horned student.",
            "used_instance_ids": ["instance_1"],
            "name_uses": [{"name": "Akira", "instance_id": "instance_1"}],
        }]})
        segments, errors = validate_story_response(valid, [panel])
        self.assertFalse(errors)
        self.assertEqual(segments[0]["primary_panel_id"], "panel_1")

        invalid = json.dumps({"segments": [{
            "primary_panel_id": "panel_1",
            "supporting_panel_ids": ["panel_1"],
            "text": "Akira races through the crowded school corridor as distant footsteps echo behind the determined horned student.",
            "used_instance_ids": ["instance_other"],
            "name_uses": [{"name": "Akira", "instance_id": "instance_other"}],
        }]})
        _, errors = validate_story_response(invalid, [panel])
        self.assertTrue(any("used_instance_ids" in error for error in errors))
        self.assertTrue(any("unapproved name" in error for error in errors))

    def test_response_validation_rejects_wrong_panel_order(self):
        panels = [
            {"panel_id": "panel_1", "characters": []},
            {"panel_id": "panel_2", "characters": []},
        ]
        response = json.dumps({"segments": [{
            "primary_panel_id": panel_id,
            "supporting_panel_ids": [panel_id],
            "text": "The quiet corridor stretches onward while shadows gather and the scene steadily moves toward an uncertain confrontation.",
            "used_instance_ids": [],
            "name_uses": [],
        } for panel_id in ("panel_2", "panel_1")]})
        _, errors = validate_story_response(response, panels)
        self.assertIn("Segments must cover every requested panel exactly once and in order.", errors)

    def test_response_rejects_future_support_and_allows_sparse_truth(self):
        panels = [
            {"panel_id": "panel_1", "order": 1, "characters": []},
            {"panel_id": "panel_2", "order": 2, "characters": [],
             "evidence_density": "sparse"},
        ]
        future = json.dumps({"segments": [{
            "primary_panel_id": "panel_1",
            "supporting_panel_ids": ["panel_1", "panel_2"],
            "text": "The empty hall remains still while a distant shadow approaches the doorway.",
            "used_instance_ids": [], "name_uses": [],
        }]})
        _, errors = validate_story_response(
            future, [panels[0]], supporting_panels=panels,
        )
        self.assertTrue(any("future panel" in error for error in errors))

        sparse = json.dumps({"segments": [{
            "primary_panel_id": "panel_2",
            "supporting_panel_ids": ["panel_2"],
            "text": "A figure quietly watches.",
            "used_instance_ids": [], "name_uses": [],
        }]})
        _, errors = validate_story_response(sparse, [panels[1]])
        self.assertFalse(errors)

    def test_pronouns_in_dialogue_do_not_trigger_a_perspective_rejection(self):
        panel = {"panel_id": "panel_1", "characters": []}
        segment = {
            "primary_panel_id": "panel_1", "supporting_panel_ids": ["panel_1"],
            "text": "DO YOU THINK THE RULE VIOLATION WILL SLIDE?",
            "used_instance_ids": [], "name_uses": [],
        }
        for text in ('The student asks, "Do you think the rule violation will slide?"',
                     'The girl says, "I will return."',
                     "She utters the words you in a moment of hesitation.",
                     "The student refuses to overlook the rule violation."):
            with self.subTest(text=text):
                segment["text"] = text
                normalized, errors = validate_story_response(json.dumps({"segments": [segment]}), [panel])
                self.assertFalse(errors)
                self.assertEqual(normalized[0]["text"], text)

    def test_prompt_contains_no_future_outline_or_character_guide(self):
        base = {
            "chapter": "chapter_001", "page": "page.jpg", "panel_number": 1,
            "characters": [], "accepted_actions": [], "accepted_transformations": [],
            "accepted_text_facts": [], "description": "A quiet room is visible.",
        }
        current = dict(base, panel_id="current", order=1)
        future = dict(base, panel_id="future", order=2,
                      description="Future secret name appears here.")
        prompt = build_story_prompt(
            [current], completed_panels=[], previous_narration="",
            tokenizer=FakeTokenizer(),
        )
        self.assertNotIn("complete_story_outline", prompt)
        self.assertNotIn("fixed_character_guide", prompt)
        self.assertNotIn(future["description"], prompt)

    def test_later_chunk_can_cite_an_already_covered_support_panel(self):
        previous = {
            "panel_id": "panel_1",
            "characters": [{"instance_id": "instance_1", "allowed_name": "Akira"}],
        }
        current = {"panel_id": "panel_2", "characters": []}
        response = json.dumps({"segments": [{
            "primary_panel_id": "panel_2",
            "supporting_panel_ids": ["panel_1", "panel_2"],
            "text": "Akira's earlier form gives way to a changed silhouette as the visible transformation reaches its final stage.",
            "used_instance_ids": ["instance_1"],
            "name_uses": [{"name": "Akira", "instance_id": "instance_1"}],
        }]})
        _, errors = validate_story_response(
            response,
            [current],
            supporting_panels=[previous, current],
        )
        self.assertFalse(errors)

    def test_generation_publishes_and_resumes_without_another_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            raw = json.dumps({"segments": [{
                "primary_panel_id": "panel_test_1",
                "supporting_panel_ids": ["panel_test_1"],
                "text": "Akira races through the crowded school corridor while the horned student searches ahead for a safe escape route.",
                "used_instance_ids": ["instance_1"],
                "name_uses": [{"name": "Akira", "instance_id": "instance_1"}],
            }]})
            generation = {"input_tokens": 100, "generated_tokens": 80,
                          "token_limit": 768, "truncated": False}
            with patch("story_writer.writer.generate_response", return_value=(raw, generation)) as call, \
                    patch("builtins.print") as progress_print:
                story = generate_story(prepared, FakeProcessor(), object(), output)
                self.assertEqual(call.call_count, 1)
            messages = [str(call.args[0]) for call in progress_print.call_args_list]
            self.assertTrue(any("Story progress: 0/1 panels (0%)" in row for row in messages))
            self.assertTrue(any("Completed 1/1 panels (100%)" in row for row in messages))
            self.assertEqual(story["segments"][0]["primary_panel_id"], "panel_test_1")
            self.assertTrue((output / "Example Manga_story.txt").is_file())
            self.assertTrue((output / "Example Manga_story.json").is_file())
            self.assertTrue((output / "Example Manga_panel_timeline.json").is_file())
            timeline = json.loads((output / "Example Manga_panel_timeline.json").read_text())
            self.assertEqual(timeline["display_groups"][0]["panel_ids"], ["panel_test_1"])

            with patch("story_writer.writer.generate_response", side_effect=AssertionError("must resume")):
                resumed = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(resumed["narration"], story["narration"])

            (output / "Example Manga_panel_timeline.json").unlink()
            with patch("story_writer.writer.generate_response", side_effect=AssertionError("must restore")):
                generate_story(prepared, FakeProcessor(), object(), output)
            self.assertTrue((output / "Example Manga_panel_timeline.json").is_file())

    def test_reported_dialogue_does_not_require_a_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            raw = json.dumps({"segments": [{
                "primary_panel_id": "panel_test_1", "supporting_panel_ids": ["panel_test_1"],
                "text": "She utters the words you in hesitation.",
                "used_instance_ids": [], "name_uses": [],
            }]})
            with patch("story_writer.writer.generate_response", return_value=(raw, {"truncated": False})) as call:
                story = generate_story(prepared, FakeProcessor(), object(), output)
            call.assert_called_once()
            self.assertEqual(story["segments"][0]["text"], "She utters the words you in hesitation.")

    def test_invalid_single_panel_waits_and_retries_without_publishing_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            generation = {"input_tokens": 100, "generated_tokens": 2,
                          "token_limit": 768, "truncated": False}
            with patch(
                "story_writer.writer.generate_response",
                return_value=("not json", generation),
            ) as call, patch("story_writer.writer.time.sleep", side_effect=KeyboardInterrupt) as wait:
                with self.assertRaises(KeyboardInterrupt):
                    generate_story(prepared, FakeProcessor(), object(), output)
            wait.assert_called_once_with(5)
            self.assertEqual(call.call_count, 2)
            progress = json.loads(
                (output / "Example Manga_story_progress.json").read_text(encoding="utf-8")
            )
            self.assertFalse(progress["complete"])
            self.assertEqual(progress["completed_panel_ids"], [])
            self.assertEqual(len(progress["chunks"][-1]["attempts"]), 2)
            self.assertEqual(progress["chunks"][-1]["status"], "failed")
            for attempt in progress["chunks"][-1]["attempts"]:
                self.assertTrue(Path(attempt["raw_response"]).is_file())
            self.assertFalse((output / "Example Manga_story.json").exists())
            self.assertIn("Previous failed response", call.call_args_list[1].args[2])
            self.assertIn("not json", call.call_args_list[1].args[2])

    def test_single_panel_eventually_recovers_without_manual_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            valid = json.dumps({"segments": [{
                "primary_panel_id": "panel_test_1", "supporting_panel_ids": ["panel_test_1"],
                "text": "The door opens slowly.", "used_instance_ids": [], "name_uses": [],
            }]})
            with patch("story_writer.writer.generate_response", side_effect=[
                ("bad json", {"truncated": False}), ("still bad", {"truncated": False}),
                (valid, {"truncated": False}),
            ]) as call, patch("story_writer.writer.time.sleep") as wait:
                story = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(call.call_count, 3)
            wait.assert_called_once_with(5)
            self.assertIn("still bad", call.call_args_list[2].args[2])
            self.assertEqual(len(story["segments"]), 1)
            progress = json.loads((output / "Example Manga_story_progress.json").read_text())
            self.assertTrue(progress["complete"])
            self.assertEqual([c["status"] for c in progress["chunks"]], ["failed", "complete"])

    def test_persistent_invalid_panel_stops_with_checkpoint_and_raw_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            with patch("story_writer.writer.generate_response", return_value=("not json", {"truncated": False})) as call, \
                 patch("story_writer.writer.time.sleep"), patch("config.STORY_MAX_PANEL_FAILURES", 2):
                with self.assertRaisesRegex(RuntimeError, "failed 2 retry cycles"):
                    generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(call.call_count, 4)
            progress = json.loads((output / "Example Manga_story_progress.json").read_text(encoding="utf-8"))
            self.assertFalse(progress["complete"])
            self.assertEqual(len(progress["chunks"]), 2)
            self.assertTrue(all(Path(a["raw_response"]).exists() for c in progress["chunks"] for a in c["attempts"]))
            self.assertFalse((output / "Example Manga_story.json").exists())

    def test_longer_segments_are_accepted_without_changing_prompt_target(self):
        panel = {"panel_id": "p1", "characters": []}
        for count in (57, 95, 150, 151):
            with self.subTest(count=count):
                raw = json.dumps({"segments": [{
                    "primary_panel_id": "p1", "supporting_panel_ids": ["p1"],
                    "text": " ".join(["word"] * count), "used_instance_ids": [], "name_uses": [],
                }]})
                _, errors = validate_story_response(raw, [panel])
                self.assertEqual(bool(errors), count > 150)
        from story_writer.writer import SYSTEM_PROMPT
        self.assertIn("Keep each segment under 50 words", SYSTEM_PROMPT)

    def test_failed_multi_panel_chunk_splits_and_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            second = copy.deepcopy(prepared["panels"][0])
            second.update(
                order=2, panel_id="panel_test_2", panel_number=2,
                description="A closed doorway stands beyond the quiet corridor.",
                characters=[], accepted_actions=[], accepted_transformations=[],
                evidence_density="sparse",
            )
            prepared["panels"].append(second)
            generation = {"input_tokens": 100, "generated_tokens": 2,
                          "token_limit": 768, "truncated": False}
            def generate(processor, model, prompt, *, max_new_tokens, response_schema):
                variants = [variant for variant in response_schema["properties"]["segments"]["items"]["anyOf"]
                            if "narration_policy" not in variant["properties"]]
                if len(variants) > 1:
                    return "not json", generation
                panel_id = variants[0]["properties"]["primary_panel_id"]["enum"][0]
                return json.dumps({"segments": [{
                    "primary_panel_id": panel_id, "supporting_panel_ids": [panel_id],
                    "text": "A door opens.", "used_instance_ids": [], "name_uses": [],
                }]}), generation

            with patch(
                "story_writer.writer.generate_response",
                side_effect=generate,
            ) as call, patch("config.STORY_CHUNK_MAX_PANELS", 2):
                story = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(call.call_count, 4)
            progress = json.loads(
                (output / "Example Manga_story_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(progress["chunks"]), 3)
            self.assertEqual(progress["chunks"][0]["status"], "failed")
            self.assertEqual([s["primary_panel_id"] for s in story["segments"]], ["panel_test_1", "panel_test_2"])

    def test_failure_saves_completed_prefix_and_resume_does_not_regenerate_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            second = copy.deepcopy(prepared["panels"][0])
            second.update(order=2, panel_id="panel_test_2", characters=[])
            prepared["panels"].append(second)

            def response(panel_id):
                return json.dumps({"segments": [{
                    "primary_panel_id": panel_id, "supporting_panel_ids": [panel_id],
                    "text": "A door opens.", "used_instance_ids": [], "name_uses": [],
                }]})

            metrics = {"truncated": False}
            with patch("config.STORY_CHUNK_MAX_PANELS", 1), \
                    patch("story_writer.writer.generate_response", side_effect=[
                        (response("panel_test_1"), metrics), ("not json", metrics), ("not json", metrics),
                    ]), patch("story_writer.writer.time.sleep", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    generate_story(prepared, FakeProcessor(), object(), output)
            progress = json.loads((output / "Example Manga_story_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["completed_panel_ids"], ["panel_test_1"])
            self.assertEqual([row["status"] for row in progress["chunks"]], ["complete", "failed"])
            self.assertFalse((output / "Example Manga_story.json").exists())

            with patch("story_writer.writer.generate_response", return_value=(response("panel_test_2"), metrics)) as call:
                story = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(call.call_count, 1)
            self.assertEqual([row["primary_panel_id"] for row in story["segments"]], ["panel_test_1", "panel_test_2"])

    def test_publication_keeps_exact_cross_chunk_duplicate(self):
        segments = [
            {"text": "The gate opens. A shadow crosses the floor.", "generation_chunk": 1},
            {"text": "A shadow crosses the floor. The student turns around.", "generation_chunk": 2},
        ]
        edited = _boundary_edit(segments)
        self.assertEqual(edited[0]["text"], segments[0]["text"])
        self.assertEqual(edited[1]["text"], segments[1]["text"])

    def test_small_writer_chunks_are_bounded_even_when_full_story_fits(self):
        base = {
            "chapter": "chapter_001", "page": "page.jpg", "panel_number": 1,
            "description": "A quiet room is visible.", "characters": [],
            "accepted_actions": [], "accepted_transformations": [],
        }
        panels = [dict(base, panel_id=f"panel_{index}") for index in range(9)]
        with patch("config.STORY_CHUNK_MAX_PANELS", 4):
            chunks = _initial_chunks({"panels": panels}, FakeTokenizer(), [])
        self.assertEqual([len(chunk) for chunk in chunks], [4, 4, 1])
        self.assertEqual([panel["panel_id"] for chunk in chunks for panel in chunk],
                         [panel["panel_id"] for panel in panels])

    def test_prompt_does_not_keep_accumulating_old_instance_ids(self):
        base = {
            "chapter": "chapter_001", "page": "page.jpg", "panel_number": 1,
            "description": "A quiet room is visible.", "characters": [],
            "accepted_actions": [], "accepted_transformations": [],
        }
        old = [dict(base, panel_id=f"old_{index}", characters=[{
            "instance_id": f"old_instance_{index}", "allowed_reference": "a student",
            "allowed_name": None,
        }]) for index in range(10)]
        prompt = build_story_prompt([dict(base, panel_id="current")], completed_panels=old,
                                    previous_narration="An earlier capsule machine scene must not leak here.", tokenizer=FakeTokenizer())
        self.assertNotIn("An earlier capsule machine scene", prompt)
        self.assertNotIn("old_instance_0", prompt)
        self.assertNotIn("old_instance_9", prompt)
        self.assertIn('"count":10', prompt)
        self.assertIn('ONLY for these primary_panel_ids in this order: ["current"]', prompt)

    def test_grammar_accepts_exact_chunk_and_blocks_runaway_rows_or_old_ids(self):
        from lmformatenforcer import JsonSchemaParser
        panel = {"panel_id": "current", "characters": []}
        schema = _story_response_schema([panel])
        segment = {"primary_panel_id": "current", "supporting_panel_ids": ["current"],
                   "text": "The door opens slowly into the empty corridor.",
                   "used_instance_ids": [], "name_uses": []}

        def accepts(value):
            parser = JsonSchemaParser(schema)
            for character in json.dumps(value, separators=(",", ":")):
                if character not in parser.get_allowed_characters():
                    return False
                parser = parser.add_character(character)
            return parser.can_end()

        self.assertTrue(accepts({"segments": [segment]}))
        self.assertFalse(accepts({"segments": [segment, segment]}))
        self.assertFalse(accepts({"segments": [dict(segment, primary_panel_id="old_panel")]}))
        self.assertFalse(accepts({"segments": [dict(segment, supporting_panel_ids=[])]}))

    def test_token_filter_accepts_merged_sentence_boundaries_without_relaxing_json(self):
        import string
        from types import SimpleNamespace
        import transformers.tokenization_utils as tokenization_utils
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        from description.schema import _complete_json_token_filter
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

        texts = list(string.printable) + ['.","', '."}', '.","bogus":']
        encoded = {text: index for index, text in enumerate(texts)}
        data = TokenEnforcerTokenizerData(
            [(index, text, False) for index, text in enumerate(texts)],
            lambda tokens: "".join(texts[index] for index in tokens), 255, False, 256,
        )
        schemas_and_values = [
            (_story_response_schema([{"panel_id": "current", "characters": []}]),
             {"segments": [{"primary_panel_id": "current", "supporting_panel_ids": ["current"],
                            "text": "The door opens into the silent corridor.",
                            "used_instance_ids": [], "name_uses": []}]}),
            (_story_response_schema([{"panel_id": "current", "characters": [
                {"instance_id": "instance", "allowed_name": "Akira"},
            ]}]),
             {"segments": [{"primary_panel_id": "current", "supporting_panel_ids": ["current"],
                            "text": "Akira opens the door.", "used_instance_ids": ["instance"],
                            "name_uses": [{"name": "Akira", "instance_id": "instance"}]}]}),
            ({"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False}, {"text": "The door opens."}),
            ({"type": "object", "properties": {
                "strings": {"type": "array", "maxItems": 0, "items": {"type": "string"}},
                "objects": {"type": "array", "maxItems": 0, "items": {"type": "object"}},
            }, "required": ["strings", "objects"], "additionalProperties": False},
             {"strings": [], "objects": []}),
            (_story_response_schema([
                {"panel_id": panel_id, "characters": []} for panel_id in ("first", "second")
            ]), {"segments": [
                {"primary_panel_id": panel_id, "supporting_panel_ids": [panel_id],
                 "text": "The horned figure opens the door.", "used_instance_ids": [], "name_uses": []}
                for panel_id in ("first", "second")
            ]}),
        ]
        with patch.object(tokenization_utils, "PreTrainedTokenizerBase", PreTrainedTokenizerBase, create=True), \
                patch.dict(os.environ, {"LMFE_MAX_CONSECUTIVE_WHITESPACES": "0"}):
            for schema, value in schemas_and_values:
                with self.subTest(value=value):
                    filter_fn = _complete_json_token_filter(data, JsonSchemaParser(schema))
                    target = json.dumps(value, separators=(",", ":"))
                    prefix = [254]  # Prompt is ignored by the parser.
                    while target:
                        piece = next((text for text in texts[-3:] if target.startswith(text)), target[0])
                        allowed = filter_fn(0, SimpleNamespace(tolist=lambda: prefix))
                        if target.startswith("]") and any(
                            "".join(texts[index] for index in prefix[1:]).endswith(f'"{key}":[')
                            for key in ("strings", "objects", "name_uses", "used_instance_ids")
                        ):
                            self.assertNotIn(encoded['"'], allowed)
                            self.assertNotIn(encoded['{'], allowed)
                        self.assertIn(encoded[piece], allowed, piece)
                        if target.startswith('.","'):
                            self.assertNotIn(encoded['.","bogus":'], allowed)
                        prefix.append(encoded[piece])
                        target = target[len(piece):]
                    self.assertIn(255, filter_fn(0, SimpleNamespace(tolist=lambda: prefix)))

    def test_grammar_keeps_support_and_instances_with_the_selected_primary_panel(self):
        from lmformatenforcer import JsonSchemaParser
        panels = [{"panel_id": "first", "characters": []},
                  {"panel_id": "second", "characters": [{"instance_id": "second_instance"}]}]
        schema = _story_response_schema(panels)

        def accepts_prefix(supporting, used):
            parser = JsonSchemaParser(schema)
            prefix = '{"segments":[{"primary_panel_id":"first","supporting_panel_ids":'
            prefix += json.dumps(supporting) + ',"text":"The door opens.","used_instance_ids":' + json.dumps(used)
            for character in prefix:
                if character not in parser.get_allowed_characters():
                    return False
                parser = parser.add_character(character)
            return True

        self.assertTrue(accepts_prefix(["first"], []))
        self.assertFalse(accepts_prefix(["second"], []))
        self.assertFalse(accepts_prefix(["first"], ["second_instance"]))

    def test_truncated_multi_panel_chunk_retries_same_panels_with_more_room(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            second = copy.deepcopy(prepared["panels"][0])
            second.update(panel_id="panel_test_2", order=2, characters=[])
            prepared["panels"].append(second)

            attempts = []
            def generate(processor, model, prompt, *, max_new_tokens, response_schema):
                ids = [variant["properties"]["primary_panel_id"]["enum"][0]
                       for variant in response_schema["properties"]["segments"]["items"]["anyOf"]]
                attempts.append(ids)
                if len(attempts) == 1:
                    return "truncated", {"truncated": True, "generated_tokens": max_new_tokens}
                return json.dumps({"segments": [{
                    "primary_panel_id": panel_id, "supporting_panel_ids": [panel_id],
                    "text": "The empty corridor remains silent as the door slowly opens.",
                    "used_instance_ids": [], "name_uses": [],
                } for panel_id in ids]}), {"truncated": False, "generated_tokens": 80}

            with patch("story_writer.writer.generate_response", side_effect=generate) as call, \
                    patch("config.STORY_CHUNK_MAX_PANELS", 2):
                story = generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(attempts, [["panel_test_1", "panel_test_2"]] * 2)
            self.assertEqual(len(story["segments"]), 2)
            progress = json.loads((output / "Example Manga_story_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(len(progress["chunks"][0]["attempts"]), 2)
            self.assertEqual(progress["chunks"][0]["status"], "complete")

    def test_single_panel_truncation_gets_more_room_not_the_identical_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            with patch("story_writer.writer.generate_response", side_effect=[
                ("truncated", {"truncated": True}), ("invalid", {"truncated": False}),
            ]) as call, patch("story_writer.writer.time.sleep", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(call.call_args_list[1].kwargs["max_new_tokens"],
                             call.call_args_list[0].kwargs["max_new_tokens"] * 2)

    def test_exact_short_fallback_resumes_but_forged_fallback_does_not(self):
        panel = {"panel_id": "panel_1", "description": "A closed door.", "characters": []}
        segment = _restricted_fallback(panel)
        _, errors = validate_story_response(json.dumps({"segments": [segment]}), [panel])
        self.assertFalse(errors)
        _, errors = validate_story_response(json.dumps({"segments": [dict(segment, text="An invented explosion.")]}), [panel])
        self.assertTrue(any("fallback evidence" in error for error in errors))

    def test_previous_model_progress_is_archived_without_destroying_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            segment = _restricted_fallback(prepared["panels"][0])
            saved = {
                "input_fingerprint": "previous_2b_model_fingerprint",
                "prompt_version": "grounded_story_writer_v4_verbatim_observations",
                "completed_panel_ids": [segment["primary_panel_id"]],
                "segments": [segment], "complete": False,
            }
            path = output / "Example Manga_story_progress.json"
            _write_json(path, saved)
            resumed = _load_progress(path, prepared, restart=False)
            self.assertEqual(resumed["segments"], [])
            self.assertEqual(resumed["input_fingerprint"], prepared["input_fingerprint"])
            archives = list(output.glob("*.stale_*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(json.loads(archives[0].read_text(encoding="utf-8")), saved)

    def test_changing_gguf_artifact_changes_writer_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            model = Path(temporary) / "test.gguf"
            with patch("config.STORY_MODEL_PATH", model):
                model.write_bytes(b"first model")
                first = prepare_story(manga, chapters, output)
                model.write_bytes(b"second different model")
                second = prepare_story(manga, chapters, output)
            self.assertNotEqual(first["input_fingerprint"], second["input_fingerprint"])

    def test_generate_response_uses_native_backend_without_loading_torch(self):
        processor, model = MagicMock(), MagicMock()
        model.generate.return_value = ("{}", {"elapsed_seconds": 1, "tokens_per_second": 2})
        schema = _story_response_schema([{"panel_id": "current", "characters": []}])
        _, metrics = generate_response(processor, model, "prompt", max_new_tokens=256, response_schema=schema)
        model.generate.assert_called_once_with("prompt", max_new_tokens=256, response_schema=schema)
        self.assertIn("elapsed_seconds", metrics)

    def test_runner_closes_owned_server_when_validation_fails(self):
        from story_writer.writer import run_story_writer
        with tempfile.TemporaryDirectory() as temporary:
            model = MagicMock()
            prepared = {"manga": "Example Manga"}
            with patch("story_writer.writer.prepare_story", return_value=prepared), \
                    patch("story_writer.writer.load_story_model", return_value=(model, model)), \
                    patch("story_writer.writer.generate_story", side_effect=ValueError("failed validation")):
                with self.assertRaisesRegex(ValueError, "failed validation"):
                    run_story_writer(Path(temporary), [], Path(temporary))
            model.close.assert_called_once()

    def test_timeline_uses_narration_word_positions(self):
        segments = [{"primary_panel_id": "panel_1", "text": "One two three four five."}]
        lookup = {"panel_1": {"chapter": "chapter_001", "panel_image": "panel.jpg"}}
        timeline = _timeline(segments, lookup, 150)
        self.assertEqual(timeline[0]["end_word"], 5)
        self.assertEqual(timeline[0]["estimated_end_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
