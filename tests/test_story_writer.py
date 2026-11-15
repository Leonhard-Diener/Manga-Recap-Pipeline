import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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
    (chapter_output / "panel_0001.jpg").write_bytes(b"panel crop")
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
            self.assertEqual(len(panel["accepted_text_facts"]), 6)
            record = _panel_prompt_record(panel)
            self.assertEqual([fact["text"] for fact in record["accepted_text_facts"]],
                             [rows[1][1], rows[4][1]])
            self.assertEqual([fact["text"] for fact in record["unattributed_speech"]],
                             [rows[2][1], rows[3][1]])
            self.assertEqual(record["sound_effects_context_only"][0]["text"], "BAM")
            self.assertNotIn(rows[0][1], json.dumps(record))
            prompt = build_story_prompt([panel], completed_panels=[], previous_narration="",
                                        tokenizer=FakeTokenizer())
            self.assertIn("unattributed_speech has NO known", prompt)
            self.assertIn("not mandatory", prompt)
            self.assertNotIn(rows[0][1], prompt)

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

    def test_invalid_single_panel_stops_without_publishing_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            generation = {"input_tokens": 100, "generated_tokens": 2,
                          "token_limit": 768, "truncated": False}
            with patch(
                "story_writer.writer.generate_response",
                return_value=("not json", generation),
            ) as call:
                with self.assertRaisesRegex(ValueError, "no failure-splitting or automatic fallback"):
                    generate_story(prepared, FakeProcessor(), object(), output)
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

    def test_failed_multi_panel_chunk_never_splits(self):
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
            with patch(
                "story_writer.writer.generate_response",
                return_value=("not json", generation),
            ) as call:
                with self.assertRaisesRegex(ValueError, "no failure-splitting"):
                    generate_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(call.call_count, 2)
            progress = json.loads(
                (output / "Example Manga_story_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(progress["chunks"]), 1)
            self.assertEqual(progress["chunks"][0]["status"], "failed")
            self.assertEqual(progress["segments"], [])

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
                    ]):
                with self.assertRaisesRegex(ValueError, "no failure-splitting"):
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
                                    previous_narration="", tokenizer=FakeTokenizer())
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

            with patch("story_writer.writer.generate_response", side_effect=generate) as call:
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
            ]) as call:
                with self.assertRaisesRegex(ValueError, "no failure-splitting"):
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
