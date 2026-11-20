"""Silent static panels retain evidence/order without forcing caption narration."""
import json
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from story_writer.writer import validate_story_response, _story_response_schema, _timeline, _panel_prompt_record, _initial_chunks, prepare_story, generate_story
from story_writer.display import VISUAL_HOLD_SECONDS
from test_video import speech_files, combine_speech, align_groups, load_inputs
from test_story_writer import FakeTokenizer, FakeProcessor, _fixture


class VisualHoldTests(unittest.TestCase):
    def setUp(self):
        self.panel = {"panel_id": "p", "chapter": "c", "page": "1.jpg", "panel_number": 1,
                      "panel_image": "p.jpg", "characters": [], "description": "A close-up of hair.",
                      "accepted_actions": [], "accepted_transformations": [], "accepted_text_facts": []}
        self.segment = {"primary_panel_id": "p", "supporting_panel_ids": ["p"], "text": "",
                        "used_instance_ids": [], "name_uses": [], "narration_policy": "visual_hold_v1"}

    def test_explicit_hold_validates_and_retains_timeline_coverage(self):
        rows, errors = validate_story_response(json.dumps({"segments": [self.segment]}), [self.panel])
        self.assertEqual(errors, [])
        self.assertEqual(rows, [self.segment])
        timeline = _timeline(rows, {"p": self.panel}, 150)
        self.assertEqual(timeline[0]["word_count"], 0)
        self.assertEqual(timeline[0]["estimated_end_seconds"], VISUAL_HOLD_SECONDS)
        self.assertTrue(_panel_prompt_record(self.panel)["visual_hold_allowed"])

    def test_missing_policy_and_forged_holds_fail(self):
        for changed in ({"narration_policy": None}, {"narration_policy": "other"},
                        {"text": "An invented event happens."}, {"used_instance_ids": ["i"]},
                        {"supporting_panel_ids": ["future"]}, {"name_uses": [{"name": "Akira"}]}):
            with self.subTest(changed=changed):
                _, errors = validate_story_response(json.dumps({"segments": [dict(self.segment, **changed)]}), [self.panel])
                self.assertTrue(errors)

    def test_accepted_facts_cannot_be_silenced(self):
        for key, value in (("accepted_actions", [{"text": "runs"}]),
                           ("accepted_transformations", [{"text": "transforms"}]),
                           ("accepted_text_facts", [{"kind": "dialogue", "text": "Run!", "speaker_instance_id": None}]),
                           ("accepted_text_facts", [{"kind": "narration", "text": "A war begins."}]),
                           ("reviewed_narration", "He waits."), ("evidence_density", "uncertain")):
            panel = dict(self.panel, **{key: value})
            with self.subTest(key=key):
                _, errors = validate_story_response(json.dumps({"segments": [self.segment]}), [panel])
                self.assertTrue(errors)
                schema = _story_response_schema([panel])
                self.assertNotIn("visual_hold_v1", json.dumps(schema))

    def test_static_hold_is_optional_schema_branch(self):
        schema = _story_response_schema([dict(self.panel, description="A quiet room.")])
        self.assertIn("visual_hold_v1", json.dumps(schema))
        self.assertEqual(schema["properties"]["segments"]["minItems"], 1)

    def test_only_explicit_static_fragments_require_silent_holds(self):
        for description, required in (("Lower body and legs of a person standing.", True),
                                      ("A character standing in a doorway in the background.", True),
                                      ("A close-up view of blonde hair.", True),
                                      ("A girl with blonde hair runs from a fire.", False)):
            panel = dict(self.panel, description=description)
            with self.subTest(description=description):
                self.assertEqual(_panel_prompt_record(panel)["visual_hold_required"], required)
                variants = _story_response_schema([panel])["properties"]["segments"]["items"]["anyOf"]
                self.assertEqual(len(variants), 1 if required else 2)

    def test_hold_survives_publication_resume_and_video_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            panel = prepared["panels"][0]
            panel.update(accepted_actions=[], accepted_transformations=[], accepted_text_facts=[])
            segment = dict(self.segment, primary_panel_id=panel["panel_id"], supporting_panel_ids=[panel["panel_id"]])
            with patch("story_writer.writer.generate_response", return_value=(json.dumps({"segments": [segment]}), {"truncated": False})) as generate:
                generate_story(prepared, FakeProcessor(), object(), output)
                story = generate_story(prepared, FakeProcessor(), object(), output)
                generate.assert_called_once()
            self.assertEqual(story["segments"][0]["narration_policy"], "visual_hold_v1")
            loaded, groups = load_inputs(output / "Example Manga_story.json", output / "Example Manga_panel_timeline.json")
            self.assertEqual(loaded["segments"][0]["text"], "")
            self.assertEqual(groups[0]["panel_ids"], [panel["panel_id"]])

    def test_observed_framing_failures_are_rejected_without_banning_story_actions(self):
        cases = [("A person stands with legs visible.", True),
                 ("He is shown in a close-up view.", True),
                 ("He touches his hair in profile.", True),
                 ("A caption box declares that a war begins.", True),
                 ("He steps inside, but the guard refuses to let him pass.", False)]
        for text, rejected in cases:
            segment = dict(self.segment, text=text)
            del segment["narration_policy"]
            with self.subTest(text=text):
                _, errors = validate_story_response(json.dumps({"segments": [segment]}), [dict(self.panel, description="An arrival.")])
                self.assertEqual(bool(errors), rejected, errors)

    def test_first_person_is_allowed_only_inside_source_quotations(self):
        for text, rejected in [('I came to love him as well.', True),
                               ('Of course, we read every book.', True),
                               ('He admits, "I came to love him as well."', False),
                               ("He admits, 'I came to love him as well.'", False),
                               ("Someone declares, 'Then I'll take her.'", False),
                               ('He admits, “I came to love him as well.”', False),
                               ("He doesn't want to leave.", False)]:
            segment = dict(self.segment, text=text)
            del segment["narration_policy"]
            with self.subTest(text=text):
                _, errors = validate_story_response(json.dumps({"segments": [segment]}), [dict(self.panel, description="An admission.")])
                self.assertEqual(bool(errors), rejected, errors)

    def test_same_page_chunks_do_not_leak_across_scene_boundary(self):
        panels = [self.panel, dict(self.panel, panel_id="p2"), dict(self.panel, panel_id="p3", page="2.jpg")]
        with patch("config.STORY_CHUNK_MAX_PANELS", 4):
            chunks = _initial_chunks({"panels": panels}, FakeTokenizer(), [])
        self.assertEqual([[p["panel_id"] for p in group] for group in chunks], [["p", "p2"], ["p3"]])

    def test_silence_before_and_after_speech_matches_real_audio_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("test_video.subprocess.run") as tts:
                self.assertEqual(speech_files([self.segment], "voice", root / "cache", root), [None])
                tts.assert_not_called()
            speech = root / "voice.wav"
            with wave.open(str(speech), "wb") as stream:
                stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x01\x00" * 8000)
            durations = combine_speech([None, speech, None], root / "combined.wav")
            self.assertEqual(durations, [VISUAL_HOLD_SECONDS, 1, VISUAL_HOLD_SECONDS])
            with wave.open(str(root / "combined.wav")) as stream:
                self.assertEqual(stream.getnframes(), 20000)
                self.assertEqual(stream.readframes(6000), b"\x00" * 12000)
            segments = [{"primary_panel_id": p} for p in ("a", "b", "c")]
            groups = [{"panel_ids": [p]} for p in ("a", "b", "c")]
            self.assertEqual(align_groups(groups, segments, durations)[-1]["audio_end_seconds"], 2.5)
            with self.assertRaises(ValueError):
                combine_speech([None], root / "empty.wav")
