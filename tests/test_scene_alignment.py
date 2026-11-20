"""Page prose shares one speech track across every source panel, without silence gaps."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch, MagicMock

from story_writer.display import narration_blocks
from story_writer.writer import _timeline
from story_writer.scenes import generate_scene_story, scene_input
from story_writer.evidence import build_ledger
from story_writer.writer import prepare_story, run_story_writer
from test_story_writer import _fixture, FakeProcessor
from test_video import load_inputs, create_video


def passing_review(ledger, draft):
    """Mock the wire format, not a factual-quality oracle."""
    return {"fact_checks": [{"fact": f["key"], "sentences": [i for i, s in enumerate(draft["sentences"], 1) if f["key"] in s["facts"]],
                             "verdict": "preserved" if any(f["key"] in s["facts"] for s in draft["sentences"]) else "excluded" if not f["usable"] else "omitted_incidental",
                             "reason": "Optional detail."} for f in ledger["facts"]],
            "sentence_checks": [{"sentence": i, "verdict": "supported", "reason": ""} for i in range(1, len(draft["sentences"]) + 1)],
            "paragraph_issues": []}


class SceneAlignmentTests(unittest.TestCase):
    def rows(self):
        return [
            {"primary_panel_id": "a", "supporting_panel_ids": ["a"], "text": "", "used_instance_ids": [], "name_uses": [],
             "narration_policy": "scene_visual_v1", "narrated_by_panel_id": "b", "chapter": "c", "page": "1.jpg"},
            {"primary_panel_id": "b", "supporting_panel_ids": ["a", "b"], "text": "He makes a dangerous discovery.",
             "scene_panel_ids": ["a", "b"], "used_instance_ids": [], "name_uses": [], "chapter": "c", "page": "1.jpg"},
        ]

    def test_scene_retains_order_and_divides_duration_without_new_silence(self):
        rows = self.rows()
        self.assertEqual(narration_blocks(rows), [[0, 1]])
        lookup = {row["primary_panel_id"]: dict(row, panel_image="panel.jpg") for row in rows}
        timeline = _timeline(rows, lookup, 60)
        self.assertEqual([r["panel_id"] for r in timeline], ["a", "b"])
        self.assertEqual(timeline[0]["estimated_end_seconds"], 2.5)
        self.assertEqual(timeline[-1]["estimated_end_seconds"], 5)
        self.assertEqual([r["word_count"] for r in timeline], [0, 5])

    def test_broken_or_cross_page_scene_links_fail(self):
        changes = [(0, "narrated_by_panel_id", "missing"), (1, "scene_panel_ids", ["b", "a"]),
                   (1, "supporting_panel_ids", ["b"]), (1, "page", "2.jpg"),
                   (0, "text", "Invented speech."), (0, "name_uses", [{"name": "wrong"}])]
        for index, field, value in changes:
            rows = copy.deepcopy(self.rows())
            rows[index][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    narration_blocks(rows)

    def test_ordinary_segments_remain_individual_blocks(self):
        self.assertEqual(narration_blocks([{"primary_panel_id": "a", "text": "He waits."},
                                           {"primary_panel_id": "b", "text": "She arrives."}]), [[0], [1]])

    def test_page_prose_publishes_resumes_and_keeps_all_panels_in_video_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            first = prepared["panels"][0]
            second = copy.deepcopy(first)
            second.update(panel_id="second", panel_number=2, order=2, characters=[], accepted_actions=[])
            prepared["panels"].append(second)
            draft = {"sentences": [{"text": "The student runs through the corridor.", "facts": ["F1"], "characters": ["C1"], "speaker": None}]}
            review = passing_review(build_ledger(prepared["panels"]), draft)
            with patch("story_writer.writer.generate_response", side_effect=[(json.dumps(p), {"truncated": False}) for p in (draft, review)]) as generate:
                story = generate_scene_story(prepared, FakeProcessor(), object(), output)
                resumed = generate_scene_story(prepared, FakeProcessor(), object(), output)
                self.assertEqual(generate.call_count, 2)
            self.assertEqual(story["narration"], resumed["narration"])
            self.assertEqual([s["primary_panel_id"] for s in story["segments"]], [first["panel_id"], "second"])
            self.assertEqual(story["segments"][0]["text"], "")
            loaded, groups = load_inputs(output / "Example Manga_story.json", output / "Example Manga_panel_timeline.json")
            self.assertEqual(narration_blocks(loaded["segments"]), [[0, 1]])
            self.assertEqual([p for group in groups for p in group["panel_ids"]], [first["panel_id"], "second"])
            self.assertEqual(len(list((output / "Example Manga" / "story_runs").rglob("*.raw.txt"))), 2)
            wav = output / "one_scene.wav"
            with wave.open(str(wav), "wb") as stream:
                stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x00\x00" * 8000)

            def ffmpeg(command, **kwargs):
                Path(command[-1]).write_bytes(b"mock rendered video")

            with patch("test_video.shutil.which", return_value="tool"), \
                 patch("test_video.speech_files", return_value=[wav]) as speech, \
                 patch("test_video.render_frame"), patch("test_video.subprocess.run", side_effect=ffmpeg):
                video = create_video(output / "Example Manga_story.json", output / "Example Manga_panel_timeline.json", output)
            self.assertEqual(len(speech.call_args.args[0]), 1)
            audio = json.loads((video.parent / "audio_timeline.json").read_text(encoding="utf-8"))
            self.assertEqual([row["duration_seconds"] for row in audio["segments"]], [.5, .5])
            with wave.open(str(video.parent / "narration.wav")) as stream:
                self.assertEqual(stream.getnframes(), 8000)

    def test_scene_input_prioritizes_story_facts_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            panels = prepared["panels"]
            before = copy.deepcopy(panels)
            combined, compact = scene_input(panels)
            self.assertEqual(panels, before)
            self.assertEqual(combined["accepted_actions"], panels[0]["accepted_actions"])
            self.assertTrue(compact["ordered_facts"][0]["accepted_actions"])
            self.assertNotIn("visible_observation", compact["ordered_facts"][0])

    def test_incidental_poses_do_not_replace_meaningful_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            panels = prepare_story(manga, chapters, output)["panels"]
            original = panels[0]["accepted_actions"][0]
            panels[0]["accepted_actions"] = [dict(original, text=text) for text in (
                "standing", "touching his hair", "looking toward a screen", "standing guard", "grabs her hair", "runs away")]
            combined, compact = scene_input(panels)
            self.assertEqual(len(combined["accepted_actions"]), 6)
            self.assertEqual([a["text"] for a in compact["ordered_facts"][0]["accepted_actions"]],
                             ["looking toward a screen", "standing guard", "grabs her hair", "runs away"])

    def test_visible_bystanders_do_not_become_narration_assignments(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            panels = prepare_story(manga, chapters, output)["panels"]
            panels[0]["characters"].append({"instance_id": "bystander", "allowed_name": "Observer",
                                            "visual_description": "a blonde girl standing nearby"})
            combined, compact = scene_input(panels)
            self.assertNotIn("bystander", [c["instance_id"] for c in combined["characters"]])
            self.assertNotIn("Observer", json.dumps(compact))
            self.assertEqual(len(panels[0]["characters"]), 2)

    def test_reviewed_narration_is_retained_from_earlier_panels(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            first = prepare_story(manga, chapters, output)["panels"][0]
            first = dict(first, reviewed_narration="A war has already begun.")
            last = dict(first, panel_id="last", reviewed_narration=None)
            combined, compact = scene_input([first, last])
            self.assertIn("A war has already begun.", compact["ordered_facts"][0]["story_context"])
            self.assertTrue(any(f["text"] == "A war has already begun." for f in combined["accepted_text_facts"]))

    def test_invalid_scene_attempts_remain_bounded_without_final_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            with patch("story_writer.writer.generate_response", return_value=("bad json", {"truncated": False})) as generate:
                with self.assertRaisesRegex(RuntimeError, "four attempts"):
                    generate_scene_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(generate.call_count, 4)
            self.assertFalse((output / "Example Manga_story.json").exists())
            self.assertEqual(len(list((output / "Example Manga" / "story_runs").rglob("*.raw.txt"))), 4)

    def test_scene_dispatch_does_not_parse_checkpoint_as_legacy_panel_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            (output / "Example Manga_story_progress.json").write_text("scene checkpoint is handled by scene writer")
            model = MagicMock()
            with patch("story_writer.writer.prepare_story", return_value=prepared), \
                 patch("story_writer.writer.load_story_model", return_value=(FakeProcessor(), model)), \
                 patch("story_writer.scenes.generate_scene_story", return_value={"segments": []}) as generate:
                run_story_writer(manga, chapters, output, narration_style="scene")
                generate.assert_called_once()
                model.close.assert_called_once()

    def test_first_person_thought_retry_keeps_grounded_thinker_and_all_facts(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            panel = prepared["panels"][0]
            panel["accepted_text_facts"] = [{"kind": "thought", "text": "I was gaming in my dreams.", "speaker_instance_id": "instance_1"}]
            invalid = {"sentences": [{"text": "I was gaming in my dreams.", "facts": ["F1"], "characters": ["C1"], "speaker": "C1"}]}
            corrected = {"sentences": [
                {"text": 'He thinks, “I was gaming in my dreams.”', "facts": ["F1"], "characters": ["C1"], "speaker": "C1"},
                {"text": "He runs through the corridor.", "facts": ["F2"], "characters": ["C1"], "speaker": None}]}
            review = passing_review(build_ledger([panel]), corrected)
            with patch("story_writer.writer.generate_response", side_effect=[(json.dumps(p), {"truncated": False}) for p in (invalid, corrected, review)]) as generate:
                story = generate_scene_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(generate.call_count, 3)
            self.assertIn("Missing required facts: F2", generate.call_args_list[1].args[2])
            self.assertEqual(story["segments"][0]["used_instance_ids"], ["instance_1"])

    def test_unknown_quotation_newlines_use_json_not_fragile_decoder_patterns(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            panel = prepared["panels"][0]
            panel["accepted_text_facts"] = [{"kind": "dialogue", "text": "Honestly,\nyou...", "speaker_instance_id": None}]
            invalid = {"sentences": [{"text": 'He says, "Honestly, you..."', "facts": ["F1", "F2"], "characters": ["C1"], "speaker": "C1"}]}
            corrected = {"sentences": [
                {"text": '“Honestly, you...”', "facts": ["F1"], "characters": [], "speaker": None},
                {"text": "He runs through the corridor.", "facts": ["F2"], "characters": ["C1"], "speaker": None}]}
            review = passing_review(build_ledger([panel]), corrected)
            with patch("story_writer.writer.generate_response", side_effect=[(json.dumps(p), {"truncated": False}) for p in (invalid, corrected, review)]) as generate:
                generate_scene_story(prepared, FakeProcessor(), object(), output)
            props = generate.call_args_list[1].kwargs["response_schema"]["properties"]["sentences"]["items"]["anyOf"][0]["properties"]
            self.assertNotIn("pattern", props["text"])
            self.assertEqual(generate.call_count, 3)

    def test_review_rejection_retries_without_publishing_and_accepted_resume_revalidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            draft = {"sentences": [{"text": "He runs through the corridor.", "facts": ["F1"], "characters": ["C1"], "speaker": None}]}
            review = passing_review(build_ledger(prepared["panels"]), draft)
            rejected = copy.deepcopy(review)
            rejected["sentence_checks"][0].update(verdict="wrong_attribution", reason="Wrong actor.")
            with patch("story_writer.writer.generate_response", side_effect=[(json.dumps(p), {}) for p in (draft, rejected, draft, review)]) as generate:
                original = generate_scene_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(generate.call_count, 4)
            self.assertIn("Wrong actor", generate.call_args_list[2].args[2])
            with patch("story_writer.writer.generate_response") as generate:
                resumed = generate_scene_story(prepared, FakeProcessor(), object(), output)
                generate.assert_not_called()
            self.assertEqual(original, resumed)
            path = output / "Example Manga_story_progress.json"
            state = json.loads(path.read_text(encoding="utf-8"))
            state["scenes"][0]["segment"]["text"] = "He destroys the world."
            path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Saved scene evidence"):
                generate_scene_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(json.loads((output / "Example Manga_story.json").read_text(encoding="utf-8")), original)

    def test_persistent_review_failure_never_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            draft = {"sentences": [{"text": "He runs through the corridor.", "facts": ["F1"], "characters": ["C1"], "speaker": None}]}
            review = passing_review(build_ledger(prepared["panels"]), draft)
            review["paragraph_issues"] = ["Meaning is unclear."]
            with patch("story_writer.writer.generate_response", side_effect=[(json.dumps(p), {}) for p in (draft, review) * 4]) as generate:
                with self.assertRaisesRegex(RuntimeError, "four attempts"):
                    generate_scene_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(generate.call_count, 8)
            self.assertFalse((output / "Example Manga_story.json").exists())
            self.assertFalse(json.loads((output / "Example Manga_story_progress.json").read_text(encoding="utf-8"))["complete"])

    def test_truncated_review_cannot_approve_even_when_its_json_looks_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            draft = {"sentences": [{"text": "He runs through the corridor.", "facts": ["F1"], "characters": ["C1"], "speaker": None}]}
            review = passing_review(build_ledger(prepared["panels"]), draft)
            responses = [(json.dumps(draft), {}), (json.dumps(review), {"truncated": True})] * 4
            with patch("story_writer.writer.generate_response", side_effect=responses):
                with self.assertRaisesRegex(RuntimeError, "truncated"):
                    generate_scene_story(prepared, FakeProcessor(), object(), output)
            self.assertFalse((output / "Example Manga_story.json").exists())

    def test_transport_failure_retains_attempt_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            with patch("story_writer.writer.generate_response", side_effect=RuntimeError("Transport unavailable")):
                with self.assertRaisesRegex(RuntimeError, "Transport unavailable"):
                    generate_scene_story(prepared, FakeProcessor(), object(), output)
            self.assertFalse((output / "Example Manga_story.json").exists())
            metrics = list((output / "Example Manga" / "story_runs").rglob("*.metrics.json"))
            self.assertEqual(len(metrics), 1)
            self.assertEqual(json.loads(metrics[0].read_text(encoding="utf-8"))["error"], "Transport unavailable")

    def test_oversized_scene_fails_before_model_without_dropping_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            prepared["panels"][0]["accepted_text_facts"] = [{"kind": "narration", "text": "fact " * 20000}]
            with patch("story_writer.writer.generate_response") as generate:
                with self.assertRaisesRegex(ValueError, "no evidence was dropped"):
                    generate_scene_story(prepared, FakeProcessor(), object(), output)
                generate.assert_not_called()
            self.assertFalse((output / "Example Manga_story.json").exists())

    def test_alias_cooldown_does_not_rewrite_the_original_source_quotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            panels = prepare_story(manga, chapters, output)["panels"]
            panels[0]["characters"][0]["allowed_name"] = "Akira"
            panels[0]["accepted_text_facts"] = [{"kind": "dialogue", "text": "Akira, wait!", "speaker_instance_id": None}]
            before = copy.deepcopy(panels)
            combined, _ = scene_input(panels, [{"name_uses": [{"name": "Akira", "instance_id": "instance_1"}]}])
            self.assertEqual(combined["accepted_text_facts"][0]["text"], "Akira, wait!")
            self.assertIsNone(combined["characters"][0]["allowed_name"])
            self.assertEqual(panels, before)

    def test_reviewed_optional_pose_can_be_a_hold_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            manga, chapters, output = _fixture(Path(temporary))
            prepared = prepare_story(manga, chapters, output)
            prepared["panels"][0]["accepted_actions"][0]["text"] = "standing nearby"
            prepared["panels"][0]["description"] = "A student stands nearby."
            before = copy.deepcopy(prepared)
            draft = {"sentences": []}
            review = passing_review(build_ledger(prepared["panels"]), draft)
            with patch("story_writer.writer.generate_response", side_effect=[(json.dumps(p), {}) for p in (draft, review)]):
                story = generate_scene_story(prepared, FakeProcessor(), object(), output)
            self.assertEqual(story["segments"][0]["narration_policy"], "visual_hold_v1")
            self.assertEqual(prepared, before)
            self.assertFalse(story["metadata"]["quality_checks"]["production_approved"])

    def test_stage_cli_defaults_to_scene_style(self):
        import test_write_storry
        with patch.object(test_write_storry, "find_manga_dir", return_value=Path("Example")), \
             patch.object(test_write_storry, "find_chapter_dirs", return_value=[]), \
             patch.object(test_write_storry, "run_story_writer", return_value={"segments": []}) as run:
            test_write_storry.main(["Example"])
            self.assertEqual(run.call_args.kwargs["narration_style"], "scene")
