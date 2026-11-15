"""Model-free regressions for the shared remote/local evidence instructions."""

import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import config
from description.describer import (
    OPENROUTER_COMBINED_PROMPT, PAGE_PROMPT, PANEL_OBSERVATION_PROMPT,
    ROLE_PROMPT, TEXT_EVIDENCE_PROMPT, VISIBLE_EVIDENCE_RULES,
    _openrouter_bundle_schema, describe_character_pages, describe_focused_pages, describe_manga,
)
from description.format import validate_format
from description.response import parse_page_response
from description.schema import panel_observation_schema
from story_entities.manifest import file_hash


class EvidenceInstructionTests(unittest.TestCase):
    def chapter_fixture(self, output):
        chapter = output / "chapter_001"
        previews = chapter / "page_previews"
        previews.mkdir(parents=True)
        pages = []
        for number in (1, 2):
            source = output / f"{number:05}.jpg"
            source.write_bytes(b"source")
            preview = previews / f"{number:05}.jpg"
            preview.write_bytes(b"preview")
            pages.append({"page": str(source), "source_hash": file_hash(source),
                          "preview_hash": file_hash(preview), "geometry_revision": "g",
                          "panels": [{"panel_number": 1}]})
        (chapter / "panel_manifest.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")
        return chapter

    def one_panel_bundle(self):
        return {"r": "story", "p": [{"n": 1, "d": "A student raises a hand.", "a": []}],
                "c": {}, "e": [], "x": {"p": [{"n": 1, "t": []}]}}

    def test_bad_remote_page_does_not_loop_or_stop_later_pages(self):
        for error in ("format", "network"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                chapter = self.chapter_fixture(output)
                valid = {"content": json.dumps(self.one_panel_bundle()), "finish_reason": "stop"}
                broken = {"content": '{"p":[]}', "finish_reason": "stop"}
                client = self.client([])
                client.cache_key = ("test",)
                failures = [broken, broken] if error == "format" else [RuntimeError("network unavailable")]
                client.complete_image.side_effect = failures + [valid]
                snapshots = []
                with patch("description.describer.time.sleep") as sleep:
                    result = describe_manga(output, [chapter], backend=client, filter_non_story=True,
                                            progress_callback=lambda value: snapshots.append(copy.deepcopy(value)))
                sleep.assert_not_called()
                calls = [call.kwargs["image_path"].name for call in client.complete_image.call_args_list]
                self.assertEqual(calls, ["00001.jpg"] * len(failures) + ["00002.jpg"])
                self.assertEqual(set(result["descriptions"][chapter.name]), {"00002.jpg"})
                self.assertEqual(result["pending_pages"], {chapter.name: ["00001.jpg"]})
                self.assertNotIn("00001.jpg", result["page_filter"]["chapters"][chapter.name])
                self.assertTrue(all(row["extraction_status"] == "complete"
                                    for row in result["descriptions"][chapter.name].values()))
                self.assertEqual(set(snapshots[1]["descriptions"][chapter.name]), {"00002.jpg"})
                attempts = list((chapter / "description_attempts").glob("*.json"))
                self.assertEqual(len(attempts), 1)
                diagnostic = json.loads(attempts[0].read_text(encoding="utf-8"))
                self.assertEqual(len(diagnostic["raw_responses"]), 2 if error == "format" else 0)
                checkpoint = json.loads((chapter / "description_checkpoint_openrouter.json").read_text(encoding="utf-8"))
                self.assertFalse(checkpoint["complete"])

    def test_resuming_incomplete_run_only_processes_pending_page(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            chapter = self.chapter_fixture(output)
            valid = {"content": json.dumps(self.one_panel_bundle()), "finish_reason": "stop"}
            client = self.client([])
            client.cache_key = ("test",)
            client.complete_image.side_effect = [RuntimeError("network unavailable"), valid]
            describe_manga(output, [chapter], backend=client, filter_non_story=True)
            checkpoint = json.loads((chapter / "description_checkpoint_openrouter.json").read_text(encoding="utf-8"))
            self.assertFalse(checkpoint["complete"])
            self.assertEqual(set(checkpoint["result"]["descriptions"]), {"00002.jpg"})
            client.complete_image.reset_mock(side_effect=True)
            client.complete_image.side_effect = [valid]
            result = describe_manga(output, [chapter], backend=client, filter_non_story=True)
            client.complete_image.assert_called_once()
            self.assertEqual(client.complete_image.call_args.kwargs["image_path"].name, "00001.jpg")
            self.assertNotIn("pending_pages", result)
            self.assertEqual(set(result["descriptions"][chapter.name]), {"00001.jpg", "00002.jpg"})
            checkpoint = json.loads((chapter / "description_checkpoint_openrouter.json").read_text(encoding="utf-8"))
            self.assertTrue(checkpoint["complete"])

    def test_network_error_during_format_repair_keeps_original_response(self):
        from description.format import FormatResponseError
        client = self.client([])
        client.complete_image.side_effect = [
            {"content": "broken JSON", "finish_reason": "stop"}, RuntimeError("network unavailable"),
        ]
        with self.assertRaises(FormatResponseError) as caught:
            describe_focused_pages([(Path("page.jpg"), [1], [])], client, client, task="observation")
        self.assertEqual(caught.exception.responses, ["broken JSON"])
        self.assertIn("network unavailable", str(caught.exception))

    def test_interruption_resumes_without_repeating_completed_page(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            chapter = self.chapter_fixture(output)
            valid = {"content": json.dumps(self.one_panel_bundle()), "finish_reason": "stop"}
            client = self.client([])
            client.cache_key = ("test",)
            client.complete_image.side_effect = [valid, KeyboardInterrupt()]
            with self.assertRaises(KeyboardInterrupt):
                describe_manga(output, [chapter], backend=client, filter_non_story=True)
            checkpoint_path = chapter / "description_checkpoint_openrouter.json"
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertFalse(saved["complete"])
            self.assertEqual(set(saved["result"]["descriptions"]), {"00001.jpg"})
            client.complete_image.reset_mock(side_effect=True)
            client.complete_image.side_effect = [valid]
            result = describe_manga(output, [chapter], backend=client, filter_non_story=True)
            client.complete_image.assert_called_once()
            self.assertEqual(client.complete_image.call_args.kwargs["image_path"].name, "00002.jpg")
            self.assertEqual(set(result["descriptions"][chapter.name]), {"00001.jpg", "00002.jpg"})
            self.assertTrue(json.loads(checkpoint_path.read_text())["complete"])

    def test_permanent_google_error_stops_without_spending_on_later_pages(self):
        from description.google import GooglePermanentError
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            chapter = self.chapter_fixture(output)
            client = self.client([])
            client.cache_key = ("test",)
            client.complete_image.side_effect = GooglePermanentError("HTTP 429: quota")
            with self.assertRaises(GooglePermanentError):
                describe_manga(output, [chapter], backend=client, filter_non_story=True)
            client.complete_image.assert_called_once()

    def test_remote_and_local_share_visible_action_and_repeated_depiction_rules(self):
        for prompt in (OPENROUTER_COMBINED_PROMPT, PAGE_PROMPT, PANEL_OBSERVATION_PROMPT):
            self.assertIn(VISIBLE_EVIDENCE_RULES, prompt)
            self.assertIn("do not invent interactions", prompt)
            self.assertIn("repeated depictions", prompt)
            self.assertIn("otherwise use null", prompt)
        for prompt in (ROLE_PROMPT, OPENROUTER_COMBINED_PROMPT):
            self.assertIn("mixed", prompt)
            self.assertIn("story", prompt)
            self.assertIn("title", prompt)
        self.assertIn("a distinct directional extension", TEXT_EVIDENCE_PROMPT)
        self.assertIn("story-relevant captions", " ".join(TEXT_EVIDENCE_PROMPT.split()))

    def test_speaker_prompts_distinguish_decorative_spikes_from_tails(self):
        for template in (OPENROUTER_COMBINED_PROMPT, TEXT_EVIDENCE_PROMPT):
            prompt = " ".join(template.format(panels="[1, 2]", labels="C1=panel 1").split())
            with self.subTest(prompt=template[:40]):
                self.assertIn("jagged edges, sharp corners or repeated spikes", prompt)
                self.assertIn("These points are not automatically speaker tails", prompt)
                self.assertIn("On an otherwise round bubble, a single pointed extension may be the tail", prompt)
                self.assertIn("even if very small; shape alone does not prove its speaker", prompt)
                self.assertIn("With many sharp corners, be especially cautious: use s:null rather than guess", prompt)
                self.assertIn("a distinct directional extension, separate from the general border pattern", prompt)
                self.assertIn("one supplied C-labeled character in the same panel", prompt)
                self.assertIn("cannot distinguish a tail from the border design, use s:null", prompt)
                self.assertIn("Keep the readable text as dialogue", prompt)

    def test_outside_bubble_text_is_not_automatically_speech_or_thought(self):
        for template in (OPENROUTER_COMBINED_PROMPT, TEXT_EVIDENCE_PROMPT):
            prompt = " ".join(template.format(panels="[1]", labels="C1=panel 1").split())
            self.assertIn("Ignore incidental text outside bubbles", prompt)
            self.assertIn("story-relevant captions, names, ranks and reveals as narration", prompt)
            self.assertIn("lettering on scene objects as sign", prompt)
            self.assertIn("without a visible speech/thought cue", prompt)
            self.assertIn("Importance alone does not make it a character's thought", prompt)

    def test_extra_role_requires_nonstory_or_clearly_unrelated_content(self):
        for template in (ROLE_PROMPT, OPENROUTER_COMBINED_PROMPT):
            prompt = " ".join(template.split())
            self.assertIn("clearly unrelated standalone", prompt)
            self.assertIn("Extra-like styling is not enough", prompt)
            self.assertIn("new scene or unfamiliar character does not prove unrelatedness", prompt)
            self.assertIn("use uncertain, not extra", prompt)

    def test_panel_scope_is_not_restricted_to_character_boxes(self):
        for template in (OPENROUTER_COMBINED_PROMPT, PAGE_PROMPT, PANEL_OBSERVATION_PROMPT):
            prompt = " ".join(template.split())
            self.assertIn("Green borders with purple numbers mark panels", prompt)
            self.assertIn("Cyan boxes with red C labels mark character occurrences, not panels", prompt)
            self.assertIn("including artwork outside C boxes", prompt)
            self.assertIn("Only character appearances are restricted to their own C boxes", prompt)
            self.assertNotIn("Describe only what is inside each box", prompt)
        for template in (OPENROUTER_COMBINED_PROMPT, TEXT_EVIDENCE_PROMPT):
            self.assertIn("Read text throughout each green panel region, not just inside C boxes", template)

    def test_visual_prompts_check_for_lettering_before_describing_effects(self):
        for template in (OPENROUTER_COMBINED_PROMPT, PAGE_PROMPT, PANEL_OBSERVATION_PROMPT):
            prompt = " ".join(template.split())
            self.assertIn("first check whether its shapes could be Japanese or Chinese characters", prompt)
            self.assertIn("You do not need to read the characters", prompt)
            self.assertIn("Only describe a physical effect when you can clearly distinguish it from lettering", prompt)
            self.assertIn("If unsure, skip the effect rather than guess", prompt)
            self.assertLess(prompt.index("Before describing an effect"), prompt.index("First identify the visible posture"))

    def test_abstention_preserves_labels_and_empty_actions_without_repair(self):
        value = self.bundle()
        value["c"]["C1"] = {"d": "", "r": "uncertain"}
        value["c"]["C2"]["d"] = "hair"
        value["p"][0]["a"] = []
        validate_format(value, _openrouter_bundle_schema([1, 2], self.labels()))
        parsed = parse_page_response(json.dumps(value), expected_panels=[1, 2],
                                     character_labels=self.labels())
        self.assertEqual(parsed["grounding_errors"], [])
        self.assertEqual(parsed["characters"][0]["visual_description"], "")
        self.assertEqual(parsed["characters"][0]["appearance_status"], "unknown")
        self.assertEqual(parsed["characters"][0]["detection_id"], "d1")
        client = self.client([value])
        describe_focused_pages([(Path("page.jpg"), [1, 2], self.labels())],
                               client, client, task="observation")
        client.page_bundle.return_value = value
        result = describe_character_pages([(Path("page.jpg"), self.labels())], client, client)[0]
        self.assertEqual(result["grounding_errors"], [])
        self.assertEqual(len(result["characters"]), 2)
        self.assertEqual(result["characters"][0]["appearance_status"], "unknown")
        self.assertEqual(result["characters"][1]["visual_description"], "hair")
        client.complete_image.assert_called_once()

    def test_abstention_does_not_accept_missing_or_malformed_descriptors(self):
        for descriptor in (None, {"d": "", "r": "invalid"}, {"r": "uncertain"}):
            with self.subTest(descriptor=descriptor):
                client = self.client([])
                client.page_bundle.return_value = {"c": {"C1": descriptor}}
                result = describe_character_pages([(Path("page.jpg"), self.labels()[:1])],
                                                  client, client)[0]
                self.assertTrue(result["grounding_errors"])
                self.assertEqual(result["characters"], [])

    def test_prompt_prefers_abstention_without_minimum_appearance_length(self):
        prompt = " ".join(OPENROUTER_COMBINED_PROMPT.split())
        self.assertIn("First identify the visible posture, then held objects", prompt)
        self.assertIn("cannot distinguish a tail from the border design, use s:null", prompt)
        self.assertIn('use d:"" if none are clear', prompt)
        self.assertNotIn("3-10", prompt)
        self.assertNotIn("equally clear visual link", prompt)

    def test_simple_prompt_prioritizes_grounding_and_disables_new_sound_effects(self):
        prompt = OPENROUTER_COMBINED_PROMPT.format(panels="[1, 2]", labels="C1=panel 1")
        self.assertLess(len(prompt.split()), 700)
        self.assertLess(prompt.index("use s:null"), prompt.index("\np:"))
        self.assertIn("Skip sound effects", prompt)
        schema = _openrouter_bundle_schema([1, 2], self.labels())
        value = self.bundle()
        value["x"]["p"][0]["t"] = [{"k": "dialogue", "x": "Hello", "s": None}]
        validate_format(value, schema)
        value["x"]["p"][0]["t"][0]["k"] = "sound_effect"
        with self.assertRaises(ValueError):
            validate_format(value, schema)

    def bundle(self):
        return {
            "r": "story", "p": [{"n": 1, "d": "A student stands outside.", "a": [
                {"s": "C1", "t": None, "x": "raises an umbrella"},
            ]}, {"n": 2, "d": "A second student watches.", "a": []}],
            "c": {"C1": {"d": "dark-haired uniformed student", "r": "live_character"},
                  "C2": {"d": "light-haired uniformed student", "r": "live_character"}},
            "e": [], "x": {"p": [{"n": 1, "t": []}, {"n": 2, "t": []}]},
        }

    def labels(self):
        return [{"label": "C1", "panel_number": 1, "detection_id": "d1"},
                {"label": "C2", "panel_number": 2, "detection_id": "d2"}]

    def client(self, responses):
        client = MagicMock()
        client.is_remote_description_client = True
        client.checkpoint_name = "description_checkpoint_openrouter.json"
        client.provenance_prefix = "test_remote"
        client.provider_name = "test"
        client.model = "test-model"
        client.base_url = "https://example.invalid"
        client.complete_image.side_effect = [{"content": json.dumps(value), "finish_reason": "stop"}
                                              for value in responses]
        return client

    def test_remote_richer_descriptions_survive_without_rewriting_or_extra_call(self):
        value = self.bundle()
        text = ("A uniformed student crouches beside a capsule machine and examines its window. "
                "A second student stands nearby, while a falling capsule is visible below the machine.")
        self.assertGreater(len(text), config.DESCRIPTION_MAX_DESCRIPTION_CHARS)
        value["p"][0]["d"] = text
        client = self.client([value])
        result = describe_focused_pages([(Path("page.jpg"), [1, 2], self.labels())],
                                       client, client, task="observation")[0]
        self.assertEqual(result["payload"]["p"][0]["d"], text)
        client.complete_image.assert_called_once()
        self.assertIn(VISIBLE_EVIDENCE_RULES, client.complete_image.call_args.kwargs["prompt"])
        self.assertEqual(panel_observation_schema([1], [])[
            "properties"]["p"]["items"]["properties"]["d"]["maxLength"],
            config.DESCRIPTION_MAX_DESCRIPTION_CHARS)

    def test_remote_wrong_panel_actor_or_speaker_gets_one_format_repair(self):
        valid = self.bundle()
        for kind in ("actor", "speaker"):
            with self.subTest(kind=kind):
                invalid = copy.deepcopy(valid)
                if kind == "actor":
                    invalid["p"][0]["a"][0]["s"] = "C2"
                else:
                    invalid["x"]["p"][0]["t"] = [{"k": "dialogue", "x": "Wait!", "s": "C2"}]
                client = self.client([invalid, valid])
                result = describe_focused_pages([(Path("page.jpg"), [1, 2], self.labels())],
                                               client, client, task="observation")[0]
                self.assertEqual(client.complete_image.call_count, 2)
                self.assertEqual(result["retry_count"], 1)
                repair_prompt = client.complete_image.call_args_list[1].kwargs["prompt"]
                self.assertIn("Character label C2 is not in the referenced panel 1", repair_prompt)
                self.assertIn("Allowed labels in panel 1: C1", repair_prompt)
                self.assertIn("Use null", repair_prompt)
                self.assertIn("do not guess a replacement label", repair_prompt)
                self.assertEqual([call.kwargs["max_tokens"] for call in client.complete_image.call_args_list],
                                 [result["token_limit"]] * 2)
                self.assertEqual(len(result["prior_raw_responses"]), 1)

    def test_unlabeled_visible_action_is_retained_without_inventing_an_actor(self):
        schema = _openrouter_bundle_schema([1], [])
        value = {"r": "story", "p": [{"n": 1, "d": "Lightning strikes the ground.",
                    "a": [{"s": None, "t": None, "x": "strikes the ground"}]}],
                 "c": {}, "e": [], "x": {"p": [{"n": 1, "t": []}]}}
        validate_format(value, schema)
        client = self.client([value])
        result = describe_focused_pages([(Path("page.jpg"), [1], [])], client, client, task="observation")[0]
        self.assertEqual(result["payload"]["p"][0]["a"][0]["s"], None)
        self.assertEqual(result["payload"]["p"][0]["d"], "Lightning strikes the ground.")

    def test_publication_wording_is_preserved_but_typed_separately_from_story(self):
        from description.describer import _parse_text_proposals
        value = {"p": [{"n": 1, "t": [
            {"k": "publication", "x": "  TWO-PART SPECIAL ONE-SHOT  ", "s": None},
            {"k": "narration", "x": "A title reserved for the strongest.", "s": None},
            {"k": "dialogue", "x": "There's even a change machine...", "s": None},
        ]}]}
        schema = _openrouter_bundle_schema([1], [])["properties"]["x"]
        validate_format(value, schema)
        rows, errors = _parse_text_proposals({"status": "complete", "payload": value}, {
            "source_hash": "hash", "panels": [{"panel_number": 1, "panel_id": "panel_1"}],
        })
        self.assertEqual(errors, [])
        self.assertEqual([row["kind"] for row in rows], ["publication", "narration", "dialogue"])
        self.assertEqual(rows[0]["text"], "  TWO-PART SPECIAL ONE-SHOT  ")
        self.assertIsNone(rows[2]["speaker_detection_id"])


if __name__ == "__main__":
    unittest.main()
