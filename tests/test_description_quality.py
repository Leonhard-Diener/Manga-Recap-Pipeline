"""CPU-only coverage for semantic gates and inference-only image preparation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from common.model_loading import _patch_safe_open

from description.describer import (
    _deterministic_page_role,
    _focused_token_limit,
    _complete_uncertain_panel_coverage,
    _parse_role,
    _resolve_model_page_role,
    classify_pages,
    describe_character_pages,
    describe_focused_pages,
)
from description.images import mask_text_regions
from description.quality import annotate_panel_quality
from description.response import parse_page_response
from panel_extractor.types import PageDetection
from story_entities.manifest import build_manifest


class QualityGateTests(unittest.TestCase):
    def test_text_budget_scales_with_detected_regions_and_retry(self):
        sparse = _focused_token_limit("text", 1, text_region_count=1)
        dense = _focused_token_limit("text", 1, text_region_count=12)
        retried = _focused_token_limit(
            "text", 1, text_region_count=12, token_multiplier=2,
        )
        self.assertGreater(dense, sparse)
        self.assertGreater(retried, dense)

    def test_failed_visual_output_does_not_manufacture_panel_coverage(self):
        page = {
            "status": "invalid_output", "panels": [],
            "quality_errors": {}, "errors": ["bad JSON"],
        }
        with self.assertRaisesRegex(ValueError, "no placeholder"):
            _complete_uncertain_panel_coverage(page, [1, 2])
        self.assertEqual(page["panels"], [])

    def test_windows_safetensors_compatibility_uses_pread_without_full_file_copy(self):
        calls = []

        def safe_open(*args, **kwargs):
            calls.append((args, kwargs))
            return "opened"

        module = SimpleNamespace(safe_open=safe_open)
        self.assertTrue(_patch_safe_open(module))
        self.assertEqual(module.safe_open("model.safetensors", framework="pt"), "opened")
        self.assertEqual(calls[-1][1]["backend"], "pread")
        module.safe_open("model.safetensors", framework="pt", backend="mmap")
        self.assertEqual(calls[-1][1]["backend"], "pread")
        module.safe_open("model.safetensors", framework="pt", backend="pread")
        self.assertEqual(calls[-1][1]["backend"], "pread")
        self.assertFalse(_patch_safe_open(module))

    def test_model_placeholder_is_preserved_without_keyword_rejection(self):
        issues, _ = annotate_panel_quality([{
            "panel_number": 1, "description": "Visible manga scene.",
        }])
        self.assertEqual(issues, {})
    def test_observed_failure_fixture_keeps_the_regression_pages(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "qwen_description_failures.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(set(fixture["chapter_001"]), {"00002", "00003", "00004", "00005"})
        self.assertEqual(
            set(fixture["chapter_002"]),
            {"00001", "00007", "00012", "00018", "00022", "00024"},
        )
        self.assertEqual(fixture["chapter_001"]["00003"]["expected_role"], "extra")

    def test_duplicate_and_fragment_descriptions_are_preserved(self):
        payload = {
            "r": "uncertain",
            "p": [
                {"n": 1, "d": "A hooded figure holds a phone."},
                {"n": 2, "d": "A hooded figure holds a phone."},
                {"n": 3, "d": "The."},
            ],
            "e": [],
        }
        result = parse_page_response(json.dumps(payload), expected_panels=[1, 2, 3])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["quality_errors"], {})
        self.assertEqual([p["description"] for p in result["panels"]], [p["d"] for p in payload["p"]])

    def test_actions_without_an_actor_are_preserved_as_missing_detection_diagnostics(self):
        labels = [{"label": "C1", "detection_id": "det", "panel_number": 1}]
        payload = {
            "r": "uncertain",
            "p": [{"n": 1, "d": "A person raises one hand.", "a": [
                {"s": None, "t": None, "x": "raises one hand", "c": 100},
                {"s": "C1", "t": None, "x": "says the title aloud", "c": 100},
            ]}],
            "e": [],
            "c": {"C1": "unclear labeled figure"},
        }
        result = parse_page_response(
            json.dumps(payload), expected_panels=[1], character_labels=labels,
        )
        self.assertEqual(len(result["panels"][0]["actions"]), 2)
        ungrounded = result["panels"][0]["actions"][0]
        self.assertEqual(ungrounded["status"], "ungrounded")
        self.assertEqual(ungrounded["grounding_status"], "actor_detection_missing")
        self.assertEqual(result["characters"][0]["visual_description"], "unclear labeled figure")
        self.assertEqual(result["rejected_records"], [])
        self.assertTrue(any("no actor detection" in error for error in result["grounding_errors"]))

    def test_role_parser_and_high_confidence_metadata_are_conservative(self):
        self.assertEqual(_parse_role("The answer is EXTRA."), "uncertain")
        self.assertEqual(_parse_role("extra"), "extra")
        self.assertEqual(_parse_role("not sure"), "uncertain")
        for layout in ("title_page", "title", "cover"):
            self.assertIsNone(_deterministic_page_role({"layout": layout}))
        self.assertEqual(_deterministic_page_role({"layout": "credits"})["page_role"], "extra")
        self.assertEqual(_deterministic_page_role({
            "layout": "title_page", "page_role_source": "review", "page_role": "story",
        })["page_role"], "story")
        self.assertIsNone(_deterministic_page_role({"layout": "page"}))

    def test_model_roles_are_not_rewritten_based_on_page_position(self):
        decision = {"page_role": "extra", "raw_response": "extra"}
        middle = _resolve_model_page_role(decision, {"page_order": 20}, 24)
        opening = _resolve_model_page_role(decision, {"page_order": 2}, 24)
        self.assertEqual(middle["page_role"], "extra")
        self.assertEqual(opening["page_role"], "extra")
        self.assertEqual(opening["source"], "description_classifier")

    def test_visible_shouting_is_not_mistaken_for_transcribed_dialogue(self):
        payload = {
            "r": "uncertain",
            "p": [{"n": 3, "d": (
                "A light-haired girl leans forward with her mouth open in a shout."
            )}],
            "e": [],
        }
        result = parse_page_response(json.dumps(payload), expected_panels=[3])
        self.assertEqual(result["quality_errors"], {})
        self.assertEqual(result["panels"][0]["status"], "usable")

    def test_character_descriptors_preserve_delimiters_and_tags_inside_strings(self):
        labels = [{"label": "C1", "detection_id": "det", "panel_number": 1}]
        base = {"r": "uncertain", "p": [{"n": 1, "d": "A person stands.", "a": []}], "e": []}
        cleaned = dict(base, c={"C1": "}dark hair, white coat, narrow eyes"})
        result = parse_page_response(
            json.dumps(cleaned), expected_panels=[1], character_labels=labels,
        )
        self.assertEqual(result["characters"][0]["visual_description"], "}dark hair, white coat, narrow eyes")

        tagged = dict(base, c={"C1": "}<box>(10, 10, 20, 20)</box>"})
        result = parse_page_response(
            json.dumps(tagged), expected_panels=[1], character_labels=labels,
        )
        self.assertEqual(result["characters"][0]["visual_description"], "}<box>(10, 10, 20, 20)</box>")

    def test_action_prose_is_not_rejected_based_on_vocabulary(self):
        labels = [{"label": "C1", "detection_id": "det", "panel_number": 1}]
        payload = {
            "r": "uncertain",
            "p": [{"n": 1, "d": "A dark-haired person looks tense.", "a": [
                {"s": "C1", "t": None, "x": "dark hair and sweat on face", "c": 100},
            ]}],
            "e": [], "c": {"C1": "dark hair and white coat"},
        }
        result = parse_page_response(
            json.dumps(payload), expected_panels=[1], character_labels=labels,
        )
        self.assertEqual(result["panels"][0]["actions"][0]["text"], "dark hair and sweat on face")
        self.assertEqual(result["rejected_records"], [])

    def test_panel_labels_are_preserved_and_confidence_is_not_evidence(self):
        labels = [{"label": "C1", "detection_id": "one", "panel_number": 1}]
        payload = {
            "r": "uncertain",
            "p": [{"n": 1, "d": "Panel 1 shows a horned girl near a window.", "a": [
                {"s": "C1", "t": "C1", "x": "attacks the nearby figure", "c": 100},
                {"s": "C1", "t": None, "x": "points toward the door", "c": 50},
            ]}],
            "e": [],
            "c": {"C1": "white-haired horned girl"},
        }
        result = parse_page_response(
            json.dumps(payload), expected_panels=[1], character_labels=labels,
        )
        self.assertEqual(result["panels"][0]["description"], payload["p"][0]["d"])
        self.assertEqual(len(result["panels"][0]["actions"]), 1)
        self.assertEqual(result["panels"][0]["actions"][0]["text"], "points toward the door")
        self.assertNotIn("confidence_percent", result["panels"][0]["actions"][0])
        reasons = {reason for row in result["rejected_records"] for reason in row["reasons"]}
        self.assertIn("self_target", reasons)

    def test_open_action_vocabulary_accepts_concrete_verb_phrases(self):
        labels = [{"label": "C1", "detection_id": "one", "panel_number": 1}]
        payload = {
            "r": "uncertain",
            "p": [{"n": 1, "d": "A girl clutches her coat.", "a": [
                {"s": "C1", "t": None, "x": "clutches her coat"},
            ]}],
            "e": [], "c": {"C1": "dark-haired girl in coat"},
        }
        result = parse_page_response(
            json.dumps(payload), expected_panels=[1], character_labels=labels,
        )
        self.assertEqual(result["panels"][0]["actions"][0]["text"], "clutches her coat")

    def test_incomplete_character_clause_is_preserved(self):
        labels = [{"label": "C1", "detection_id": "one", "panel_number": 1}]
        payload = {
            "r": "uncertain", "p": [{"n": 1, "d": "A girl stands.", "a": []}],
            "e": [], "c": {"C1": "long hair and her eyes are"},
        }
        result = parse_page_response(
            json.dumps(payload), expected_panels=[1], character_labels=labels,
        )
        self.assertEqual(result["characters"][0]["visual_description"], "long hair and her eyes are")

    def test_focused_character_pass_preserves_detection_ids(self):
        labels = [{"label": "C1", "detection_id": "det_one", "panel_number": 2}]
        processor, model = MagicMock(), MagicMock()
        input_ids = MagicMock()
        input_ids.shape = (1, 20)
        processor.apply_chat_template.return_value = {"input_ids": input_ids}
        processor.decode.return_value = json.dumps({"C1": {"d": "white-haired horned girl", "r": "live_character"}})
        model.generate.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()):
            result = describe_character_pages([(Path("page.jpg"), labels)], processor, model)[0]
        self.assertEqual(result["characters"][0]["detection_id"], "det_one")
        self.assertEqual(result["characters"][0]["visual_description"], "white-haired horned girl")

    def test_character_crop_retry_does_not_claim_an_annotation_is_present(self):
        labels = [{"label": "C1", "detection_id": "det_one", "panel_number": 2}]
        processor, model = MagicMock(), MagicMock()
        input_ids = MagicMock()
        input_ids.shape = (1, 20)
        processor.apply_chat_template.return_value = {"input_ids": input_ids}
        processor.decode.return_value = json.dumps({"C1": {"d": "white hair and small horns", "r": "live_character"}})
        model.generate.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()):
            describe_character_pages(
                [(Path("crop.jpg"), labels, True)], processor, model,
            )
        prompt = processor.apply_chat_template.call_args.args[0][0][0]["content"][1]["text"]
        self.assertIn("tight crop", prompt)
        self.assertNotIn("cyan-boxed", prompt)

    def test_description_only_pass_has_one_small_task(self):
        processor, model = MagicMock(), MagicMock()
        input_ids = MagicMock()
        input_ids.shape = (1, 20)
        processor.apply_chat_template.return_value = {"input_ids": input_ids}
        processor.decode.return_value = json.dumps({
            "p": [{"n": 1, "d": "A horned girl stands near a window."}],
        })
        generated = MagicMock()
        generated.shape = (20,)
        model.generate.return_value.__getitem__.return_value = generated
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()):
            result = describe_focused_pages(
                [(Path("page.jpg"), [1], [])], processor, model, task="description",
            )[0]
        prompt = processor.apply_chat_template.call_args.args[0][0][0]["content"][1]["text"]
        self.assertNotIn("transformation", prompt.lower())
        self.assertEqual(result["payload"]["p"][0]["n"], 1)

    def test_observation_pass_combines_only_descriptions_and_actions(self):
        processor, model = MagicMock(), MagicMock()
        input_ids = MagicMock()
        input_ids.shape = (1, 20)
        processor.apply_chat_template.return_value = {"input_ids": input_ids}
        processor.decode.return_value = json.dumps({
            "p": [{"n": 1, "d": "A horned girl raises one hand.", "a": [
                {"s": "C1", "t": None, "x": "raises one hand"},
            ]}],
        })
        generated = MagicMock()
        generated.shape = (20,)
        model.generate.return_value.__getitem__.return_value = generated
        labels = [{"label": "C1", "detection_id": "det", "panel_number": 1}]
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()):
            result = describe_focused_pages(
                [(Path("page.jpg"), [1], labels)], processor, model, task="observation",
            )[0]
        prompt = processor.apply_chat_template.call_args.args[0][0][0]["content"][1]["text"]
        self.assertNotIn("transformation", prompt.lower())
        self.assertNotIn("appearance phrase", prompt.lower())
        self.assertEqual(result["payload"]["p"][0]["a"][0]["s"], "C1")

    def test_role_classifier_builds_one_conversation_and_output_per_page(self):
        processor, model = MagicMock(), MagicMock()
        input_ids = MagicMock()
        input_ids.shape = (2, 20)
        processor.apply_chat_template.return_value = {"input_ids": input_ids}
        processor.decode.side_effect = ["story", "extra"]
        output = MagicMock()
        model.generate.return_value = output

        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data):
            result = classify_pages(
                [Path("first.jpg"), Path("second.jpg")], processor, model,
            )

        conversations = processor.apply_chat_template.call_args.args[0]
        self.assertEqual(len(conversations), 2)
        self.assertTrue(all(isinstance(conversation, list) for conversation in conversations))
        self.assertEqual([item["page_role"] for item in result], ["story", "extra"])
        self.assertEqual(output.__getitem__.call_count, 2)


class TextMaskTests(unittest.TestCase):
    def test_mask_is_temporary_and_manifest_carries_detector_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.jpg"
            source_image = Image.new("RGB", (80, 80), "white")
            ImageDraw.Draw(source_image).line((40, 20, 40, 60), fill="black", width=3)
            source_image.save(source)
            masked = mask_text_regions(source, [[20, 20, 60, 60]], root / "masked.jpg")
            with Image.open(source) as original, Image.open(masked) as prepared:
                self.assertEqual(original.getpixel((5, 5)), (255, 255, 255))
                self.assertGreater(sum(prepared.getpixel((40, 40))), 300)
                self.assertLess(sum(original.getpixel((40, 40))), 10)

            detection = PageDetection(
                source, [[0, 0, 80, 80]], [.9], text_boxes=[[20, 20, 60, 60]],
            )
            manifest = build_manifest([detection])
            self.assertEqual(manifest["pages"][0]["text_regions"], [
                {"box": [20, 20, 60, 60]},
            ])


if __name__ == "__main__":
    unittest.main()
