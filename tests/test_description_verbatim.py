"""Verbatim model output and bounded, format-only repair regressions."""

import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from description.describer import (
    _mark_remote_provenance, _remote_result, _parse_text_proposals,
    describe_focused_pages, describe_character_pages, describe_chapter,
)
from description.format import FormatResponseError, parse_json
from description.response import parse_page_response
from description.schema import panel_description_schema
from story_entities.manifest import file_hash


class VerbatimTests(unittest.TestCase):
    def test_prose_survives_exactly_in_all_fields(self):
        description = '  Panel 1: A dark screen with white text says "RUN!".  '
        appearance = '}she has cyan box labels and long hair '
        action = '  Says "RUN!" while wearing a coat  '
        labels = [{"label": "C1", "detection_id": "d1", "panel_number": 1}]
        result = parse_page_response(json.dumps({
            "r": "story", "p": [{"n": 1, "d": description, "a": [
                {"s": "C1", "t": None, "x": action},
            ]}], "c": {"C1": {"d": appearance, "r": "live_character"}}, "e": [],
        }), expected_panels=[1], character_labels=labels)
        self.assertEqual(result["panels"][0]["description"], description)
        self.assertEqual(result["panels"][0]["actions"][0]["text"], action)
        self.assertEqual(result["characters"][0]["visual_description"], appearance)
        self.assertEqual(result["quality_errors"], {})

    def test_transcription_whitespace_and_repeated_lines_survive(self):
        source = {"source_hash": "h", "panels": [{"panel_number": 1, "panel_id": "p"}]}
        raw = '  NO!\n  NO!  '
        proposals, errors = _parse_text_proposals({
            "status": "complete", "payload": {"p": [{"n": 1, "t": [
                {"k": "dialogue", "x": raw, "s": None},
                {"k": "dialogue", "x": raw, "s": None},
            ]}]},
        }, source)
        self.assertEqual(errors, [])
        self.assertEqual([row["text"] for row in proposals], [raw, raw])

    def test_provenance_conversion_does_not_rewrite_model_strings(self):
        value = {"description": "qwen_is printed here", "raw_response": "qwen_raw",
                 "provenance": "qwen_visual_observation"}
        result = _mark_remote_provenance(value, "api")
        self.assertEqual(result["description"], value["description"])
        self.assertEqual(result["raw_response"], value["raw_response"])
        self.assertEqual(result["provenance"], "api_visual_observation")

    def test_provenance_conversion_repairs_a_different_remote_backend_label(self):
        value = {"provenance": "openrouter_vision_focused_character_observation"}
        self.assertEqual(
            _mark_remote_provenance(value, "google_vision")["provenance"],
            "google_vision_focused_character_observation",
        )

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self):
        for raw in ('{"p":1,"p":2}', '{"p":NaN}', '{"p":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_json(raw)

    def remote(self, responses):
        client = MagicMock()
        client.complete_image.side_effect = [{
            "content": raw, "finish_reason": "stop",
            "elapsed_seconds": .1, "completion_tokens": 10,
        } for raw in responses]
        return client

    def test_valid_remote_prose_never_triggers_a_second_call(self):
        raw = json.dumps({"p": [{"n": 1, "d": "The."}]})
        client = self.remote([raw])
        result = _remote_result(client, Path("page.jpg"), "prompt", panel_description_schema([1]), 128)
        self.assertEqual(result["payload"]["p"][0]["d"], "The.")
        self.assertEqual(result["retry_count"], 0)
        client.complete_image.assert_called_once()

    def test_remote_invalid_json_or_schema_gets_only_one_repair(self):
        valid = json.dumps({"p": [{"n": 1, "d": "The model's text."}]})
        for invalid in ('not JSON', '{"p":[]}', '{"p":[{"n":2,"d":"wrong ID"}]}'):
            with self.subTest(invalid=invalid):
                client = self.remote([invalid, valid])
                result = _remote_result(client, Path("page.jpg"), "prompt", panel_description_schema([1]), 128)
                self.assertEqual(client.complete_image.call_count, 2)
                self.assertEqual(result["prior_raw_responses"], [invalid])
                self.assertEqual(result["generated_tokens"], 20)
                self.assertEqual(result["retry_count"], 1)
                self.assertTrue(all(call.kwargs["max_tokens"] == 128 for call in client.complete_image.call_args_list))

    def test_remote_two_invalid_responses_fail_without_placeholder(self):
        client = self.remote(['broken', 'still broken'])
        with self.assertRaises(FormatResponseError) as caught:
            _remote_result(client, Path("page.jpg"), "prompt", panel_description_schema([1]), 128)
        self.assertEqual(client.complete_image.call_count, 2)
        self.assertEqual(caught.exception.responses, ['broken', 'still broken'])

    def test_remote_combined_response_stops_after_two_bad_panel_label_mappings(self):
        labels = [
            {"label": "C1", "detection_id": "d1", "panel_number": 1},
            {"label": "C2", "detection_id": "d2", "panel_number": 2},
        ]
        raw = json.dumps({
            "r": "story",
            "p": [
                {"n": 1, "d": "One figure stands.", "a": []},
                {"n": 2, "d": "Another figure runs.", "a": [
                    {"s": "C1", "t": None, "x": "runs"},
                ]},
            ],
            "c": {
                "C1": {"d": "short dark-haired figure", "r": "live_character"},
                "C2": {"d": "tall light-haired figure", "r": "live_character"},
            },
            "e": [],
            "x": {"p": [{"n": 1, "t": []}, {"n": 2, "t": []}]},
        })
        client = self.remote([raw, raw])
        client.is_remote_description_client = True
        client.store_page_bundle = MagicMock()

        with self.assertRaises(FormatResponseError) as caught:
            describe_focused_pages(
                [(Path("page.jpg"), [1, 2], labels, {"text_region_count": 0})],
                client, None, task="observation",
            )
        self.assertEqual(caught.exception.responses, [raw, raw])
        self.assertEqual(client.complete_image.call_count, 2)
        client.store_page_bundle.assert_not_called()

    def test_role_request_accepts_its_non_json_format(self):
        client = self.remote(['extra'])
        result = _remote_result(client, Path("page.jpg"), "role", None, 4)
        self.assertEqual(result["raw_response"], 'extra')
        client.complete_image.assert_called_once()

    def test_local_format_repair_is_bounded_and_uses_same_budget(self):
        processor, model = MagicMock(), MagicMock()
        ids = MagicMock()
        ids.shape = (1, 20)
        processor.apply_chat_template.return_value = {"input_ids": ids}
        valid = json.dumps({"p": [{"n": 1, "d": "The."}]})
        processor.decode.side_effect = ['bad JSON', valid]
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()):
            result = describe_focused_pages([(Path("page.jpg"), [1], [])], processor, model, task="description")[0]
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual(result["payload"]["p"][0]["d"], "The.")
        self.assertEqual(result["prior_raw_responses"], ['bad JSON'])
        self.assertEqual(model.generate.call_args_list[0].kwargs["max_new_tokens"], model.generate.call_args_list[1].kwargs["max_new_tokens"])

    def test_valid_local_character_fragment_is_used_without_retry(self):
        processor, model = MagicMock(), MagicMock()
        ids = MagicMock()
        ids.shape = (1, 20)
        processor.apply_chat_template.return_value = {"input_ids": ids}
        raw = '  long hair and her eyes are  '
        processor.decode.return_value = json.dumps({"C1": {"d": raw, "r": "uncertain"}})
        labels = [{"label": "C1", "detection_id": "d1", "panel_number": 1}]
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()):
            result = describe_character_pages([(Path("page.jpg"), labels)], processor, model)[0]
        model.generate.assert_called_once()
        self.assertEqual(result["characters"][0]["visual_description"], raw)

    def test_api_uses_existing_interior_role_without_extra_requests(self):
        class Backend:
            is_remote_description_client = True

            def __init__(self):
                self.calls = 0
                self.bundles = {}

            def clear_page_bundles(self):
                self.bundles.clear()

            def store_page_bundle(self, path, bundle):
                self.bundles[Path(path).name] = bundle

            def page_bundle(self, path):
                return self.bundles[Path(path).name]

            def complete_image(self, **kwargs):
                self.calls += 1
                role = "extra" if Path(kwargs["image_path"]).name == "004.jpg" else "story"
                return {"content": json.dumps({
                    "r": role, "p": [{"n": 1, "d": "The.", "a": []}],
                    "c": {}, "e": [], "x": {"p": [{"n": 1, "t": []}]},
                }), "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as directory:
            chapter = Path(directory)
            preview_dir = chapter / "page_previews"
            preview_dir.mkdir()
            pages = []
            for index in range(8):
                name = f"{index:03}.jpg"
                source = chapter / name
                source.write_bytes(b"fake image")
                preview = preview_dir / name
                preview.write_bytes(b"fake image")
                pages.append({
                    "page": str(source), "page_order": index,
                    "source_hash": file_hash(source), "preview_hash": file_hash(preview),
                    "panels": [{"panel_number": 1, "panel_id": f"p{index}"}],
                })
            (chapter / "panel_manifest.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")
            backend = Backend()
            result = describe_chapter(chapter, backend, backend, filter_non_story=True)
        self.assertEqual(backend.calls, 8)
        self.assertEqual(result["page_filter"]["004.jpg"]["page_role"], "extra")
        self.assertFalse(result["page_filter"]["004.jpg"]["included_in_story"])
        self.assertNotIn("004.jpg", result["descriptions"])

    def test_local_two_invalid_formats_stop_after_two_attempts(self):
        processor, model = MagicMock(), MagicMock()
        ids = MagicMock()
        ids.shape = (1, 20)
        processor.apply_chat_template.return_value = {"input_ids": ids}
        processor.decode.side_effect = ['broken', 'still broken']
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()):
            with self.assertRaises(FormatResponseError) as caught:
                describe_focused_pages([(Path("page.jpg"), [1], [])], processor, model, task="description")
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual(caught.exception.responses, ['broken', 'still broken'])


if __name__ == "__main__":
    unittest.main()
