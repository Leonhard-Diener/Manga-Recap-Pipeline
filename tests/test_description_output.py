"""CPU-only regression tests; no models, network calls, or production outputs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import test_description as runner
from description.describer import (
    PAGE_PROMPT,
    _page_messages,
    describe_chapter,
    describe_manga,
    describe_page,
    describe_pages,
)
from description.response import EVENT_TYPES, parse_page_response
from description.schema import page_response_schema


def response_payload():
    return {
        "panels": [
            {"panel_number": 1, "description": "A horned figure is covered in scales."},
            {"panel_number": 2, "description": "A dragon stands in the clearing."},
        ],
        "events": [{
            "type": "transformation",
            "source_panel": 1,
            "target_panel": 2,
            "source_description": "horned figure",
            "target_description": "winged dragon",
            "evidence": "Scales and wings emerge from the figure.",
            "evidence_type": "visible_change",
        }],
    }


class ResponseTests(unittest.TestCase):
    def test_enforced_schema_requires_all_compact_sections(self):
        schema = page_response_schema([1, 2])
        self.assertEqual(schema["required"], ["r", "p", "e"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["p"]["minItems"], 2)
        self.assertEqual(schema["properties"]["p"]["maxItems"], 2)
        number = schema["properties"]["p"]["items"]["properties"]["n"]
        self.assertEqual(number["enum"], [1, 2])

    def test_prompt_has_no_fictional_values_for_model_to_copy(self):
        self.assertNotIn("horned woman", PAGE_PROMPT.lower())
        self.assertNotIn('"Emi"', PAGE_PROMPT)
        messages = _page_messages(Path("page.jpg"), [1])
        prompt = messages[0]["content"][1]["text"]
        self.assertIn("exactly once and no other numbers: [1]", prompt)

    def test_event_keeps_panel_evidence_and_is_always_pending(self):
        payload = response_payload()
        payload["events"][0]["status"] = "confirmed"
        payload["events"][0]["identity"] = "invented_name"
        result = parse_page_response(json.dumps(payload))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["events"][0]["target_panel"], 2)
        self.assertEqual(result["events"][0]["status"], "pending_review")
        self.assertNotIn("identity", result["events"][0])
        self.assertNotIn("Scales and wings", result["panels"][0]["description"])

    def test_no_events_is_an_explicit_success(self):
        payload = response_payload()
        payload["events"] = []
        result = parse_page_response(json.dumps(payload))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["events"], [])

    def test_missing_events_is_not_no_events(self):
        payload = response_payload()
        del payload["events"]
        result = parse_page_response(json.dumps(payload))
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(len(result["panels"]), 2)

    def test_truncated_output_is_not_a_negative_finding(self):
        for text in (json.dumps(response_payload()), '{"panels": ['):
            with self.subTest(text=text):
                result = parse_page_response(text, truncated=True)
                self.assertEqual(result["status"], "truncated")
                self.assertEqual(result["events"], [])

    def test_malformed_json_or_wrong_root(self):
        for text in ("PANEL 1: a dragon", "{}", "null", "[]", '{"panels": ['):
            with self.subTest(text=text):
                result = parse_page_response(text)
                self.assertEqual(result["status"], "invalid_output")
                self.assertEqual(result["events"], [])

    def test_fenced_json_is_supported(self):
        result = parse_page_response("```json\n" + json.dumps(response_payload()) + "\n```")
        self.assertEqual(result["status"], "complete")

    def test_compact_model_schema_expands_to_public_schema(self):
        payload = {
            "r": "story",
            "p": [{"n": 1, "d": "A horned figure changes."},
                  {"n": 2, "d": "A dragon stands nearby."}],
            "e": [{"t": "transformation", "sp": 1, "tp": 2,
                   "sd": "horned figure", "td": "dragon",
                   "x": "wings visibly emerge", "xt": "visible_change"}],
        }
        result = parse_page_response(json.dumps(payload))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["page_role"], "story")
        self.assertEqual(result["panels"][0]["panel_number"], 1)
        self.assertEqual(result["events"][0]["source_description"], "horned figure")

    def test_wrong_references_and_event_fields_fail_closed(self):
        for key, value in (
            ("source_panel", 99), ("source_panel", True), ("target_panel", "2"),
            ("type", "same_character"), ("type", []), ("evidence", ""),
            ("source_description", None), ("evidence_type", "guess"),
        ):
            with self.subTest(key=key, value=value):
                payload = response_payload()
                payload["events"][0][key] = value
                result = parse_page_response(json.dumps(payload))
                self.assertEqual(result["status"], "invalid_output")
                self.assertEqual(result["events"], [])

    def test_duplicate_panel_mapping_invalidates_events(self):
        payload = response_payload()
        payload["panels"].append(copy.deepcopy(payload["panels"][0]))
        result = parse_page_response(json.dumps(payload))
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["events"], [])

    def test_same_panel_event_and_other_event_types(self):
        for event_type in EVENT_TYPES:
            with self.subTest(event_type=event_type):
                payload = response_payload()
                payload["events"][0].update(type=event_type, source_panel=2)
                result = parse_page_response(json.dumps(payload))
                self.assertEqual(result["events"][0]["type"], event_type)

    def test_descriptions_remove_labels_quotes_and_dangling_clauses(self):
        payload = response_payload()
        payload["panels"] = [{
            "panel_number": 1,
            "description": (
                "Panel 1: A winged girl flies through the air, shouting "
                "'YOU AGAIN? PERSISTENT.'"
            ),
        }]
        payload["events"] = []
        result = parse_page_response(json.dumps(payload), expected_panels=[1])
        self.assertEqual(
            result["panels"][0]["description"],
            "A winged girl flies through the air.",
        )


class ReturnAndExportTests(unittest.TestCase):
    def test_page_uses_one_generation_for_both_results(self):
        processor, model = MagicMock(), MagicMock()
        input_ids = MagicMock()
        input_ids.shape = (1, 50)
        processor.apply_chat_template.return_value = {"input_ids": input_ids}
        processor.decode.return_value = json.dumps(response_payload())
        generated = MagicMock()
        generated.shape = (200,)
        model.generate.return_value.__getitem__.return_value = generated
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data):
            result = describe_page(Path("page.jpg"), processor, model)
        model.generate.assert_called_once()
        self.assertIn("prefix_allowed_tokens_fn", model.generate.call_args.kwargs)
        processor.apply_chat_template.assert_called_once()
        self.assertEqual(len(result["panels"]), 2)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["status"], "complete")

    def test_two_pages_share_one_generation_without_lowering_token_limit(self):
        processor, model = MagicMock(), MagicMock()
        input_ids = MagicMock()
        input_ids.shape = (2, 50)
        processor.apply_chat_template.return_value = {"input_ids": input_ids}
        processor.decode.side_effect = [json.dumps(response_payload()),
                                        json.dumps(response_payload())]
        generated = MagicMock()
        generated.shape = (200,)
        output = MagicMock()
        output.__getitem__.side_effect = [generated, generated]
        model.generate.return_value = output

        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.build_prefix_allowed_tokens_fn", return_value=MagicMock()), \
                patch("description.describer.move_to_device", side_effect=lambda data, device: data):
            results = describe_pages(
                [(Path("first.jpg"), [1, 2]), (Path("second.jpg"), [1, 2])],
                processor,
                model,
            )

        model.generate.assert_called_once()
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["batch_size"] == 2 for result in results))
        self.assertTrue(all(result["generation_attempts"] == 1 for result in results))
        self.assertTrue(all(result["token_limit"] == 1024 for result in results))
        self.assertTrue(all(result["status"] == "complete" for result in results))

    def test_chapter_collects_both_outputs_from_one_batched_call(self):
        page = parse_page_response(json.dumps(response_payload()))
        page.update(generation_time=1.5, generated_tokens=200, raw_response="raw")
        failure = parse_page_response("broken")
        failure.update(generation_time=1.0, generated_tokens=10, raw_response="broken")
        pages = [Path("first.jpg"), Path("second.jpg")]
        with patch("description.describer.list_images", return_value=pages), \
                patch("description.describer.describe_pages", return_value=[page, failure]) as infer:
            result = describe_chapter(Path("chapter_001"), object(), object())
        infer.assert_called_once()
        self.assertEqual(result["descriptions"]["first.jpg"]["panels"], page["panels"])
        self.assertEqual(len(result["transformations"]["first.jpg"]["events"]), 1)
        failed = result["transformations"]["second.jpg"]
        self.assertEqual(failed["extraction_status"], "invalid_output")
        self.assertEqual(failed["raw_response"], "broken")

    def test_batch_oom_switches_remaining_chapter_to_single_pages(self):
        page = parse_page_response(json.dumps(response_payload()))
        page.update(generation_time=1.0, generated_tokens=200, raw_response="raw")
        pages = [Path("first.jpg"), Path("second.jpg")]
        with patch.dict("sys.modules", {"torch": MagicMock()}), \
                patch("description.describer.list_images", return_value=pages), \
                patch("description.describer.describe_pages", side_effect=[
                    RuntimeError("CUDA out of memory"), [page], [page],
                ]) as infer:
            result = describe_chapter(Path("chapter_001"), object(), object())

        self.assertEqual([len(call.args[0]) for call in infer.call_args_list], [2, 1, 1])
        self.assertEqual(set(result["descriptions"]), {"first.jpg", "second.jpg"})

    def test_opening_page_isolated_from_later_batches(self):
        page = parse_page_response(json.dumps(response_payload()))
        page.update(generation_time=1.0, generated_tokens=200, raw_response="raw")
        pages = [Path(f"{number:05}.jpg") for number in range(1, 5)]

        def infer(requests, processor, model):
            return [page] * len(requests)

        with patch("description.describer.list_images", return_value=pages), \
                patch("description.describer.describe_pages", side_effect=infer) as batched:
            describe_chapter(Path("chapter_001"), object(), object())

        self.assertEqual([len(call.args[0]) for call in batched.call_args_list], [1, 2, 1])

    def test_manga_loads_once_returns_both_and_writes_nothing(self):
        model = MagicMock()
        model.parameters.return_value = iter([MagicMock(dtype="float16")])
        chapter_result = {
            "descriptions": {"00001.jpg": {"panels": []}},
            "transformations": {"00001.jpg": {"events": [], "extraction_status": "complete"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.dict("sys.modules", {"torch": MagicMock()}), \
                    patch("description.describer.load_describer", return_value=(object(), model)) as load, \
                    patch("description.describer.describe_chapter", return_value=chapter_result):
                result = describe_manga(output, [Path("chapter_001"), Path("chapter_002")])
            load.assert_called_once()
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(set(result["descriptions"]), {"chapter_001", "chapter_002"})
            self.assertEqual(set(result["transformations"]["chapters"]), {"chapter_001", "chapter_002"})
            self.assertNotIn("names", result)

    def test_test_runner_owns_combined_json_export_and_returns_data(self):
        expected = {
            "descriptions": {"chapter_001": {"00001.jpg": {"panels": []}}},
            "transformations": {"schema_version": 1, "scope": "within_page", "chapters": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.object(runner.config, "OUTPUT_DIR", output), \
                    patch.object(runner, "find_manga_dir", return_value=Path("Manga")), \
                    patch.object(runner, "find_chapter_dirs", return_value=[Path("chapter_001")]), \
                    patch.object(runner, "describe_manga", return_value=expected) as describe, \
                    patch.object(runner.sys, "argv", ["test_description.py", "Manga"]):
                result = runner.main()
            describe.assert_called_once_with(output / "Manga", [Path("chapter_001")])
            self.assertIs(result, expected)
            self.assertEqual(set(path.name for path in output.iterdir()), {"Manga_descriptions.json"})
            actual = json.loads((output / "Manga_descriptions.json").read_text(encoding="utf-8"))
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
