"""CPU-only regression tests; no models, network calls, or production outputs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import test_description as runner
from description.describer import describe_chapter, describe_manga, describe_page
from description.response import EVENT_TYPES, parse_page_response


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
                patch("description.describer.move_to_device", side_effect=lambda data, device: data):
            result = describe_page(Path("page.jpg"), processor, model)
        model.generate.assert_called_once()
        processor.apply_chat_template.assert_called_once()
        self.assertEqual(len(result["panels"]), 2)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["status"], "complete")

    def test_chapter_collects_both_outputs_from_one_call_per_page(self):
        page = parse_page_response(json.dumps(response_payload()))
        page.update(generation_time=1.5, generated_tokens=200, raw_response="raw")
        failure = parse_page_response("broken")
        failure.update(generation_time=1.0, generated_tokens=10, raw_response="broken")
        pages = [Path("first.jpg"), Path("second.jpg")]
        with patch("description.describer.list_images", return_value=pages), \
                patch("description.describer.describe_page", side_effect=[page, failure]) as infer:
            result = describe_chapter(Path("chapter_001"), object(), object())
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(result["descriptions"]["first.jpg"]["panels"], page["panels"])
        self.assertEqual(len(result["transformations"]["first.jpg"]["events"]), 1)
        failed = result["transformations"]["second.jpg"]
        self.assertEqual(failed["extraction_status"], "invalid_output")
        self.assertEqual(failed["raw_response"], "broken")

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

    def test_test_runner_owns_both_json_exports_and_returns_data(self):
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
            self.assertEqual(set(path.name for path in output.iterdir()), {
                "Manga_descriptions.json", "Manga_transformations.json",
            })
            for key in ("descriptions", "transformations"):
                actual = json.loads((output / f"Manga_{key}.json").read_text(encoding="utf-8"))
                self.assertEqual(actual, expected[key])


if __name__ == "__main__":
    unittest.main()
