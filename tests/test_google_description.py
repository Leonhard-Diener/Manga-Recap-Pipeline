from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
import httpx

from description.google import (
    GoogleDescriptionClient, GooglePermanentError, normalize_google_model_id, _google_panel_slots,
    _google_response_schema, _restore_panel_arrays,
)
from description.describer import _openrouter_bundle_schema, _remote_result
from description.format import validate_format
from test_description_google import _parser, main


class GoogleDescriptionClientTests(unittest.TestCase):
    def test_runner_reports_incomplete_export_instead_of_success(self):
        result = {"pending_pages": {"chapter_001": ["00029.jpg"]}}
        with tempfile.TemporaryDirectory() as directory:
            manga = Path(directory) / "Example"
            with patch("test_description_google._load_env_file"), \
                    patch("test_description_google.os.getenv", return_value="test-key"), \
                    patch("test_description_google.find_manga_dir", return_value=manga), \
                    patch("test_description_google.find_chapter_dirs", return_value=[]), \
                    patch("test_description_google.GoogleDescriptionClient") as client_class, \
                    patch("test_description_google.describe_manga", return_value=result), \
                    patch("test_description_google.write_json"), \
                    patch("builtins.print") as print_message:
                returned = main(["Example", "--output", str(Path(directory) / "description.json")])
        self.assertEqual(returned, result)
        self.assertEqual(client_class.call_args.kwargs["response_dir"],
                         config.OUTPUT_DIR / "Example" / "google_responses")
        messages = "\n".join(str(call.args[0]) for call in print_message.call_args_list)
        self.assertIn("Run incomplete", messages)
        self.assertIn("00029.jpg", messages)
        self.assertNotIn("Saved descriptions and character events", messages)

    def test_panel_slots_require_every_panel_and_scope_actor_target_and_speaker(self):
        labels = [{"label": "C1", "panel_number": 1}, {"label": "C2", "panel_number": 2}]
        canonical = _openrouter_bundle_schema([1, 2, 3], labels)
        wire = _google_panel_slots(canonical, labels)
        sent = _google_response_schema(wire)
        self.assertEqual(canonical["properties"]["p"]["type"], "array")
        self.assertEqual(sent["properties"]["p"]["required"], ["panel_1", "panel_2", "panel_3"])
        self.assertEqual(sent["properties"]["x"]["properties"]["p"]["required"],
                         ["panel_1", "panel_2", "panel_3"])
        for field, array in (("p", "a"), ("x", "t")):
            panels = sent["properties"]["p"] if field == "p" else sent["properties"]["x"]["properties"]["p"]
            for number, allowed in ((1, ["C1"]), (2, ["C2"]), (3, [])):
                item = panels["properties"][f"panel_{number}"]
                self.assertNotIn("n", item["properties"])
                endpoint = item["properties"][array]["items"]["properties"]["s"]
                if allowed:
                    self.assertEqual(endpoint["anyOf"][0]["enum"], allowed)
                    validate_format(allowed[0], endpoint)
                    with self.assertRaises(ValueError):
                        validate_format("C99", endpoint)
                else:
                    self.assertEqual(endpoint, {"type": "null"})
                validate_format(None, endpoint)
        with self.assertRaisesRegex(ValueError, "missing required field"):
            validate_format({"panel_1": {"d": "A figure stands.", "a": []}}, wire["properties"]["p"])

    def test_native_panel_slots_round_trip_without_changing_observations(self):
        labels = [{"label": "C1", "panel_number": 1}]
        schema = _openrouter_bundle_schema([1], labels)
        wire_value = {
            "r": "story", "p": {"panel_1": {"d": "  A figure raises a hand.  ", "a": [
                {"s": "C1", "t": None, "x": "raises a hand"}]}},
            "c": {"C1": {"d": "dark-haired figure", "r": "live_character"}}, "e": [],
            "x": {"p": {"panel_1": {"t": [{"k": "dialogue", "x": " Wait! ", "s": "C1"}]}}},
        }
        response = Mock()
        response.json.return_value = {"candidates": [{"content": {"parts": [
            {"text": json.dumps(wire_value)}]}, "finishReason": "STOP"}]}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"test image")
            client = GoogleDescriptionClient(api_key="secret", max_retries=0)
            client._prepared = True
            with patch("description.google.httpx.post", return_value=response) as post:
                result = _remote_result(client, image, "Analyze", schema, 1000, character_labels=labels)
        self.assertEqual(result["payload"], _restore_panel_arrays(wire_value))
        self.assertEqual(result["payload"]["p"][0]["d"], "  A figure raises a hand.  ")
        self.assertEqual(result["payload"]["x"]["p"][0]["t"][0]["x"], " Wait! ")
        self.assertIn("Transport format", post.call_args.kwargs["json"]["contents"][0]["parts"][1]["text"])
        post.assert_called_once()

    def test_missing_wire_slot_is_not_manufactured(self):
        labels = [{"label": "C1", "panel_number": 1}]
        schema = _openrouter_bundle_schema([1], labels)
        wire_schema = _google_panel_slots(schema, labels)
        with self.assertRaises(ValueError):
            validate_format({"r": "story", "p": {}, "c": {}, "e": [], "x": {"p": {}}}, wire_schema)
        self.assertEqual(_restore_panel_arrays({"p": {}}), {"p": []})

    def test_model_and_runner_defaults(self):
        self.assertEqual(normalize_google_model_id("models/gemini-3.8-flash"), "gemini-3.8-flash")
        self.assertEqual(_parser().parse_args([]).model, "gemini-3.5-flash-lite")
        self.assertEqual(_parser().parse_args([]).media_resolution, "high")
        self.assertIsNone(_parser().parse_args([]).max_retries)
        self.assertEqual(_parser().parse_args(["--max-retries", "0"]).max_retries, 0)

    def test_unlimited_503_waits_past_old_limit_and_preserves_each_reply(self):
        request = httpx.Request("POST", "https://example.test")
        busy = httpx.Response(503, request=request, headers={"Retry-After": "90"},
                              json={"error": {"code": 503, "message": "Busy"}})
        ready = httpx.Response(200, request=request, json={"candidates": [{
            "content": {"parts": [{"text": '{"p":[]}'}]}, "finishReason": "STOP"}]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "page.jpg"
            image.write_bytes(b"image")
            client = GoogleDescriptionClient(api_key="secret", response_dir=root / "archives")
            client._prepared = True
            with patch("description.google.httpx.post", side_effect=[busy] * 5 + [ready]) as post, \
                    patch("description.google.time.sleep") as sleep:
                result = client.complete_image(image_path=image, prompt="Analyze", schema=None, max_tokens=100)
            self.assertEqual(len(list((root / "archives").glob("*.json"))), 6)
        self.assertEqual(result["content"], '{"p":[]}')
        self.assertEqual(post.call_count, 6)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [90, 90, 120, 240, 300])

    def test_prepare_also_waits_for_transient_service_failure(self):
        request = httpx.Request("GET", "https://example.test")
        busy = httpx.Response(503, request=request)
        ready = httpx.Response(200, request=request, json={"supportedGenerationMethods": ["generateContent"]})
        client = GoogleDescriptionClient(api_key="secret", max_retries=None)
        with patch("description.google.httpx.get", side_effect=[busy, ready]) as get, \
                patch("description.google.time.sleep") as sleep:
            client.prepare()
        self.assertTrue(client._prepared)
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(45)

    def test_retry_after_date_and_non_json_503_are_supported(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime
        request = httpx.Request("POST", "https://example.test")
        after = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=180))
        busy = httpx.Response(503, request=request, headers={"Retry-After": after}, text="Busy")
        ready = httpx.Response(200, request=request, json={"candidates": [{
            "content": {"parts": [{"text": "story"}]}, "finishReason": "STOP"}]})
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"image")
            client = GoogleDescriptionClient(api_key="secret", max_retries=None)
            client._prepared = True
            with patch("description.google.httpx.post", side_effect=[busy, ready]), \
                    patch("description.google.time.sleep") as sleep:
                result = client.complete_image(image_path=image, prompt="Role", schema=None, max_tokens=100)
        self.assertEqual(result["content"], "story")
        self.assertGreater(sleep.call_args.args[0], 170)
        self.assertLessEqual(sleep.call_args.args[0], 180)

    def test_network_timeout_retries_but_access_and_quota_errors_do_not(self):
        request = httpx.Request("POST", "https://example.test")
        ready = httpx.Response(200, request=request, json={"candidates": [{
            "content": {"parts": [{"text": "story"}]}, "finishReason": "STOP"}]})
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"image")
            client = GoogleDescriptionClient(api_key="secret", max_retries=None)
            client._prepared = True
            with patch("description.google.httpx.post", side_effect=[httpx.ReadTimeout("timeout"), ready]), \
                    patch("description.google.time.sleep") as sleep:
                result = client.complete_image(image_path=image, prompt="Role", schema=None, max_tokens=100)
            self.assertEqual(result["content"], "story")
            sleep.assert_called_once_with(45)
            for code in (400, 401, 403, 404, 429):
                with self.subTest(status=code), \
                        patch("description.google.httpx.post", return_value=httpx.Response(
                            code, request=request, json={"error": {"code": code}})) as post, \
                        patch("description.google.time.sleep") as sleep, \
                        self.assertRaises(GooglePermanentError):
                    _remote_result(client, image, "Role", None, 100)
                post.assert_called_once()
                sleep.assert_not_called()

    def test_explicit_retry_limit_still_bounds_requests(self):
        busy = httpx.Response(503, request=httpx.Request("POST", "https://example.test"),
                              json={"error": {"code": 503}})
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"image")
            client = GoogleDescriptionClient(api_key="secret", max_retries=1)
            client._prepared = True
            with patch("description.google.httpx.post", return_value=busy) as post, \
                    patch("description.google.time.sleep") as sleep, \
                    self.assertRaises(RuntimeError):
                client.complete_image(image_path=image, prompt="Role", schema=None, max_tokens=100)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(45)

    def test_bad_retry_after_does_not_break_waiting(self):
        request = httpx.Request("POST", "https://example.test")
        client = GoogleDescriptionClient(api_key="secret", max_retries=None)
        for header, expected in (("bad-date", 45), ("nan", 45), ("inf", 45),
                                 ("-10", 45), ("1e300", 86400)):
            response = httpx.Response(503, request=request, headers={"Retry-After": header})
            error = httpx.HTTPStatusError("Busy", request=request, response=response)
            with self.subTest(header=header), patch("description.google.time.sleep") as sleep:
                self.assertTrue(client._wait_retry(error, 0))
            sleep.assert_called_once_with(expected)

    def test_native_request_uses_key_schema_image_and_resolution(self):
        metadata = Mock()
        metadata.raise_for_status.return_value = None
        metadata.json.return_value = {"supportedGenerationMethods": ["generateContent"]}
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": '{"p":[]}'}]},
                "finishReason": "MAX_TOKENS",
            }],
            "usageMetadata": {"candidatesTokenCount": 7},
        }
        schema = {
            "type": "object",
            "properties": {"p": {
                "type": "array", "maxItems": 3,
                "items": {"type": "integer", "enum": [1, 2, 3]},
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"fake jpeg")
            client = GoogleDescriptionClient(
                api_key="secret", model="gemini-3.8-flash",
                media_resolution="medium", max_retries=0,
            )
            with patch("description.google.httpx.get", return_value=metadata) as get, \
                    patch("description.google.httpx.post", return_value=response) as post:
                result = client.complete_image(
                    image_path=image, prompt="Describe", schema=schema, max_tokens=99,
                )

        get.assert_called_once()
        request = post.call_args.kwargs
        self.assertEqual(request["headers"]["x-goog-api-key"], "secret")
        self.assertTrue(request["json"]["contents"][0]["parts"][0]["inlineData"]["data"])
        generation = request["json"]["generationConfig"]
        self.assertEqual(generation["mediaResolution"], "MEDIA_RESOLUTION_MEDIUM")
        self.assertEqual(generation["thinkingConfig"], {"thinkingLevel": "HIGH", "includeThoughts": True})
        sent_schema = generation["responseJsonSchema"]
        self.assertEqual(sent_schema["type"], "object")
        self.assertIn("p", sent_schema["properties"])
        self.assertNotIn("maxItems", sent_schema["properties"]["p"])
        self.assertNotIn("enum", sent_schema["properties"]["p"]["items"])
        self.assertEqual(schema["properties"]["p"]["type"], "array")
        self.assertEqual(schema["properties"]["p"]["maxItems"], 3)
        self.assertEqual(generation["maxOutputTokens"], 99 + 4096)
        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(result["finish_reason"], "length")

    def test_google_backend_has_separate_checkpoint_identity(self):
        client = GoogleDescriptionClient(api_key="secret")
        self.assertEqual(client.provider_name, "google")
        self.assertEqual(client.checkpoint_name, "description_checkpoint_google.json")
        client._prepared = True
        self.assertEqual(client.cache_key[0], "google_generate_content_v2")
        self.assertEqual(client.cache_key[-2:], ("HIGH", True))
        self.assertIsNone(client.max_retries)

    def test_thought_summary_is_archived_separately_and_never_parsed_as_answer(self):
        data = {"candidates": [{"content": {"parts": [
            {"text": "An exposed summary, not JSON.", "thought": True},
            {"text": '{"p":[]}', "thoughtSignature": "opaque-signature"},
        ]}, "finishReason": "STOP"}],
            "usageMetadata": {"candidatesTokenCount": 7, "thoughtsTokenCount": 123}}
        response = Mock()
        response.json.return_value = data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "page.jpg"
            image.write_bytes(b"test image")
            client = GoogleDescriptionClient(api_key="secret", max_retries=0,
                                             response_dir=root / "archives")
            client._prepared = True
            with patch("description.google.httpx.post", return_value=response):
                result = _remote_result(client, image, "Analyze", None, 1000)
                again = _remote_result(client, image, "Analyze", None, 1000)
            self.assertEqual(result["payload"], {"p": []})
            self.assertEqual(result["raw_response"], '{"p":[]}')
            self.assertEqual(result["thought_summary"], "An exposed summary, not JSON.")
            self.assertEqual(result["thinking_tokens"], 123)
            self.assertEqual(result["token_limit"], 1000 + 4096)
            self.assertEqual(result["raw_api_response"], data)
            archives = list((root / "archives").glob("*.json"))
            self.assertEqual(len(archives), 2)
            self.assertNotEqual(result["response_archive"], again["response_archive"])
            saved = json.loads(Path(result["response_archive"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["final_answer"], '{"p":[]}')
            self.assertEqual(saved["thought_summary"], result["thought_summary"])
            self.assertEqual(saved["raw_api_response"], data)
            self.assertNotIn("secret", json.dumps(saved))

    def test_absent_summary_is_empty_without_inventing_one(self):
        response = Mock()
        response.json.return_value = {"candidates": [{"content": {"parts": [
            {"text": '{"p":[]}'}]}, "finishReason": "STOP"}]}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"test image")
            client = GoogleDescriptionClient(api_key="secret", max_retries=0)
            client._prepared = True
            with patch("description.google.httpx.post", return_value=response):
                result = client.complete_image(image_path=image, prompt="Analyze", schema=None, max_tokens=100)
        self.assertEqual(result["content"], '{"p":[]}')
        self.assertEqual(result["thought_summary"], "")
        self.assertEqual(result["thinking_tokens"], 0)

    def test_format_repair_keeps_both_original_api_archives(self):
        responses = []
        for text in ("broken JSON", '{"p":[]}'):
            response = Mock()
            response.json.return_value = {"candidates": [{"content": {"parts": [
                {"text": "Summary for " + text, "thought": True}, {"text": text},
            ]}, "finishReason": "STOP"}]}
            responses.append(response)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "page.jpg"
            image.write_bytes(b"test image")
            client = GoogleDescriptionClient(api_key="secret", max_retries=0,
                                             response_dir=root / "archives")
            client._prepared = True
            with patch("description.google.httpx.post", side_effect=responses):
                result = _remote_result(client, image, "Analyze", None, 1000)
            saved = [json.loads(p.read_text(encoding="utf-8"))
                     for p in (root / "archives").glob("*.json")]
        self.assertEqual(result["prior_raw_responses"], ["broken JSON"])
        self.assertEqual(result["thought_summary"], 'Summary for {"p":[]}')
        self.assertCountEqual([row["final_answer"] for row in saved], ["broken JSON", '{"p":[]}'])
        self.assertTrue(all(row["thought_summary"] for row in saved))

    def test_thought_only_reply_is_archived_but_rejected_as_missing_answer(self):
        data = {"candidates": [{"content": {"parts": [
            {"text": "Summary only", "thought": True}]}, "finishReason": "MAX_TOKENS"}]}
        response = Mock()
        response.json.return_value = data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "page.jpg"
            image.write_bytes(b"test image")
            client = GoogleDescriptionClient(api_key="secret", max_retries=0,
                                             response_dir=root / "archives")
            client._prepared = True
            with patch("description.google.httpx.post", return_value=response), \
                    self.assertRaisesRegex(RuntimeError, "returned empty content"):
                client.complete_image(image_path=image, prompt="Analyze", schema=None, max_tokens=100)
            saved = json.loads(next((root / "archives").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["final_answer"], "")
            self.assertEqual(saved["thought_summary"], "Summary only")
            self.assertEqual(saved["raw_api_response"], data)

    def test_gemma_uses_supported_minimal_thinking_level(self):
        response = Mock()
        response.json.return_value = {"candidates": [{
            "content": {"parts": [{"text": '{"p":[]}'}]}, "finishReason": "STOP",
        }]}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"test image")
            client = GoogleDescriptionClient(api_key="secret", model="gemma-4-31b-it", max_retries=0)
            client._prepared = True
            with patch("description.google.httpx.post", return_value=response) as post:
                client.complete_image(image_path=image, prompt="Describe", schema=None, max_tokens=100)
        self.assertEqual(post.call_args.kwargs["json"]["generationConfig"]["thinkingConfig"],
                         {"thinkingLevel": "MINIMAL"})


if __name__ == "__main__":
    unittest.main()
