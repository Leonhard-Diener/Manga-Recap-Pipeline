from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from description.describer import (
    _openrouter_bundle_schema,
    describe_character_pages,
    describe_focused_pages,
    describe_manga,
)
from description.openrouter import OpenRouterDescriptionClient
from description.schema import character_response_schema
from test_description_openrouter import _parser


class OpenRouterClientTests(unittest.TestCase):
    def test_default_model_is_free_gemma_and_remains_overridable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_parser().parse_args([]).model, "google/gemma-4-26b-a4b-it:free")
            self.assertEqual(_parser().parse_args(["--model", "vendor/model"]).model, "vendor/model")
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "vendor/env-model"}):
            self.assertEqual(_parser().parse_args([]).model, "vendor/env-model")

    def test_page_progress_is_checkpointed_and_resumed_after_interruption(self):
        class Backend:
            cache_key = ("test-remote",)
            model = "vendor/model"
            base_url = "https://example.invalid/v1"
            provenance_prefix = "remote_test"

        partial_chapter = {
            "descriptions": {"00001.jpg": {"panels": []}},
            "transformations": {"00001.jpg": {"events": []}},
            "page_filter": {"00001.jpg": {"included_in_story": True}},
        }
        published = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            chapter_output = output / "chapter_001"
            chapter_output.mkdir()
            (chapter_output / "panel_manifest.json").write_text(
                json.dumps({"schema_version": 2, "pages": []}), encoding="utf-8",
            )
            chapters = [Path("chapter_001")]

            def interrupted(*args, **kwargs):
                kwargs["progress_callback"](partial_chapter)
                raise RuntimeError("simulated interruption")

            with patch("description.describer.describe_chapter", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    describe_manga(
                        output, chapters, filter_non_story=True, backend=Backend(),
                        progress_callback=published.append,
                    )

            checkpoint_path = chapter_output / "description_checkpoint_openrouter.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertFalse(checkpoint["complete"])
            self.assertEqual(checkpoint["result"], partial_chapter)
            self.assertEqual(
                published[-1]["descriptions"]["chapter_001"],
                partial_chapter["descriptions"],
            )

            def resumed(*args, **kwargs):
                self.assertEqual(kwargs["resume_result"], partial_chapter)
                return kwargs["resume_result"]

            with patch("description.describer.describe_chapter", side_effect=resumed):
                result = describe_manga(
                    output, chapters, filter_non_story=True, backend=Backend(),
                )
            self.assertEqual(
                result["descriptions"]["chapter_001"], partial_chapter["descriptions"],
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertTrue(checkpoint["complete"])

    def test_combined_schema_has_no_empty_enum_without_character_labels(self):
        schema = _openrouter_bundle_schema([1], [])

        def visit(value):
            if isinstance(value, dict):
                self.assertNotEqual(value.get("enum"), [])
                if "enum" in value:
                    self.assertNotIn("minimum", value)
                    self.assertNotIn("maximum", value)
                    self.assertNotIn("minLength", value)
                    self.assertNotIn("maxLength", value)
                    self.assertNotIn("pattern", value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(schema)
        panel = schema["properties"]["p"]["items"]
        self.assertIn("a", panel["required"])
        self.assertIn("c", schema["required"])

    def test_character_schemas_require_representation_type(self):
        labels = [{"label": "C1", "detection_id": "det1", "panel_number": 1}]
        for schema in (
            character_response_schema(labels),
            _openrouter_bundle_schema([1], labels)["properties"]["c"],
        ):
            value = schema["properties"]["C1"]
            self.assertEqual(value["required"], ["d", "r"])
            self.assertIn("toy", value["properties"]["r"]["enum"])

    def test_remote_character_bundle_preserves_representation_type(self):
        class FakeRemote:
            is_remote_description_client = True

            def clear_page_bundles(self):
                pass

            def page_bundle(self, path):
                return {"c": {"C1": {"d": "small armored boy figurine", "r": "toy"}}}

        labels = [{"label": "C1", "detection_id": "det1", "panel_number": 1}]
        result = describe_character_pages(
            [(Path("page.jpg"), labels)], FakeRemote(), FakeRemote(),
        )[0]
        self.assertEqual(result["characters"][0]["representation_type"], "toy")
        self.assertEqual(result["characters"][0]["visual_description"], "small armored boy figurine")

    def test_client_sends_image_model_and_json_schema(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "message": {"content": '{"p":[]}'},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 7},
        }
        schema = {"type": "object", "properties": {}}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"not-a-real-jpeg")
            client = OpenRouterDescriptionClient(
                api_key="secret", model="vendor/vision-model", max_retries=0,
                response_format="json_schema",
            )
            with patch("description.openrouter.httpx.post", return_value=response) as post:
                result = client.complete_image(
                    image_path=image, prompt="Describe it", schema=schema, max_tokens=99,
                )

        request = post.call_args.kwargs
        self.assertEqual(request["json"]["model"], "vendor/vision-model")
        self.assertEqual(request["json"]["max_tokens"], 99)
        self.assertEqual(request["json"]["reasoning"], {"enabled": False})
        self.assertEqual(request["json"]["provider"], {"require_parameters": True})
        self.assertFalse(request["json"]["stream"])
        self.assertEqual(
            request["json"]["response_format"]["json_schema"]["schema"], schema,
        )
        image_url = request["json"]["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(result["completion_tokens"], 7)

    def test_empty_reasoning_response_is_diagnostic_and_not_retried(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "message": {"content": None, "reasoning_details": [{"type": "reasoning.text"}]},
                "finish_reason": "length",
                "native_finish_reason": "length",
            }],
            "usage": {
                "completion_tokens": 1024,
                "completion_tokens_details": {"reasoning_tokens": 1024},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"not-a-real-jpeg")
            client = OpenRouterDescriptionClient(
                api_key="secret", model="vendor/reasoning-model", max_retries=3,
                response_format="json_schema",
            )
            with patch("description.openrouter.httpx.post", return_value=response) as post:
                with self.assertRaisesRegex(RuntimeError, "reasoning_tokens=1024"):
                    client.complete_image(
                        image_path=image, prompt="Describe it", schema=None, max_tokens=1024,
                    )
        post.assert_called_once()

    def test_free_gemma_uses_json_object_with_schema_in_prompt_in_one_call(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"p":[]}'}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 7},
        }
        schema = {"type": "object", "properties": {"p": {"type": "array"}}, "required": ["p"]}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"fake jpeg")
            client = OpenRouterDescriptionClient(
                api_key="secret", model="google/gemma-4-31b-it:free", max_retries=0,
                response_format="json_object",
            )
            with patch("description.openrouter.httpx.post", return_value=response) as post:
                result = client.complete_image(
                    image_path=image, prompt="Describe it", schema=schema, max_tokens=99,
                )
        post.assert_called_once()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "google/gemma-4-31b-it:free")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["provider"], {"require_parameters": True})
        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertEqual(payload["max_tokens"], 99)
        prompt = payload["messages"][0]["content"][1]["text"]
        self.assertTrue(prompt.startswith("Describe it\n"))
        self.assertIn(json.dumps(schema, separators=(",", ":")), prompt)
        self.assertEqual(result["content"], '{"p":[]}')
        self.assertEqual(client.cache_key[3], "json_object")

    def test_free_gemma_json_mode_still_validates_and_repairs_schema_locally(self):
        def completion(raw):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 10},
            }
            return response

        valid = {
            "r": "story", "p": [{"n": 1, "d": "The exact model wording.", "a": []}],
            "c": {}, "e": [], "x": {"p": [{"n": 1, "t": []}]},
        }
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"fake jpeg")
            client = OpenRouterDescriptionClient(
                api_key="secret", model="google/gemma-4-31b-it:free", max_retries=0,
                response_format="json_object",
            )
            with patch("description.openrouter.httpx.post", side_effect=[
                completion('{"wrong":"schema"}'), completion(json.dumps(valid)),
            ]) as post:
                result = describe_focused_pages(
                    [(image, [1], [])], client, client, task="observation",
                )[0]
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["prior_raw_responses"], ['{"wrong":"schema"}'])
        self.assertEqual(result["payload"]["p"][0]["d"], "The exact model wording.")
        self.assertTrue(all(call.kwargs["json"]["response_format"] == {"type": "json_object"}
                            for call in post.call_args_list))

    def test_focused_pipeline_accepts_remote_backend(self):
        class FakeRemote:
            is_remote_description_client = True

            def __init__(self):
                self.bundles = {}
                self.calls = 0

            def complete_image(self, **kwargs):
                self.calls += 1
                self.request = kwargs
                return {
                    "content": json.dumps({
                        "r": "story",
                        "p": [{"n": 1, "d": "A swordsman crosses a stone bridge.", "a": []}],
                        "c": {},
                        "e": [],
                        "x": {"p": [{"n": 1, "t": []}]},
                    }),
                    "finish_reason": "stop",
                    "completion_tokens": 12,
                    "elapsed_seconds": 0.25,
                }

            def store_page_bundle(self, path, bundle):
                self.bundles[Path(path).name] = bundle

            def page_bundle(self, path):
                return self.bundles.get(Path(path).name)

        backend = FakeRemote()
        result = describe_focused_pages(
            [(Path("page.jpg"), [1], [])], backend, backend, task="observation",
        )[0]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["payload"]["p"][0]["n"], 1)
        self.assertEqual(result["generated_tokens"], 12)
        transformation = describe_focused_pages(
            [(Path("page.jpg"), [1], [])], backend, backend, task="transformation",
        )[0]
        text = describe_focused_pages(
            [(Path("page.jpg"), [1], [])], backend, backend, task="text",
        )[0]
        characters = describe_character_pages([], backend, backend)
        self.assertEqual(transformation["payload"], {"e": []})
        self.assertEqual(text["payload"]["p"][0]["n"], 1)
        self.assertEqual(characters, [])
        self.assertEqual(backend.calls, 1)
        self.assertIn("x", backend.request["schema"]["required"])


if __name__ == "__main__":
    unittest.main()
