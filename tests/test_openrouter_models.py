"""Model switching and metadata discovery, with no network/inference calls."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from description.openrouter import OpenRouterDescriptionClient, normalize_model_id
from test_description_openrouter import _parser


def metadata(parameters, *, modalities=None, endpoints=None):
    response = Mock()
    response.json.return_value = {"data": {
        "architecture": {"input_modalities": ["text", "image"] if modalities is None else modalities},
        "endpoints": [{"supported_parameters": parameters}] if endpoints is None else endpoints,
    }}
    return response


def completion():
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"p":[]}'}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 4},
    }
    return response


class ModelSwitchTests(unittest.TestCase):
    def test_model_id_or_full_link_produces_the_same_id(self):
        model = "google/gemma-4-26b-a4b-it:free"
        for value in (model, f"https://openrouter.ai/{model}?view=api", f"https://www.openrouter.ai/{model}/#api"):
            self.assertEqual(normalize_model_id(value), model)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(_parser().parse_args(["--model", value]).model, model)
        with patch.dict(os.environ, {"OPENROUTER_MODEL": f"https://openrouter.ai/{model}?view=api"}):
            self.assertEqual(_parser().parse_args([]).model, model)

    def test_invalid_links_and_ids_are_rejected(self):
        for value in ("", "gemma", "provider/", "a/b/c", "a/b?view=api", "a/b c", "https://other.example/a/b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_model_id(value)

    def test_auto_uses_capabilities_not_model_names_and_fetches_only_once(self):
        cases = [
            (["max_tokens", "temperature", "reasoning", "response_format", "structured_outputs"], "json_schema"),
            (["max_tokens", "response_format"], "json_object"),
            (["max_completion_tokens"], "text"),
        ]
        schema = {"type": "object", "properties": {"p": {"type": "array"}}, "required": ["p"]}
        for parameters, expected in cases:
            with self.subTest(mode=expected), tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "page.jpg"
                image.write_bytes(b"image")
                client = OpenRouterDescriptionClient(api_key="secret", model="any-provider/any-model")
                with patch("description.openrouter.httpx.get", return_value=metadata(parameters)) as get, \
                        patch("description.openrouter.httpx.post", return_value=completion()) as post:
                    client.prepare()
                    fingerprint = client.cache_key
                    for _ in range(2):
                        client.complete_image(image_path=image, prompt="Describe", schema=schema, max_tokens=99)
                get.assert_called_once()
                self.assertEqual(post.call_count, 2)
                self.assertEqual(client.response_format, expected)
                self.assertEqual(fingerprint[3], expected)
                payload = post.call_args.kwargs["json"]
                if expected == "text":
                    self.assertNotIn("response_format", payload)
                else:
                    self.assertEqual(payload["response_format"]["type"], expected)
                if expected != "json_schema":
                    self.assertIn(json.dumps(schema, separators=(",", ":")), payload["messages"][0]["content"][1]["text"])
                self.assertEqual("reasoning" in payload, "reasoning" in parameters)
                self.assertEqual("temperature" in payload, "temperature" in parameters)
                token_key = "max_tokens" if "max_tokens" in parameters else "max_completion_tokens"
                self.assertEqual(payload[token_key], 99)
                self.assertEqual(payload["model"], "any-provider/any-model")

    def test_new_free_gemma_is_normalized_and_detected_before_inference(self):
        client = OpenRouterDescriptionClient(
            api_key="secret", model="https://openrouter.ai/google/gemma-4-26b-a4b-it:free?view=api",
        )
        with patch("description.openrouter.httpx.get", return_value=metadata([
            "reasoning", "include_reasoning", "max_tokens", "temperature", "response_format",
        ])) as get, patch("description.openrouter.httpx.post") as post:
            client.prepare()
        self.assertEqual(client.model, "google/gemma-4-26b-a4b-it:free")
        self.assertEqual(client.response_format, "json_object")
        self.assertTrue(get.call_args.args[0].endswith("/models/google/gemma-4-26b-a4b-it:free/endpoints"))
        post.assert_not_called()

    def test_unavailable_or_nonvision_model_fails_before_inference(self):
        for response in (metadata(["max_tokens"], modalities=["text"]), metadata([], endpoints=[])):
            client = OpenRouterDescriptionClient(api_key="secret", model="vendor/model")
            with patch("description.openrouter.httpx.get", return_value=response), \
                    patch("description.openrouter.httpx.post") as post:
                with self.assertRaisesRegex(RuntimeError, "Cannot configure OpenRouter model"):
                    client.prepare()
            post.assert_not_called()

    def test_parameter_selection_never_combines_incompatible_endpoints(self):
        endpoints = [
            {"supported_parameters": ["max_tokens", "response_format", "temperature"]},
            {"supported_parameters": ["max_tokens", "response_format", "reasoning"]},
        ]
        client = OpenRouterDescriptionClient(api_key="secret", model="vendor/model")
        with patch("description.openrouter.httpx.get", return_value=metadata([], endpoints=endpoints)):
            client.prepare()
        self.assertIn(client._supported_parameters, [set(row["supported_parameters"]) for row in endpoints])
        self.assertEqual(client.response_format, "json_object")

    def test_explicit_format_override_requires_no_metadata_request(self):
        client = OpenRouterDescriptionClient(api_key="secret", model="vendor/model", response_format="json_object")
        with patch("description.openrouter.httpx.get") as get:
            client.prepare()
            self.assertEqual(client.response_format, "json_object")
            self.assertEqual(client.cache_key[3], "json_object")
        get.assert_not_called()

    def test_model_and_format_both_change_the_checkpoint_key(self):
        clients = [OpenRouterDescriptionClient(api_key="secret", model=model, response_format=mode)
                   for model, mode in (("vendor/a", "json_schema"), ("vendor/b", "json_schema"), ("vendor/a", "json_object"))]
        self.assertEqual(len({client.cache_key for client in clients}), 3)


if __name__ == "__main__":
    unittest.main()
