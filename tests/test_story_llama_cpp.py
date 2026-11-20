import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import requests

from story_writer.llama_cpp import LlamaCppWriter


class LlamaCppTests(unittest.TestCase):
    def client(self, value):
        client = LlamaCppWriter.__new__(LlamaCppWriter)
        client.url = "http://127.0.0.1:12345"
        client.session = MagicMock()
        client.session.post.return_value.ok = True
        client.session.post.return_value.json.return_value = value
        return client

    def test_generation_uses_native_schema_and_reports_speed_and_truncation(self):
        client = self.client({
            "choices": [{"message": {"content": '{"segments":[]}'}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 90, "completion_tokens": 40},
            "timings": {"predicted_per_second": 20},
        })
        schema = {"type": "object"}
        with patch("story_writer.llama_cpp.time.perf_counter", side_effect=[10, 12]):
            raw, metrics = client.generate("prompt", max_new_tokens=40, response_schema=schema)
        self.assertEqual(raw, '{"segments":[]}')
        self.assertTrue(metrics["truncated"])
        self.assertEqual(metrics["tokens_per_second"], 20)
        self.assertEqual(metrics["timings"]["predicted_per_second"], 20)
        payload = client.session.post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], schema)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "prompt"}])

    def test_token_count_is_cached_and_does_not_generate(self):
        client = self.client({"tokens": [1, 2, 3]})
        self.assertEqual(client.encode("text"), [1, 2, 3])
        self.assertEqual(client.encode("text"), [1, 2, 3])
        self.assertEqual(client.session.post.call_count, 1)
        self.assertTrue(client.session.post.call_args.args[0].endswith("/tokenize"))

    def test_http_errors_are_not_silently_accepted(self):
        client = self.client({})
        client.session.post.return_value.ok = False
        client.session.post.return_value.status_code = 400
        client.session.post.return_value.text = "Unsupported schema"
        with self.assertRaisesRegex(RuntimeError, "400.*Unsupported schema"):
            client.generate("prompt", max_new_tokens=40, response_schema={})

    def test_transient_errors_wait_and_retry_without_manual_restart(self):
        client = self.client({})
        unavailable = MagicMock(ok=False, status_code=503)
        success = MagicMock(ok=True)
        success.json.return_value = {"tokens": [1, 2]}
        client.session.post.side_effect = [requests.Timeout(), unavailable, success]
        with patch("story_writer.llama_cpp.time.sleep") as wait:
            self.assertEqual(client.encode("retry text"), [1, 2])
        self.assertEqual(client.session.post.call_count, 3)
        self.assertEqual(wait.call_count, 2)

    def test_startup_is_private_and_close_only_terminates_owned_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = root / "llama-server.exe", root / "model.gguf"
            exe.touch()
            model.touch()
            session, process = MagicMock(), MagicMock()
            process.poll.return_value = None
            session.get.return_value.status_code = 200
            with patch("config.STORY_LLAMA_SERVER", exe), patch("config.STORY_MODEL_PATH", model), \
                    patch("config.OUTPUT_DIR", root), patch("config.STORY_GPU_LAYERS", 24), \
                    patch("story_writer.llama_cpp.requests.Session", return_value=session), \
                    patch("story_writer.llama_cpp.subprocess.Popen", return_value=process) as popen:
                client = LlamaCppWriter()
                client.close()
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertEqual(command[command.index("--n-gpu-layers") + 1], "24")
            self.assertIn('{"enable_thinking":false}', command)
            self.assertFalse(session.trust_env)
            process.terminate.assert_called_once()
            process.kill.assert_not_called()
            session.close.assert_called_once()

    def test_persistent_network_failure_stops_after_three_attempts(self):
        client = self.client({})
        client.session.post.side_effect = requests.Timeout()
        with patch("story_writer.llama_cpp.time.sleep"), self.assertRaisesRegex(RuntimeError, "3 attempts"):
            client.encode("persistent failure")
        self.assertEqual(client.session.post.call_count, 3)

    def test_dead_owned_server_does_not_retry_forever(self):
        client = self.client({})
        client.process = MagicMock()
        client.process.poll.return_value = 1
        client.session.post.side_effect = requests.ConnectionError()
        with self.assertRaisesRegex(RuntimeError, "1 attempts"):
            client.encode("dead server")
        self.assertEqual(client.session.post.call_count, 1)

    def test_close_kills_only_its_process_if_termination_times_out(self):
        client = self.client({})
        client.process = MagicMock()
        client.process.poll.return_value = None
        client.process.wait.side_effect = [subprocess.TimeoutExpired("llama", 10), None]
        client.log = MagicMock()
        client.close()
        client.process.kill.assert_called_once()
        client.log.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
