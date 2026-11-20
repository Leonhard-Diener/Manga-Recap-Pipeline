"""Owned, localhost-only llama.cpp server for the text-only story writer."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import socket
import subprocess
import time

import requests

import config


class LlamaCppWriter:
    def __init__(self):
        executable = Path(config.STORY_LLAMA_SERVER)
        model = Path(config.STORY_MODEL_PATH)
        for path in (executable, model):
            if not path.is_file():
                raise FileNotFoundError(f"Missing story runtime/model: {path}. See STORY_WRITER_SPEC.md.")
        self.tokenizer = self
        self.session = requests.Session()
        self.session.trust_env = False
        self.process = None
        self.log = None
        log_dir = config.OUTPUT_DIR / "llama_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"story_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        try:
            self.log = self.log_path.open("w", encoding="utf-8")
            self.process = subprocess.Popen([
                str(executable), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                "--ctx-size", str(config.STORY_CONTEXT_TOKEN_BUDGET), "--parallel", "1",
                "--n-gpu-layers", str(config.STORY_GPU_LAYERS), "--jinja", "--reasoning", "off",
                "--chat-template-kwargs", '{"enable_thinking":false}', "--no-warmup",
            ], stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama.cpp exited during startup; see {self.log_path}")
                try:
                    response = self.session.get(self.url + "/health", timeout=2)
                    if response.status_code == 200:
                        return
                except (requests.ConnectionError, requests.Timeout):
                    pass
                time.sleep(0.2)
            raise RuntimeError(f"llama.cpp startup timed out; see {self.log_path}")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.session.close()
        if self.log is not None:
            self.log.close()

    def _post(self, route, payload):
        for attempt in range(3):
            try:
                response = self.session.post(self.url + route, json=payload, timeout=(10, 600))
            except (requests.Timeout, requests.ConnectionError) as error:
                reason = type(error).__name__
            else:
                if response.ok:
                    return response.json()
                if response.status_code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"llama.cpp HTTP {response.status_code}: {response.text[:1200]}")
                reason = f"HTTP {response.status_code}"
            process = getattr(self, "process", None)
            if attempt == 2 or (process is not None and process.poll() is not None):
                raise RuntimeError(f"llama.cpp {reason}; request failed after {attempt + 1} attempts.")
            print(f"llama.cpp {reason}; waiting 10 seconds before retry {attempt + 2}/3 (Ctrl+C to stop).", flush=True)
            time.sleep(10)

    @lru_cache(maxsize=128)
    def encode(self, text, add_special_tokens=False):
        return self._post("/tokenize", {"content": text, "add_special": add_special_tokens})["tokens"]

    def generate(self, prompt, *, max_new_tokens, response_schema):
        started = time.perf_counter()
        result = self._post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_new_tokens, "seed": 0,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "story", "strict": True, "schema": response_schema,
            }},
        })
        choice = result["choices"][0]
        raw = choice["message"].get("content")
        if not isinstance(raw, str):
            raise ValueError("llama.cpp returned no narration JSON.")
        elapsed = time.perf_counter() - started
        usage = result.get("usage", {})
        generated = usage.get("completion_tokens", 0)
        return raw, {
            "input_tokens": usage.get("prompt_tokens", 0), "generated_tokens": generated,
            "token_limit": max_new_tokens, "truncated": choice.get("finish_reason") == "length",
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(generated / elapsed, 2) if elapsed else 0.0,
            "timings": result.get("timings", {}), "backend": "llama.cpp",
            "gpu_layers": config.STORY_GPU_LAYERS,
        }
