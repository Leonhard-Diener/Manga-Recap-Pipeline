"""Small OpenRouter vision client used by the alternate description runner."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx


def normalize_model_id(value: str) -> str:
    """Accept a provider/model ID or its OpenRouter model-page URL."""
    value = value.strip()
    if "://" in value:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or url.netloc.lower() not in {"openrouter.ai", "www.openrouter.ai"}:
            raise ValueError("Use an OpenRouter model link or a provider/model ID.")
        value = unquote(url.path).strip("/")
    if (value.count("/") != 1 or any(char.isspace() for char in value)
            or any(char in value for char in "?#") or not all(value.split("/"))):
        raise ValueError("Expected a provider/model ID, such as google/gemma-4-26b-a4b-it:free.")
    return value


class OpenRouterDescriptionClient:
    """Call an OpenAI-compatible OpenRouter chat-completions endpoint."""

    is_remote_description_client = True
    provenance_prefix = "openrouter_vision"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 180.0,
        max_retries: int = 3,
        app_url: str | None = None,
        app_name: str = "Manga Recap",
        response_format: str = "auto",
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENROUTER_API_KEY is empty.")
        if response_format not in {"auto", "json_schema", "json_object", "text"}:
            raise ValueError("Response format must be auto, json_schema, json_object, or text.")
        self.api_key = api_key.strip()
        self.model = normalize_model_id(model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.app_url = app_url
        self.app_name = app_name
        self._page_bundles: dict[str, dict] = {}
        self._response_format = None if response_format == "auto" else response_format
        self._supported_parameters = {"max_tokens", "temperature", "reasoning"}

    def prepare(self) -> None:
        """Check endpoint capabilities once; never execute or switch a model."""
        if self._response_format is not None:
            return
        try:
            response = httpx.get(
                f"{self.base_url}/models/{self.model}/endpoints",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=min(self.timeout, 30.0),
            )
            response.raise_for_status()
            metadata = response.json()["data"]
            modalities = metadata.get("architecture", {}).get("input_modalities", [])
            if "image" not in modalities:
                raise ValueError(f"{self.model} does not advertise image input.")
            endpoints = metadata.get("endpoints", [])
            if not endpoints:
                raise ValueError(f"No endpoints are available for {self.model}.")
            options = []
            for endpoint in endpoints:
                parameters = set(endpoint.get("supported_parameters", []))
                if not parameters.intersection({"max_tokens", "max_completion_tokens"}):
                    continue
                mode = (
                    "json_schema" if {"response_format", "structured_outputs"} <= parameters
                    else "json_object" if "response_format" in parameters else "text"
                )
                options.append((mode, parameters))
            if not options:
                raise ValueError("No endpoint supports an output-token limit.")
            # Select a compatible parameter set from ONE actual endpoint, not a
            # union of parameters that no single provider could handle.
            priority = {"json_schema": 0, "json_object": 1, "text": 2}
            mode, parameters = min(options, key=lambda item: (priority[item[0]], "reasoning" not in item[1]))
            self._supported_parameters = parameters
            self._response_format = mode
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as error:
            raise RuntimeError(
                f"Cannot configure OpenRouter model {self.model}: {error}. "
                "Check the model link and connection. For compatible custom endpoints, "
                "set --response-format json_schema, json_object, or text explicitly."
            ) from error

    @property
    def response_format(self) -> str:
        self.prepare()
        return self._response_format

    @property
    def cache_key(self) -> tuple:
        """Values that invalidate description checkpoints when changed."""
        self.prepare()
        return (
            "openrouter_combined_v7_auto_capabilities", self.base_url, self.model,
            self.response_format, tuple(sorted(self._supported_parameters.intersection({
                "max_tokens", "max_completion_tokens", "temperature", "reasoning",
                "response_format", "structured_outputs",
            }))),
        )

    def _image_url(self, path: Path) -> str:
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def clear_page_bundles(self) -> None:
        """Drop chapter-local combined responses before processing a chapter."""
        self._page_bundles.clear()

    def store_page_bundle(self, path: Path, bundle: dict) -> None:
        self._page_bundles[Path(path).name] = bundle

    def page_bundle(self, path: Path) -> dict | None:
        return self._page_bundles.get(Path(path).name)

    @staticmethod
    def _content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
            )
        return ""

    def complete_image(
        self,
        *,
        image_path: Path,
        prompt: str,
        schema: dict | None,
        max_tokens: int,
    ) -> dict:
        """Send one image request and return content plus usage metadata."""
        self.prepare()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if schema is not None and self.response_format != "json_schema":
            prompt += (
                "\nReturn a JSON object matching this schema. Preserve the required "
                "keys and value types exactly; use empty lists when appropriate.\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._image_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "stream": False,
            # Require the chosen format (JSON object or strict JSON Schema).
            # The extraction layer validates the full schema in either mode.
            "provider": {"require_parameters": True},
        }
        token_parameter = "max_tokens" if "max_tokens" in self._supported_parameters else "max_completion_tokens"
        payload[token_parameter] = max_tokens
        if "temperature" in self._supported_parameters:
            payload["temperature"] = 0
        if "reasoning" in self._supported_parameters:
            # Avoid spending the output budget on reasoning when supported.
            payload["reasoning"] = {"enabled": False}
        if schema is not None:
            if self.response_format == "json_object":
                payload["response_format"] = {"type": "json_object"}
            elif self.response_format == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "manga_description",
                        "strict": True,
                        "schema": schema,
                    },
                }

        started = time.perf_counter()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = self._content_text(message.get("content"))
                if not content.strip():
                    usage = data.get("usage") or {}
                    token_details = usage.get("completion_tokens_details") or {}
                    raise RuntimeError(
                        "OpenRouter returned empty content "
                        f"(finish_reason={choice.get('finish_reason')!r}, "
                        f"native_finish_reason={choice.get('native_finish_reason')!r}, "
                        f"completion_tokens={usage.get('completion_tokens')!r}, "
                        f"reasoning_tokens={token_details.get('reasoning_tokens')!r})."
                    )
                usage = data.get("usage") or {}
                return {
                    "content": content,
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens") or 0,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except (httpx.HTTPError, ValueError, KeyError, IndexError, RuntimeError) as error:
                if isinstance(error, httpx.HTTPStatusError):
                    try:
                        error_body = error.response.text.strip()
                    except Exception:
                        error_body = ""
                    if len(error_body) > 2000:
                        error_body = error_body[:2000] + "..."
                    last_error = RuntimeError(
                        f"HTTP {error.response.status_code}: "
                        f"{error_body or error.response.reason_phrase}"
                    )
                else:
                    last_error = error
                if isinstance(error, httpx.HTTPStatusError):
                    retryable = (
                        error.response.status_code == 429
                        or error.response.status_code >= 500
                    )
                else:
                    # Malformed/empty successful responses are deterministic
                    # and must not spend the user's quota on blind retries.
                    retryable = isinstance(error, httpx.RequestError)
                if attempt >= self.max_retries or not retryable:
                    break
                time.sleep(min(8.0, 1.0 * (2 ** attempt)))
        raise RuntimeError(f"OpenRouter request failed: {last_error}") from last_error
