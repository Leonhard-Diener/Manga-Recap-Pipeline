"""Google Gemini vision client for the shared description pipeline."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import mimetypes
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import uuid4

import httpx

from common.json_io import write_json


_MODEL_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_RESOLUTIONS = {"low", "medium", "high"}
_EXPENSIVE_SCHEMA_CONSTRAINTS = {
    "minimum", "maximum", "minLength", "maxLength", "pattern",
    "minItems", "maxItems",
}


class GooglePermanentError(RuntimeError):
    """An API rejection that must stop the run rather than spending on later pages."""

    stop_run = True


def _google_response_schema(value):
    """Keep response shape while avoiding Gemini's small schema-state limit."""
    if isinstance(value, list):
        return [_google_response_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _google_response_schema(item)
        for key, item in value.items()
        if key not in _EXPENSIVE_SCHEMA_CONSTRAINTS
    }
    enum = value.get("enum")
    if isinstance(enum, list) and any(type(item) is int for item in enum):
        result.pop("enum", None)
    if "type" not in result and isinstance(enum, list) and enum:
        if all(isinstance(item, str) for item in enum):
            result["type"] = "string"
        elif all(type(item) is int for item in enum):
            result["type"] = "integer"
    return result


def _google_panel_slots(schema, character_labels):
    """Use required keyed panels on the wire, retaining the public array format."""
    schema = copy.deepcopy(schema)
    if not isinstance(schema, dict):
        return schema
    properties = schema.get("properties", {})
    panels = properties.get("p", {})
    item = panels.get("items", {})
    numbers = item.get("properties", {}).get("n", {}).get("enum")
    if panels.get("type") == "array" and numbers:
        slots = {}
        for number in numbers:
            panel = copy.deepcopy(item)
            panel["properties"].pop("n")
            panel["required"].remove("n")
            allowed = [row["label"] for row in (character_labels or [])
                       if row.get("panel_number") == number]
            endpoint = {"anyOf": [{"type": "string", "enum": allowed}, {"type": "null"}]} \
                if allowed else {"type": "null"}
            for field, endpoints in (("a", ("s", "t")), ("t", ("s",))):
                entry = panel["properties"].get(field, {}).get("items", {}).get("properties", {})
                for name in endpoints:
                    if name in entry:
                        entry[name] = copy.deepcopy(endpoint)
            slots[f"panel_{number}"] = panel
        properties["p"] = {
            "type": "object", "properties": slots, "required": list(slots),
            "additionalProperties": False,
        }
    if "x" in properties:
        properties["x"] = _google_panel_slots(properties["x"], character_labels)
    return schema


def _restore_panel_arrays(value):
    """Undo transport keys only; never repair model observations or missing slots."""
    if not isinstance(value, dict):
        return value
    value = dict(value)
    if isinstance(value.get("p"), dict):
        value["p"] = [{**panel, "n": int(key.removeprefix("panel_"))}
                      for key, panel in value["p"].items()]
    if "x" in value:
        value["x"] = _restore_panel_arrays(value["x"])
    return value


def normalize_google_model_id(value: str) -> str:
    value = value.strip()
    if value.startswith("models/"):
        value = value[7:]
    if not _MODEL_ID.fullmatch(value):
        raise ValueError("Expected a Google model ID such as gemini-3.8-flash.")
    return value


class GoogleDescriptionClient:
    """Call Google's native Gemini generateContent endpoint."""

    is_remote_description_client = True
    provenance_prefix = "google_vision"
    provider_name = "google"
    checkpoint_name = "description_checkpoint_google.json"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        media_resolution: str = "high",
        timeout: float = 180.0,
        max_retries: int | None = None,
        response_dir: Path | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is empty.")
        if media_resolution not in _RESOLUTIONS:
            raise ValueError("Media resolution must be low, medium, or high.")
        self.api_key = api_key.strip()
        self.model = normalize_google_model_id(model)
        self.base_url = base_url.rstrip("/")
        self.media_resolution = media_resolution
        self.timeout = timeout
        self.max_retries = max(0, max_retries) if max_retries is not None else None
        self.response_dir = Path(response_dir) if response_dir is not None else None
        self.response_format = "json_schema"
        self._prepared = False
        self._page_bundles: dict[str, dict] = {}

    def _wait_retry(self, error, attempt: int) -> bool:
        status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
        transient = status in {408, 500, 502, 503, 504} or isinstance(error, httpx.RequestError)
        if not transient or (self.max_retries is not None and attempt >= self.max_retries):
            return False
        delay = max(45, min(300, 30 * 2 ** min(attempt, 4)))
        if status is not None:
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    seconds = float(retry_after)
                except ValueError:
                    try:
                        seconds = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                    except (TypeError, ValueError, OverflowError):
                        seconds = 0
                if math.isfinite(seconds):
                    # Bound absurd server values so sleep cannot overflow.
                    delay = max(delay, min(seconds, 86400))
        print(f"Google temporary failure ({status or 'network'}); waiting {delay:.0f}s "
              f"before retry {attempt + 1}. Completed progress is retained.", flush=True)
        time.sleep(delay)
        return True

    def prepare(self) -> None:
        """Validate model availability without running inference."""
        if self._prepared:
            return
        attempt = 0
        while True:
            try:
                response = httpx.get(
                    f"{self.base_url}/models/{self.model}",
                    headers={"x-goog-api-key": self.api_key},
                    timeout=min(self.timeout, 30.0),
                )
                response.raise_for_status()
                methods = response.json().get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    raise ValueError(f"{self.model} does not support generateContent.")
                break
            except (httpx.HTTPError, ValueError, TypeError) as error:
                if self._wait_retry(error, attempt):
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"Cannot configure Google model {self.model}: {error}. "
                    "Check GEMINI_API_KEY, GOOGLE_MODEL, and your AI Studio access."
                ) from error
        self._prepared = True

    @property
    def cache_key(self) -> tuple:
        self.prepare()
        return (
            "google_generate_content_v2", self.base_url, self.model,
            self.media_resolution, self.response_format,
            "MINIMAL" if self.model.startswith("gemma-") else "HIGH",
            not self.model.startswith("gemma-"),
        )

    def clear_page_bundles(self) -> None:
        self._page_bundles.clear()

    def store_page_bundle(self, path: Path, bundle: dict) -> None:
        self._page_bundles[Path(path).name] = bundle

    def page_bundle(self, path: Path) -> dict | None:
        return self._page_bundles.get(Path(path).name)

    @staticmethod
    def _response_text(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        return "".join(
            part.get("text", "")
            for part in candidates[0].get("content", {}).get("parts", [])
            if isinstance(part, dict) and not part.get("thought")
        )

    def complete_image(
        self,
        *,
        image_path: Path,
        prompt: str,
        schema: dict | None,
        max_tokens: int,
        character_labels=None,
    ) -> dict:
        """Send one structured image request and return shared usage fields."""
        self.prepare()
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        gemma = self.model.startswith("gemma-")
        generation_config = {
            "temperature": 0,
            # Thinking shares the output limit; reserve room without shrinking the answer.
            "maxOutputTokens": max_tokens if gemma else max_tokens + 4096,
            "mediaResolution": f"MEDIA_RESOLUTION_{self.media_resolution.upper()}",
            "thinkingConfig": {"thinkingLevel": "MINIMAL"} if gemma else {
                "thinkingLevel": "HIGH", "includeThoughts": True,
            },
        }
        wire_schema = _google_panel_slots(schema, character_labels) if character_labels is not None else schema
        if schema is not None:
            generation_config.update(
                responseMimeType="application/json",
                responseJsonSchema=_google_response_schema(wire_schema),
            )
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": media_type, "data": image}},
                    {"text": prompt},
                ],
            }],
            "generationConfig": generation_config,
        }
        if wire_schema != schema:
            payload["contents"][0]["parts"][1]["text"] += (
                "\nTransport format: p must be an object with EVERY required panel_N key, "
                "not an array. This applies to x.p too. Panel numbers are in the keys; "
                "do not return n inside panel objects. Each panel's s/t fields allow "
                "only labels from that panel or null. Follow the supplied response schema."
            )

        started = time.perf_counter()
        last_error = None
        attempt = 0
        while True:
            try:
                response = httpx.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                try:
                    data = response.json()
                except ValueError:
                    response.raise_for_status()
                    raise
                content = self._response_text(data)
                candidates = data.get("candidates") or []
                thought_summary = "\n".join(
                    part.get("text", "")
                    for part in (candidates[0].get("content", {}).get("parts", []) if candidates else [])
                    if isinstance(part, dict) and part.get("thought")
                )
                response_archive = None
                if self.response_dir is not None:
                    response_archive = self.response_dir / f"{image_path.stem}_{uuid4().hex}.json"
                    write_json(response_archive, {
                        "model": self.model, "image": str(image_path),
                        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        "prompt": payload["contents"][0]["parts"][1]["text"],
                        "generation_config": generation_config,
                        "final_answer": content, "thought_summary": thought_summary,
                        "raw_api_response": data,
                    })
                response.raise_for_status()
                finish_reason = candidates[0].get("finishReason") if candidates else None
                if finish_reason == "MAX_TOKENS":
                    finish_reason = "length"
                if not content.strip():
                    raise RuntimeError(
                        "Google returned empty content "
                        f"(finish_reason={finish_reason!r}, "
                        f"prompt_feedback={data.get('promptFeedback')!r})."
                    )
                usage = data.get("usageMetadata") or {}
                normalized = content
                validation_error = None
                if wire_schema != schema and finish_reason != "length":
                    from description.format import parse_json, validate_format
                    try:
                        decoded = parse_json(content)
                        validate_format(decoded, wire_schema)
                        normalized = json.dumps(_restore_panel_arrays(decoded), ensure_ascii=False)
                    except (TypeError, ValueError) as error:
                        # Return the original response for the shared bounded repair/audit path.
                        normalized = content
                        validation_error = str(error)
                return {
                    "content": normalized,
                    "wire_content": content,
                    "thought_summary": thought_summary,
                    "raw_api_response": data,
                    "response_archive": str(response_archive) if response_archive else None,
                    "thinking_tokens": usage.get("thoughtsTokenCount") or 0,
                    "token_limit": generation_config["maxOutputTokens"],
                    "validation_error": validation_error,
                    "finish_reason": finish_reason,
                    "completion_tokens": usage.get("candidatesTokenCount") or 0,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except (httpx.HTTPError, ValueError, KeyError, IndexError, RuntimeError) as error:
                if isinstance(error, httpx.HTTPStatusError):
                    body = error.response.text.strip()
                    error_type = GooglePermanentError if error.response.status_code in {
                        400, 401, 402, 403, 404, 429,
                    } else RuntimeError
                    last_error = error_type(
                        f"HTTP {error.response.status_code}: "
                        f"{body[:2000] or error.response.reason_phrase}"
                    )
                else:
                    last_error = error
                if not self._wait_retry(error, attempt):
                    break
                attempt += 1
        if isinstance(last_error, GooglePermanentError):
            raise GooglePermanentError(f"Google request failed: {last_error}") from last_error
        raise RuntimeError(f"Google request failed: {last_error}") from last_error
