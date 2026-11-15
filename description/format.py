"""Small validator for the JSON Schema subset used by extraction requests."""

import json


class FormatResponseError(ValueError):
    """Bounded format repair failed; retain attempts for runner diagnostics."""

    def __init__(self, message, responses):
        super().__init__(message)
        self.responses = responses


def validate_grounding(value: dict, labels) -> None:
    """Validate request-local endpoint mappings, not the content of prose."""
    mapping = {row["label"]: row.get("panel_number") for row in (labels or [])}

    def endpoint(label, number):
        if label is not None and (label not in mapping or mapping[label] != number):
            allowed = [key for key, panel in mapping.items() if panel == number]
            raise ValueError(
                f"Character label {label} is not in the referenced panel {number}; "
                f"its assigned panel is {mapping.get(label)}. "
                f"Allowed labels in panel {number}: {', '.join(allowed) or 'none'}. "
                "Use null if the visible actor/target/speaker has no supplied label in this panel; "
                "do not reuse another depiction's label or move the action to another panel."
            )

    for row in value.get("p", []):
        for action in row.get("a", []):
            endpoint(action.get("s"), row["n"])
            endpoint(action.get("t"), row["n"])
            if action.get("s") is not None and action.get("s") == action.get("t"):
                raise ValueError("Action endpoints must not be identical")
        for text in row.get("t", []):
            endpoint(text.get("s"), row["n"])
    for event in value.get("e", []):
        endpoint(event.get("si"), event.get("sp"))
        endpoint(event.get("ti"), event.get("tp"))
    if isinstance(value.get("x"), dict):
        validate_grounding(value["x"], labels)


def parse_json(text: str):
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    def unique_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_keys)


def validate_format(value, schema: dict, path: str = "$") -> None:
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate_format(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: invalid value type or label")
    kind = schema.get("type")
    matches = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "integer": type(value) is int,
        "null": value is None,
    }
    if kind is not None and not matches.get(kind, False):
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum value")
    if kind == "object":
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            raise ValueError(f"{path}: missing required field")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path}: unexpected field")
        for key, item in value.items():
            if key in properties:
                validate_format(item, properties[key], f"{path}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            raise ValueError(f"{path}: invalid item count")
        for index, item in enumerate(value):
            validate_format(item, schema.get("items", {}), f"{path}[{index}]")
        # Panel rows must cover each requested number once, not repeat an enum.
        numbers = schema.get("items", {}).get("properties", {}).get("n", {}).get("enum")
        if numbers is not None and (len(value) != len(numbers) or {row["n"] for row in value} != set(numbers)):
            raise ValueError(f"{path}: panel coverage does not match the request")
    elif kind == "string":
        if schema.get("minLength", 0) > 0 and not value.strip():
            raise ValueError(f"{path}: empty string")
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", float("inf")):
            raise ValueError(f"{path}: invalid string length")
    elif kind == "integer":
        if not schema.get("minimum", float("-inf")) <= value <= schema.get("maximum", float("inf")):
            raise ValueError(f"{path}: integer outside allowed range")
