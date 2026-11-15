"""Compact, mechanically enforced response schema for page description."""

from __future__ import annotations

import functools
import os
import pickle
from pathlib import Path

import config


def page_response_schema(expected_panels=None, character_labels=None) -> dict:
    """Build the JSON schema used for one page or a homogeneous page batch."""
    panel_numbers = None
    if expected_panels is not None:
        panel_numbers = sorted({int(number) for number in expected_panels})
        if not panel_numbers:
            raise ValueError("expected_panels must not be empty")

    number_schema = {"type": "integer", "minimum": 1}
    if panel_numbers is not None:
        number_schema["enum"] = panel_numbers

    grounded = character_labels is not None
    supplied_labels = sorted({
        row.get("label") for row in (character_labels or [])
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    }) if grounded else None
    # Grounded requests may emit only labels that are actually drawn on this
    # page. This also shrinks the constrained token search compared with the
    # previous open-ended C-number pattern.
    label_schema = (
        {"type": "string", "enum": supplied_labels}
        if grounded else
        {"type": "string", "pattern": "^C[1-9][0-9]*$"}
    )
    action_items = {
        "type": "object",
        "properties": {
            "s": {"anyOf": [label_schema, {"type": "null"}]},
            "t": {"anyOf": [label_schema, {"type": "null"}]},
            "x": {
                "type": "string", "minLength": 1,
                "maxLength": config.DESCRIPTION_MAX_ACTION_CHARS,
            },
        },
        "required": ["s", "t", "x"],
        "additionalProperties": False,
    }
    panel_properties = {
        "n": number_schema,
        "d": {
            "type": "string",
            "minLength": 1,
            "maxLength": config.DESCRIPTION_MAX_DESCRIPTION_CHARS,
        },
    }
    panel_required = ["n", "d"]
    if grounded:
        panel_properties["a"] = {
            "type": "array", "items": action_items, "maxItems": 8,
        }
        panel_required.append("a")
    panel_items = {
        "type": "object",
        "properties": panel_properties,
        "required": panel_required,
        "additionalProperties": False,
    }
    panels_schema = {
        "type": "array",
        "items": panel_items,
        "minItems": len(panel_numbers) if panel_numbers is not None else 1,
        "maxItems": len(panel_numbers) if panel_numbers is not None else 12,
    }

    event_items = {
        "type": "object",
        "properties": {
            "t": {"enum": [
                "transformation", "reversion", "disguise_reveal", "possession",
                "summoning", "splitting", "fusion",
            ]},
            "sp": number_schema,
            "tp": number_schema,
            "sd": {"type": "string", "minLength": 1, "maxLength": 80},
            "td": {"type": "string", "minLength": 1, "maxLength": 80},
            "x": {"type": "string", "minLength": 1, "maxLength": 160},
            "xt": {"enum": ["visible_change", "explicit_text", "ambiguous_sequence"]},
            "c": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["t", "sp", "tp", "sd", "td", "x", "xt", "c"],
        "additionalProperties": False,
    }
    if grounded:
        event_items["properties"].update(si=label_schema, ti=label_schema)
        event_items["required"].extend(["si", "ti"])

    properties = {
        "r": {"enum": ["story", "reference", "extra", "uncertain"]},
        "p": panels_schema,
        "e": {"type": "array", "items": event_items, "maxItems": 8},
    }
    required = ["r", "p", "e"]
    if grounded:
        properties["c"] = {
            "type": "object",
            "properties": {
                label: {
                    "type": "object",
                    "properties": {
                        "d": {"type": "string", "maxLength": 160},
                        "r": {"enum": [
                            "live_character", "toy", "statue", "image", "uncertain",
                        ]},
                    },
                    "required": ["d", "r"],
                    "additionalProperties": False,
                }
                for label in supplied_labels
            },
            "required": supplied_labels,
            "additionalProperties": False,
        }
        required.append("c")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def character_response_schema(character_labels) -> dict:
    """Exact label-to-appearance object for the focused character pass."""
    labels = sorted({
        row.get("label") for row in (character_labels or [])
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    })
    return {
        "type": "object",
        "properties": {
            # Leave enough character headroom for the model to close a phrase;
            # semantic validation enforces the shorter word limit. An 80-char
            # hard stop produced accepted fragments such as "and displaying an".
            label: {
                "type": "object",
                "properties": {
                    "d": {"type": "string", "maxLength": 160},
                    "r": {"enum": [
                        "live_character", "toy", "statue", "image", "uncertain",
                    ]},
                },
                "required": ["d", "r"],
                "additionalProperties": False,
            }
            for label in labels
        },
        "required": labels,
        "additionalProperties": False,
    }


def panel_description_schema(expected_panels) -> dict:
    numbers = sorted({int(number) for number in expected_panels})
    number_schema = {"type": "integer", "enum": numbers}
    return {
        "type": "object",
        "properties": {
            "p": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "n": number_schema,
                        "d": {
                            "type": "string", "minLength": 1,
                            "maxLength": config.DESCRIPTION_MAX_DESCRIPTION_CHARS,
                        },
                    },
                    "required": ["n", "d"],
                    "additionalProperties": False,
                },
                "minItems": len(numbers),
                "maxItems": len(numbers),
            },
        },
        "required": ["p"],
        "additionalProperties": False,
    }


def panel_observation_schema(expected_panels, character_labels) -> dict:
    """Descriptions and exact grounded visible actions, without unrelated tasks."""
    numbers = sorted({int(number) for number in expected_panels})
    labels = sorted({
        row.get("label") for row in (character_labels or [])
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    })
    label_schema = {"type": "string", "enum": labels}
    action = {
        "type": "object",
        "properties": {
            # Null records a visible action whose actor detector box is absent.
            # It remains useful diagnostic evidence but cannot become a
            # grounded story fact.
            "s": {"anyOf": [label_schema, {"type": "null"}]},
            "t": {"anyOf": [label_schema, {"type": "null"}]},
            "x": {
                "type": "string", "minLength": 1,
                "maxLength": config.DESCRIPTION_MAX_ACTION_CHARS,
            },
        },
        "required": ["s", "t", "x"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "p": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "enum": numbers},
                        "d": {
                            "type": "string", "minLength": 1,
                            "maxLength": config.DESCRIPTION_MAX_DESCRIPTION_CHARS,
                        },
                        "a": {"type": "array", "items": action, "maxItems": 8},
                    },
                    "required": ["n", "d", "a"],
                    "additionalProperties": False,
                },
                "minItems": len(numbers),
                "maxItems": len(numbers),
            },
        },
        "required": ["p"],
        "additionalProperties": False,
    }


def text_evidence_schema(expected_panels, character_labels) -> dict:
    """Exact panel-linked transcription proposals; downstream review is required."""
    numbers = sorted({int(number) for number in expected_panels})
    labels = sorted({
        row.get("label") for row in (character_labels or [])
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    })
    speaker = {"type": "null"}
    if labels:
        speaker = {"anyOf": [{"type": "string", "enum": labels}, speaker]}
    text_row = {
        "type": "object",
        "properties": {
            "k": {
                "type": "string",
                "enum": ["dialogue", "thought", "narration", "sign", "publication"],
            },
            "x": {"type": "string", "minLength": 1, "maxLength": 240},
            "s": speaker,
        },
        "required": ["k", "x", "s"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "p": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "enum": numbers},
                        "t": {"type": "array", "items": text_row, "maxItems": 32},
                    },
                    "required": ["n", "t"],
                    "additionalProperties": False,
                },
                "minItems": len(numbers),
                "maxItems": len(numbers),
            },
        },
        "required": ["p"],
        "additionalProperties": False,
    }


def transformation_response_schema(expected_panels, character_labels) -> dict:
    numbers = sorted({int(number) for number in expected_panels})
    labels = sorted({
        row.get("label") for row in (character_labels or [])
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    })
    label_schema = {"type": "string", "enum": labels}
    event = {
        "type": "object",
        "properties": {
            "t": {"enum": [
                "transformation", "reversion", "disguise_reveal", "possession",
                "summoning", "splitting", "fusion",
            ]},
            "sp": {"type": "integer", "enum": numbers},
            "tp": {"type": "integer", "enum": numbers},
            "sd": {"type": "string", "minLength": 1, "maxLength": 80},
            "td": {"type": "string", "minLength": 1, "maxLength": 80},
            "x": {"type": "string", "minLength": 1, "maxLength": 160},
            "xt": {"enum": ["visible_change", "ambiguous_sequence"]},
            "c": {"type": "integer", "minimum": 0, "maximum": 100},
            "si": label_schema,
            "ti": label_schema,
        },
        "required": ["t", "sp", "tp", "sd", "td", "x", "xt", "c", "si", "ti"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"e": {"type": "array", "items": event, "maxItems": 8}},
        "required": ["e"],
        "additionalProperties": False,
    }


_TOKENIZER_DATA = {}


def _decode_tokens(tokenizer, tokens) -> str:
    return tokenizer.decode(tokens).rstrip("\ufffd")


def _cache_file(tokenizer) -> Path:
    model_path = Path(getattr(tokenizer, "name_or_path", ""))
    if not model_path.is_dir():
        model_path = Path(config.QWEN_MODEL_PATH)
    return model_path / ".lmfe_vocab_v1.pkl"


def _tree_cache_file(tokenizer) -> Path:
    return _cache_file(tokenizer).with_name(".lmfe_tokenizer_tree_v1.pkl")


def _tokenizer_fingerprint(tokenizer) -> dict:
    return {
        "version": 1,
        "tokenizer": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "special_ids": sorted(tokenizer.all_special_ids),
    }


def _write_cache(path: Path, payload: dict) -> None:
    """Atomically persist a derived local cache; inference can continue if it fails."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _regular_tokens(tokenizer) -> list[tuple[int, str, bool]]:
    """Build or load the expensive tokenizer-to-text index used by LMFE."""
    vocab_size = len(tokenizer)
    fingerprint = _tokenizer_fingerprint(tokenizer)
    cache_path = _cache_file(tokenizer)
    if cache_path.is_file():
        try:
            cached = pickle.loads(cache_path.read_bytes())
            if cached.get("fingerprint") == fingerprint:
                return cached["regular_tokens"]
        except (OSError, EOFError, ValueError, TypeError, pickle.PickleError):
            pass

    token_zero = tokenizer.encode("0", add_special_tokens=False)[-1]
    special_ids = set(tokenizer.all_special_ids)
    regular_tokens = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        decoded_after_zero = tokenizer.decode([token_zero, token_id])[1:]
        decoded_regular = tokenizer.decode([token_id])
        regular_tokens.append(
            (token_id, decoded_after_zero, len(decoded_after_zero) > len(decoded_regular))
        )

    _write_cache(cache_path, {
        "fingerprint": fingerprint,
        "regular_tokens": regular_tokens,
    })
    return regular_tokens


def build_prefix_allowed_tokens_fn(tokenizer, expected_panels=None, character_labels=None,
                                    *, response_schema=None, complete_json_tokens=False):
    """Return a Transformers prefix filter, caching the expensive vocab scan."""
    # lm-format-enforcer 0.11 imports this class from its Transformers 4.x
    # location. Transformers 5 moved it, so expose the compatible alias before
    # importing the integration. This can be removed when upstream supports 5.x.
    import transformers.tokenization_utils as tokenization_utils
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(tokenization_utils, "PreTrainedTokenizerBase"):
        tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase

    os.environ.setdefault("LMFE_STRICT_JSON_FIELD_ORDER", "1")
    os.environ.setdefault("LMFE_MAX_CONSECUTIVE_WHITESPACES", "0")
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

    key = id(tokenizer)
    tokenizer_data = _TOKENIZER_DATA.get(key)
    if tokenizer_data is None:
        fingerprint = _tokenizer_fingerprint(tokenizer)
        tree_cache_path = _tree_cache_file(tokenizer)
        tokenizer_data = None
        if tree_cache_path.is_file():
            try:
                cached = pickle.loads(tree_cache_path.read_bytes())
                if cached.get("fingerprint") == fingerprint:
                    tokenizer_data = cached["tokenizer_data"]
            except (OSError, EOFError, ValueError, TypeError, pickle.PickleError):
                pass
        decoder = functools.partial(_decode_tokens, tokenizer)
        if tokenizer_data is None:
            tokenizer_data = TokenEnforcerTokenizerData(
                _regular_tokens(tokenizer), decoder, tokenizer.eos_token_id,
                False, len(tokenizer),
            )
            tokenizer_data.decoder = None
            _write_cache(tree_cache_path, {
                "fingerprint": fingerprint,
                "tokenizer_data": tokenizer_data,
            })
        tokenizer_data.decoder = decoder
        _TOKENIZER_DATA.clear()
        _TOKENIZER_DATA[key] = tokenizer_data

    parser = JsonSchemaParser(
        response_schema or page_response_schema(expected_panels, character_labels)
    )
    if complete_json_tokens:
        return _complete_json_token_filter(tokenizer_data, parser)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)


def _complete_json_token_filter(tokenizer_data, parser):
    """Keep LMFE's fast string cache, but also permit merged JSON boundaries."""
    from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
    from lmformatenforcer.tokenenforcer import TokenEnforcer
    from lmformatenforcer.tokenizerprefixtree import TokenizerPrefixTreeNode
    from lmformatenforcer.characterlevelparser import CharacterLevelParser, UnionParser
    from lmformatenforcer.jsonschemaparser import JsonSchemaParser, ListParsingState

    def close_empty_arrays(state):
        if isinstance(state, UnionParser):
            for branch in state.parsers:
                close_empty_arrays(branch)
        elif isinstance(state, JsonSchemaParser):
            stack = state.object_stack
            # LMFE pushes a first-item parser even when maxItems is zero.
            if len(stack) >= 2 and isinstance(stack[-2], ListParsingState) and stack[-2].max_items == 0:
                stack.pop()

    class EmptyArraySafeParser(CharacterLevelParser):
        def __init__(self, inner):
            self.inner = inner

        @property
        def config(self):
            return self.inner.config

        @config.setter
        def config(self, value):
            # A zero limit also bans spaces inside prose, not just JSON layout.
            value.max_consecutive_whitespaces = max(12, value.max_consecutive_whitespaces)
            self.inner.config = value

        def add_character(self, character):
            state = self.inner.add_character(character)
            if character == "[":
                close_empty_arrays(state)
            return EmptyArraySafeParser(state)

        def get_allowed_characters(self):
            return self.inner.get_allowed_characters()

        def can_end(self):
            return self.inner.can_end()

        def shortcut_key(self):
            return self.inner.shortcut_key()

    boundary_tree = getattr(tokenizer_data, "json_boundary_tree", None)
    if boundary_tree is None:
        boundary_tree = TokenizerPrefixTreeNode()
        for token_id, text, _ in tokenizer_data.regular_tokens:
            # LMFE's freetext shortcut skips tokens such as ."," and ."}.
            if text.startswith('"') or '"' not in text[:-1]:
                continue
            node = boundary_tree
            for character in text:
                node = node.children.setdefault(character, TokenizerPrefixTreeNode())
            node.tokens.append(token_id)
        tokenizer_data.json_boundary_tree = boundary_tree

    class CompleteJsonTokenEnforcer(TokenEnforcer):
        def _collect_allowed_tokens(self, parser, tree_node, allowed_tokens, shortcut_key):
            super()._collect_allowed_tokens(parser, tree_node, allowed_tokens, shortcut_key)
            if isinstance(shortcut_key, tuple) and shortcut_key[0] == "json_freetext":
                super()._collect_allowed_tokens(parser, boundary_tree, allowed_tokens, None)

    return TransformersPrefixAllowedTokensFn(CompleteJsonTokenEnforcer(tokenizer_data, EmptyArraySafeParser(parser)))
