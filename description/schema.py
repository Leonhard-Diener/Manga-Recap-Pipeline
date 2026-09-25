"""Compact, mechanically enforced response schema for page description."""

from __future__ import annotations

import functools
import os
import pickle
from pathlib import Path

import config


def page_response_schema(expected_panels=None) -> dict:
    """Build the JSON schema used for one page or a homogeneous page batch."""
    panel_numbers = None
    if expected_panels is not None:
        panel_numbers = sorted({int(number) for number in expected_panels})
        if not panel_numbers:
            raise ValueError("expected_panels must not be empty")

    number_schema = {"type": "integer", "minimum": 1}
    if panel_numbers is not None:
        number_schema["enum"] = panel_numbers

    panel_items = {
        "type": "object",
        "properties": {
            "n": number_schema,
            "d": {
                "type": "string",
                "minLength": 1,
                "maxLength": config.DESCRIPTION_MAX_DESCRIPTION_CHARS,
            },
        },
        "required": ["n", "d"],
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
    return {
        "type": "object",
        "properties": {
            "r": {"enum": ["story", "reference", "extra", "uncertain"]},
            "p": panels_schema,
            "e": {"type": "array", "items": event_items, "maxItems": 8},
        },
        "required": ["r", "p", "e"],
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


def build_prefix_allowed_tokens_fn(tokenizer, expected_panels=None):
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

    parser = JsonSchemaParser(page_response_schema(expected_panels))
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
