"""Small compatibility fixes for loading local model checkpoints safely."""

from __future__ import annotations

import sys


def _patch_safe_open(module) -> bool:
    """Make one Transformers safetensors alias use pread instead of Windows mmap."""
    original = getattr(module, "safe_open", None)
    if original is None or getattr(original, "_manga_windows_pread", False):
        return False

    def safe_open_windows_pread(*args, **kwargs):
        # Windows charges copy-on-write mmap against system commit even for
        # read-only model weights. The safetensors pread backend is lazy without
        # that reservation and is the upstream Transformers fix for this issue.
        if kwargs.get("backend", "mmap") == "mmap":
            kwargs["backend"] = "pread"
        return original(*args, **kwargs)

    safe_open_windows_pread._manga_windows_pread = True
    module.safe_open = safe_open_windows_pread
    return True


def configure_windows_safetensors_loader() -> bool:
    """Apply the upstream pread behavior missing from some Transformers builds."""
    if sys.platform != "win32":
        return False

    import transformers.modeling_utils as modeling_utils

    changed = _patch_safe_open(modeling_utils)
    # MTP-capable models have a second module-level safe_open alias. Qwen3.5-2B
    # does not currently need it, but patching it mirrors the complete upstream
    # Windows fix and avoids a future regression if the architecture changes.
    try:
        import transformers.modeling_layers as modeling_layers
    except (ImportError, AttributeError):
        modeling_layers = None
    if modeling_layers is not None:
        changed = _patch_safe_open(modeling_layers) or changed
    return changed
