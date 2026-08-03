import glob
import os
import re
from collections import OrderedDict

import torch


_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


def _resolve_project_path(path):
    """Resolve a user-supplied path without embedding machine-specific roots."""
    expanded_path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded_path):
        return os.path.normpath(expanded_path)
    return os.path.normpath(os.path.join(_PROJECT_ROOT, expanded_path))


def get_isaac_pretrained_cfg(hypes):
    cfg = hypes.get("isaac_pretrained", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return {}
    return cfg


def resolve_isaac_checkpoint(path, checkpoint_mode="bestval"):
    if not path:
        raise ValueError("isaac_pretrained.path is required when enabled")
    path = _resolve_project_path(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Isaac pretrained path not found: {path}")

    if checkpoint_mode == "bestval":
        candidates = glob.glob(os.path.join(path, "net_epoch_bestval_at*.pth"))
        if candidates:
            if len(candidates) > 1:
                raise ValueError(f"Multiple bestval checkpoints found in {path}")
            return candidates[0]

    candidates = []
    for checkpoint_path in glob.glob(os.path.join(path, "net_epoch*.pth")):
        filename = os.path.basename(checkpoint_path)
        if filename.startswith("net_epoch_bestval_at"):
            continue
        match = re.match(r"net_epoch(\d+)\.pth$", filename)
        if match:
            candidates.append((int(match.group(1)), checkpoint_path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    raise FileNotFoundError(f"No checkpoint found in {path}")


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _starts_with_any(name, prefixes):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _remap_key(key, key_remap):
    for old_prefix, new_prefix in key_remap.items():
        if key == old_prefix:
            return new_prefix
        if key.startswith(old_prefix + "."):
            return new_prefix + key[len(old_prefix):]
    return key


def load_isaac_pretrained_weights(model, hypes, context="train"):
    cfg = get_isaac_pretrained_cfg(hypes)
    if not cfg:
        return OrderedDict()

    checkpoint_path = resolve_isaac_checkpoint(
        cfg.get("path", ""), cfg.get("checkpoint_mode", "bestval")
    )
    load_prefixes = _as_list(cfg.get("load_prefixes"))
    key_remap = OrderedDict(cfg.get("key_remap", {}))
    model_state = model.state_dict()
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    filtered_state = OrderedDict()
    skipped_shape = []
    skipped_prefix = 0
    skipped_missing = 0

    for source_key, value in checkpoint_state.items():
        target_key = _remap_key(source_key, key_remap)
        if load_prefixes and not _starts_with_any(target_key, load_prefixes):
            skipped_prefix += 1
            continue
        if target_key not in model_state:
            skipped_missing += 1
            continue
        if hasattr(value, "shape") and hasattr(model_state[target_key], "shape"):
            if tuple(value.shape) != tuple(model_state[target_key].shape):
                skipped_shape.append((source_key, target_key, tuple(value.shape), tuple(model_state[target_key].shape)))
                continue
        filtered_state[target_key] = value

    model.load_state_dict(filtered_state, strict=False)
    summary = OrderedDict(
        [
            ("checkpoint_path", checkpoint_path),
            ("loaded_keys", len(filtered_state)),
            ("skipped_by_prefix", skipped_prefix),
            ("skipped_missing", skipped_missing),
            ("skipped_shape", len(skipped_shape)),
        ]
    )
    print(
        f"Isaac {context} pretrained load: {len(filtered_state)} keys from "
        f"{checkpoint_path}"
    )
    if skipped_shape:
        print("Isaac pretrained skipped shape-mismatch keys:")
        for source_key, target_key, source_shape, target_shape in skipped_shape[:20]:
            print(f"  {source_key} -> {target_key}: {source_shape} vs {target_shape}")
        if len(skipped_shape) > 20:
            print(f"  ... {len(skipped_shape) - 20} more")
    return summary


def apply_isaac_freeze_config(model, hypes):
    cfg = get_isaac_pretrained_cfg(hypes)
    if not cfg:
        return OrderedDict()

    freeze_prefixes = _as_list(cfg.get("freeze_prefixes"))
    if not freeze_prefixes:
        return OrderedDict()

    frozen_params = 0
    frozen_tensors = 0
    for name, param in model.named_parameters():
        if _starts_with_any(name, freeze_prefixes):
            param.requires_grad_(False)
            frozen_params += param.numel()
            frozen_tensors += 1

    print(
        "Isaac pretrained freeze: "
        f"{frozen_tensors} tensors / {frozen_params} parameters frozen "
        f"for prefixes {freeze_prefixes}"
    )
    return OrderedDict(
        [
            ("freeze_prefixes", freeze_prefixes),
            ("frozen_tensors", frozen_tensors),
            ("frozen_params", frozen_params),
        ]
    )


def set_isaac_frozen_modules_eval(model, hypes):
    cfg = get_isaac_pretrained_cfg(hypes)
    if not cfg or not cfg.get("keep_frozen_modules_eval", True):
        return

    module_prefixes = _as_list(cfg.get("frozen_module_prefixes"))
    if not module_prefixes:
        module_prefixes = _as_list(cfg.get("freeze_prefixes"))
    if not module_prefixes:
        return

    for name, module in model.named_modules():
        if _starts_with_any(name, module_prefixes):
            module.eval()
