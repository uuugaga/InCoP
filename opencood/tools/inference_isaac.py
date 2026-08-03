import argparse
import atexit
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict, defaultdict
import importlib
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils import eval_utils_isaac
from opencood.utils import pcd_utils
from opencood.visualization import simple_vis_isaac
from opencood.utils.common_utils import update_dict
from opencood.utils import box_utils, common_utils
from opencood.utils.transformation_utils import x1_to_x2
torch.multiprocessing.set_sharing_strategy("file_descriptor")

IOU_THRESHOLDS = (0.25, 0.3, 0.4, 0.5, 0.7)
PAPER_MAP_THRESHOLDS = (0.25, 0.3, 0.4, 0.5)
PAPER_SIZE_GROUPS = OrderedDict([
    ("small", ("fire_extinguisher", "trash_can", "traffic_cone", "wet_floor_sign")),
    ("non_small", ("chair", "medical_bag", "potted_plant")),
])
PAPER_CASE_TYPES = ("distance", "occlusion")
INDOOR_TP_CENTER_DISTANCE = 0.5
VISIBILITY_SUBSETS = ("ego_only", "partner_only", "shared")
VISIBILITY_IOU_THRESHOLDS = (0.3, 0.5)


def parse_bool_arg(value):
    """Parse an explicit command-line boolean value."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value, got {value!r}"
    )


def apply_cli_shortcuts(opt):
    """Expand high-level CLI modes into their concrete inference options."""
    if opt.video_compare_fusion:
        if opt.fusion_method != 'intermediate':
            print(
                'Video fusion comparison: overriding fusion_method '
                f'{opt.fusion_method!r} -> "intermediate".'
            )
        opt.fusion_method = 'intermediate'
        if not opt.stream_video_output:
            opt.stream_video_output = os.path.join(
                opt.model_dir, 'video_compare_fusion.mp4'
            )
        opt.video_only = True
    return opt


def parse_sample_indices(indices_text):
    if not indices_text:
        return None
    indices = []
    for item in indices_text.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            parts = item.split(':')
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid sample index range: {item}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step <= 0:
                raise ValueError(f"Invalid non-positive sample index step: {item}")
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(item))
    ordered = []
    seen = set()
    for idx in indices:
        if idx < 0:
            raise ValueError(f"Sample index must be non-negative: {idx}")
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered


def collect_model_debug_features(output_dict, include_co_feature=False):
    """Collect optional BEV/head debug tensors emitted by model forward."""
    features = {}
    if not isinstance(output_dict, dict):
        return features

    ego_output = output_dict.get("ego")
    if isinstance(ego_output, dict):
        ego_debug = ego_output.get("debug_features", {})
        if isinstance(ego_debug, dict):
            features.update(ego_debug)

    if not include_co_feature:
        features.pop("co_bev_feature", None)
        features.pop("co_bev_feature_raw", None)
    return features


def _feature_tensor_to_energy_map(feature):
    if feature is None:
        return None
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().float().cpu().numpy()
    else:
        feature = np.asarray(feature, dtype=np.float32)

    feature = np.squeeze(feature)
    if feature.ndim == 3:
        fmap = np.sqrt(np.mean(np.square(feature), axis=0))
    elif feature.ndim == 2:
        fmap = feature
    else:
        return None

    fmap = np.nan_to_num(fmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Match Isaac BEV canvas orientation, which uses flip_y_axis=True.
    return np.flipud(fmap)


def _head_prediction_to_response_map(head_pred):
    if head_pred is None:
        return None
    if isinstance(head_pred, torch.Tensor):
        head_pred = head_pred.detach().float().cpu()
        response = torch.sigmoid(head_pred)
        while response.ndim > 3:
            response = response[0]
        if response.ndim == 3:
            response = response.max(dim=0).values
        elif response.ndim != 2:
            return None
        fmap = response.numpy()
    else:
        response = 1.0 / (1.0 + np.exp(-np.asarray(head_pred, dtype=np.float32)))
        response = np.squeeze(response)
        while response.ndim > 3:
            response = response[0]
        if response.ndim == 3:
            response = response.max(axis=0)
        elif response.ndim != 2:
            return None
        fmap = response

    fmap = np.nan_to_num(fmap.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.flipud(np.clip(fmap, 0.0, 1.0))


def _add_center_head_prediction_debug(debug_features, output_dict):
    if not isinstance(debug_features, dict) or not isinstance(output_dict, dict):
        return debug_features
    ego_output = output_dict.get("ego")
    if not isinstance(ego_output, dict):
        return debug_features
    center_head_preds = ego_output.get("center_head_preds", {})
    if not isinstance(center_head_preds, dict):
        return debug_features
    hm = center_head_preds.get("hm")
    if hm is not None:
        debug_features["ego_head_prediction"] = hm.detach().float().cpu()
    return debug_features


def _normalize_feature_map(fmap, lo, hi):
    if fmap is None:
        return None
    if hi <= lo:
        return np.zeros_like(fmap, dtype=np.float32)
    return np.clip((fmap - lo) / (hi - lo), 0.0, 1.0)


def _colorize_feature_map(fmap):
    try:
        import matplotlib.cm as cm
        rgba = cm.get_cmap("viridis")(fmap)
        return (rgba[:, :, :3] * 255.0).astype(np.uint8)
    except Exception:
        # Lightweight fallback close to a blue-green-yellow heatmap.
        x = np.clip(fmap, 0.0, 1.0)
        r = np.clip(1.5 * x - 0.35, 0.0, 1.0)
        g = np.clip(1.5 - np.abs(2.0 * x - 1.0), 0.0, 1.0)
        b = np.clip(1.0 - 1.4 * x, 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _resize_feature_map_to_match(fmap, target_shape):
    if fmap.shape == target_shape:
        return fmap
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = Image.fromarray(fmap.astype(np.float32), mode="F")
    image = image.resize((target_shape[1], target_shape[0]), resampling)
    return np.asarray(image, dtype=np.float32)


def _compose_ego_co_feature_overlay(ego_norm, co_norm):
    co_norm = _resize_feature_map_to_match(co_norm, ego_norm.shape)
    ego_norm = np.clip(ego_norm, 0.0, 1.0)
    co_norm = np.clip(co_norm, 0.0, 1.0)
    overlay = np.stack([ego_norm, co_norm, co_norm], axis=-1)
    return (overlay * 255.0).astype(np.uint8)


def save_isaac_feature_debug_views(debug_features, sample_vis_dir, bev_vis_index, opt):
    if not debug_features:
        return
    feature_dir = os.path.join(sample_vis_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)
    meta = OrderedDict()
    meta["feature_vis_method"] = OrderedDict(
        [
            ("bev_features", "channel_l2"),
            ("ego_bev_feature_raw", "ego raw BEV feature before message-backbone downsampling"),
            ("ego_bev_feature", "ego BEV feature after message-backbone downsampling"),
            ("co_bev_feature_raw", "first non-ego raw BEV feature before message-backbone downsampling"),
            ("co_bev_feature", "first non-ego BEV feature after downsampling and pairwise warp to ego frame"),
            ("fusion_bev_feature", "ego-frame fused BEV feature before shared decoder"),
            ("decoder_bev_feature", "decoded fused BEV feature before prediction head"),
            ("ego_co_bev_feature_overlay", "RGB downsampled-BEV overlay, red=ego and cyan=noisy-warped co"),
            ("head_prediction", "sigmoid + max over classes"),
        ]
    )
    meta["feature_vis_cmap"] = "viridis"
    meta["normalization"] = (
        "BEV features use shared per-sample 2nd-98th percentile clipping + "
        "min-max; head prediction uses native 0-1 responses."
    )
    meta["orientation"] = "y-axis flipped to match Isaac BEV canvas"
    meta["features"] = OrderedDict()
    meta["overlays"] = OrderedDict()

    feature_maps = OrderedDict()
    for feature_name in (
        "ego_bev_feature_raw",
        "ego_bev_feature",
        "co_bev_feature_raw",
        "co_bev_feature",
        "fusion_bev_feature",
        "decoder_bev_feature",
    ):
        if feature_name not in debug_features:
            continue
        feature = debug_features[feature_name]
        fmap = _feature_tensor_to_energy_map(feature)
        if fmap is None:
            continue
        shape = list(feature.shape) if hasattr(feature, "shape") else list(np.asarray(feature).shape)
        feature_maps[feature_name] = (shape, fmap, False)

    if "ego_head_prediction" in debug_features:
        feature = debug_features["ego_head_prediction"]
        fmap = _head_prediction_to_response_map(feature)
        if fmap is not None:
            shape = list(feature.shape) if hasattr(feature, "shape") else list(np.asarray(feature).shape)
            feature_maps["ego_head_prediction"] = (shape, fmap, True)

    if not feature_maps:
        return

    finite_values = [
        fmap[np.isfinite(fmap)]
        for _, fmap, already_normalized in feature_maps.values()
        if not already_normalized
    ]
    finite_values = [values for values in finite_values if values.size > 0]
    if finite_values:
        shared_values = np.concatenate(finite_values)
        lo, hi = np.percentile(shared_values, [2.0, 98.0])
        if hi <= lo:
            lo, hi = float(shared_values.min()), float(shared_values.max())
    else:
        lo, hi = 0.0, 0.0
    meta["normalization_lo"] = float(lo)
    meta["normalization_hi"] = float(hi)

    normalized_maps = OrderedDict()
    for feature_name, (shape, fmap, already_normalized) in feature_maps.items():
        if already_normalized:
            normalized = np.clip(fmap, 0.0, 1.0)
        else:
            normalized = _normalize_feature_map(fmap, lo, hi)
        normalized_maps[feature_name] = normalized
        image = _colorize_feature_map(normalized)
        image_path = os.path.join(feature_dir, f"{feature_name}_{bev_vis_index}.png")
        Image.fromarray(image).save(image_path)
        finite = fmap[np.isfinite(fmap)]
        entry = OrderedDict()
        entry["shape"] = shape
        entry["png"] = os.path.basename(image_path)
        entry["already_normalized"] = bool(already_normalized)
        if finite.size > 0:
            entry["raw_min"] = float(finite.min())
            entry["raw_max"] = float(finite.max())
            entry["raw_p2"] = float(np.percentile(finite, 2.0))
            entry["raw_p98"] = float(np.percentile(finite, 98.0))
        meta["features"][feature_name] = entry

    if "ego_bev_feature" in normalized_maps and "co_bev_feature" in normalized_maps:
        overlay_name = "ego_co_bev_feature_overlay"
        overlay_image = _compose_ego_co_feature_overlay(
            normalized_maps["ego_bev_feature"],
            normalized_maps["co_bev_feature"],
        )
        overlay_path = os.path.join(feature_dir, f"{overlay_name}_{bev_vis_index}.png")
        Image.fromarray(overlay_image).save(overlay_path)
        meta["overlays"][overlay_name] = OrderedDict(
            [
                ("png", os.path.basename(overlay_path)),
                ("red", "ego_bev_feature"),
                ("cyan", "co_bev_feature after downsampling and pairwise warp"),
            ]
        )

    yaml_utils.save_yaml(meta, os.path.join(feature_dir, f"feature_meta_{bev_vis_index}.yaml"))


def _method_scalar_to_bev_map(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 2:
        return None
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.flipud(np.clip(array, 0.0, 1.0))


def _opg_assignment_to_bev(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        assignment = value.detach().float().cpu().numpy()
    else:
        assignment = np.asarray(value, dtype=np.float32)
    while assignment.ndim > 3 and assignment.shape[0] == 1:
        assignment = assignment[0]
    if assignment.ndim != 3:
        return None
    assignment = np.nan_to_num(
        assignment.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0
    )
    return np.flip(np.clip(assignment, 0.0, 1.0), axis=1)


def _opg_palette(num_prototypes):
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab10", max(num_prototypes, 2))
        rgba = cmap(np.arange(num_prototypes))
        return (rgba[:, :3] * 255.0).astype(np.uint8)
    except Exception:
        base = np.asarray(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [188, 189, 34],
            ],
            dtype=np.uint8,
        )
        repeats = int(np.ceil(float(num_prototypes) / float(len(base))))
        return np.tile(base, (repeats, 1))[:num_prototypes]


def _module_debug_to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    else:
        value = np.asarray(value, dtype=np.float32)
    return np.nan_to_num(
        np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def save_isaac_module_feature_views(
    debug_features, sample_vis_dir, bev_vis_index, opt
):
    """Export internal tensors of the proposed fusion module for inference."""
    del opt
    scalar_specs = OrderedDict(
        [
            ("Q_ego", "support_Q_ego"),
            ("Q_partner", "support_Q_partner"),
            ("M_ego", "support_M_ego"),
            ("M_partner", "support_M_partner"),
            ("C_partner", "complementarity_C_partner"),
        ]
    )
    residual_specs = OrderedDict(
        [
            ("delta_local_window", "delta_local_window"),
            ("delta_local_shifted", "delta_local_shifted"),
            ("delta_local", "delta_local"),
        ]
    )
    module_debug_keys = set(scalar_specs.values()) | set(residual_specs.values())
    if not any(key in debug_features for key in module_debug_keys):
        return

    module_dir = os.path.join(sample_vis_dir, "module_features")
    os.makedirs(module_dir, exist_ok=True)
    meta = OrderedDict()
    meta["orientation"] = "PNG y-axis is flipped to match the Isaac BEV canvas"
    meta["raw_npz_orientation"] = "native model tensor orientation; no y-axis flip"
    meta["scalar_maps"] = OrderedDict()
    meta["residual_maps"] = OrderedDict()
    raw_arrays = OrderedDict()

    for output_name, debug_name in scalar_specs.items():
        if debug_name not in debug_features:
            continue
        raw = np.squeeze(_module_debug_to_numpy(debug_features[debug_name]))
        if raw.ndim != 2:
            continue
        raw_arrays[output_name] = raw
        fmap = np.flipud(np.clip(raw, 0.0, 1.0))
        image_path = os.path.join(
            module_dir, f"{output_name}_{bev_vis_index:05d}.png"
        )
        Image.fromarray(_colorize_feature_map(fmap)).save(image_path)
        meta["scalar_maps"][output_name] = OrderedDict(
            [
                ("source", debug_name),
                ("shape", list(raw.shape)),
                ("png", os.path.basename(image_path)),
                ("min", float(raw.min())),
                ("max", float(raw.max())),
                ("mean", float(raw.mean())),
            ]
        )

    residual_maps = OrderedDict()
    for output_name, debug_name in residual_specs.items():
        if debug_name not in debug_features:
            continue
        raw = np.squeeze(_module_debug_to_numpy(debug_features[debug_name]))
        if raw.ndim != 3:
            continue
        raw_arrays[output_name] = raw
        energy = np.sqrt(np.mean(np.square(raw), axis=0)).astype(np.float32)
        residual_maps[output_name] = (raw, np.flipud(energy))

    if residual_maps:
        finite_energy = np.concatenate(
            [energy.reshape(-1) for _, energy in residual_maps.values()]
        )
        residual_hi = float(np.percentile(finite_energy, 99.0))
        if residual_hi <= 0.0:
            residual_hi = float(finite_energy.max())
        meta["residual_normalization"] = OrderedDict(
            [
                ("method", "shared channel-RMS energy clipped at pooled p99"),
                ("lo", 0.0),
                ("hi", residual_hi),
            ]
        )
        for output_name, (raw, energy) in residual_maps.items():
            normalized = _normalize_feature_map(energy, 0.0, residual_hi)
            image_path = os.path.join(
                module_dir, f"{output_name}_{bev_vis_index:05d}.png"
            )
            Image.fromarray(_colorize_feature_map(normalized)).save(image_path)
            native_energy = np.flipud(energy)
            meta["residual_maps"][output_name] = OrderedDict(
                [
                    ("shape", list(raw.shape)),
                    ("visualization", "channel RMS magnitude"),
                    ("png", os.path.basename(image_path)),
                    ("energy_min", float(native_energy.min())),
                    ("energy_max", float(native_energy.max())),
                    ("energy_mean", float(native_energy.mean())),
                    ("tensor_l2", float(np.linalg.norm(raw))),
                ]
            )

    if raw_arrays:
        npz_path = os.path.join(
            module_dir, f"module_features_{bev_vis_index:05d}.npz"
        )
        np.savez_compressed(npz_path, **raw_arrays)
        meta["raw_npz"] = os.path.basename(npz_path)
    yaml_utils.save_yaml(
        meta,
        os.path.join(
            module_dir, f"module_feature_meta_{bev_vis_index:05d}.yaml"
        ),
    )


def save_isaac_method_debug_views(debug_features, sample_vis_dir, bev_vis_index, opt):
    method_keys = OrderedDict(
        [
            ("A_ego", "raw_A_ego"),
            ("A_partner", "raw_A_partner"),
            ("R_ego", "range_R_ego"),
            ("R_partner", "range_R_partner"),
            ("W_ego", "agent_W_ego"),
            ("W_partner", "agent_W_partner"),
        ]
    )
    if not any(key in debug_features for key in method_keys.values()) and (
        "opg_assignment_pi" not in debug_features
    ):
        return

    method_dir = os.path.join(sample_vis_dir, "features_method")
    os.makedirs(method_dir, exist_ok=True)
    meta = OrderedDict()
    meta["normalization"] = "fixed [0, 1]; no percentile or per-sample rescaling"
    meta["orientation"] = "y-axis flipped to match Isaac BEV canvas"
    meta["maps"] = OrderedDict()
    raw_arrays = OrderedDict()

    for output_name, debug_name in method_keys.items():
        if debug_name not in debug_features:
            continue
        fmap = _method_scalar_to_bev_map(debug_features[debug_name])
        if fmap is None:
            continue
        raw_arrays[output_name] = fmap
        image_path = os.path.join(
            method_dir, f"{output_name}_{bev_vis_index:05d}.png"
        )
        Image.fromarray(_colorize_feature_map(fmap)).save(image_path)
        meta["maps"][output_name] = OrderedDict(
            [
                ("symbol", output_name.split("_")[0]),
                ("agent", output_name.split("_", 1)[1]),
                ("png", os.path.basename(image_path)),
                ("min", float(fmap.min())),
                ("max", float(fmap.max())),
                ("mean", float(fmap.mean())),
            ]
        )

    if "W_ego" in raw_arrays and "W_partner" in raw_arrays:
        weight_sum = raw_arrays["W_ego"] + raw_arrays["W_partner"]
        meta["W_sum_max_abs_error"] = float(np.max(np.abs(weight_sum - 1.0)))

    assignment = _opg_assignment_to_bev(debug_features.get("opg_assignment_pi"))
    if assignment is not None:
        num_prototypes = int(assignment.shape[0])
        assignment_sum = assignment.sum(axis=0)
        argmax_map = np.argmax(assignment, axis=0)
        max_probability = np.max(assignment, axis=0)
        uniform_probability = 1.0 / float(max(num_prototypes, 1))
        certainty = np.clip(
            (max_probability - uniform_probability)
            / max(1.0 - uniform_probability, 1.0e-6),
            0.0,
            1.0,
        )
        palette = _opg_palette(num_prototypes)
        argmax_rgb = palette[argmax_map]
        certainty_rgb = (
            255.0
            - certainty[:, :, None]
            * (255.0 - argmax_rgb.astype(np.float32))
        ).astype(np.uint8)

        argmax_path = os.path.join(
            method_dir, f"OPG_argmax_{bev_vis_index:05d}.png"
        )
        certainty_path = os.path.join(
            method_dir, f"OPG_argmax_certainty_{bev_vis_index:05d}.png"
        )
        confidence_path = os.path.join(
            method_dir, f"OPG_assignment_confidence_{bev_vis_index:05d}.png"
        )
        Image.fromarray(argmax_rgb).save(argmax_path)
        Image.fromarray(certainty_rgb).save(certainty_path)
        Image.fromarray(_colorize_feature_map(certainty)).save(confidence_path)

        usage = assignment.mean(axis=(1, 2))
        legend = OrderedDict()
        for prototype_idx in range(num_prototypes):
            color = palette[prototype_idx]
            legend[f"prototype_{prototype_idx}"] = OrderedDict(
                [
                    ("rgb", [int(v) for v in color]),
                    ("usage", float(usage[prototype_idx])),
                ]
            )
        meta["opg"] = OrderedDict(
            [
                ("num_prototypes", num_prototypes),
                ("argmax_png", os.path.basename(argmax_path)),
                ("certainty_weighted_argmax_png", os.path.basename(certainty_path)),
                ("assignment_confidence_png", os.path.basename(confidence_path)),
                ("certainty_definition", "(max(pi)-1/K)/(1-1/K)"),
                ("sum_to_one_max_abs_error", float(np.max(np.abs(assignment_sum - 1.0)))),
                ("legend", legend),
            ]
        )
        raw_arrays["opg_assignment_pi"] = assignment
        raw_arrays["opg_argmax"] = argmax_map.astype(np.int16)
        raw_arrays["opg_certainty"] = certainty

    if raw_arrays:
        npz_path = os.path.join(
            method_dir, f"method_features_{bev_vis_index:05d}.npz"
        )
        np.savez_compressed(npz_path, **raw_arrays)
        meta["raw_npz"] = os.path.basename(npz_path)

    yaml_utils.save_yaml(
        meta,
        os.path.join(method_dir, f"method_feature_meta_{bev_vis_index:05d}.yaml"),
    )


def is_isaac_center_head(hypes):
    loss_name = hypes.get("loss", {}).get("core_method", "")
    return loss_name == "center_head_loss"


def _uses_late_pose_correction(model):
    return bool(getattr(model, "late_pose_correction", False))


def _apply_corrected_late_transforms(data_dict, output_dict):
    corrected = output_dict.get("__corrected_transformation_matrices")
    if not corrected:
        return data_dict

    post_data = OrderedDict()
    for cav_id, cav_content in data_dict.items():
        if cav_id in corrected:
            updated = dict(cav_content)
            updated["transformation_matrix"] = corrected[cav_id].to(
                device=cav_content["transformation_matrix"].device,
                dtype=cav_content["transformation_matrix"].dtype,
            )
            post_data[cav_id] = updated
        else:
            post_data[cav_id] = cav_content
    return post_data


def _extract_communication_stats(output_dict):
    if not isinstance(output_dict, dict):
        return None
    ego_output = output_dict.get("ego", output_dict)
    if not isinstance(ego_output, dict):
        return None
    stats = ego_output.get("communication")
    return dict(stats) if isinstance(stats, dict) else None


def summarize_communication_stats(records):
    if not records:
        return None
    additive_keys = (
        "feature_bytes", "index_bytes", "metadata_bytes", "total_bytes",
        "message_count", "feature_elements", "index_elements", "metadata_elements",
    )
    summary = OrderedDict()
    summary["method"] = records[0].get("method")
    summary["mode"] = records[0].get("mode")
    summary["num_samples"] = len(records)
    for key in additive_keys:
        summary[key] = sum(float(record.get(key, 0)) for record in records)
    summary["total_KB"] = summary["total_bytes"] / 1000.0
    summary["total_MB"] = summary["total_bytes"] / 1_000_000.0
    summary["mean_per_sample_KB"] = summary["total_KB"] / len(records)
    summary["mean_per_sample_MB"] = summary["total_MB"] / len(records)
    dense_bytes = sum(float(record.get("dense_feature_bytes", 0)) for record in records)
    if dense_bytes > 0:
        summary["dense_feature_bytes"] = dense_bytes
        summary["feature_payload_ratio_to_dense"] = summary["feature_bytes"] / dense_bytes
    summary["accounting"] = (
        "collaborator-to-ego payload; ego-local tensor excluded; feature, sparse index, "
        "and metadata/control bytes reported separately"
    )
    return summary


def inference_isaac_center_head(batch_data, model, postprocess_params, fusion_method):
    """CenterHead inference path that preserves predicted/GT class ids."""
    if fusion_method == "late":
        if _uses_late_pose_correction(model):
            output_dict = model(batch_data)
            post_data = _apply_corrected_late_transforms(batch_data, output_dict)
        else:
            output_dict = OrderedDict()
            for cav_id, cav_content in batch_data.items():
                output_dict[cav_id] = model(cav_content)
            post_data = batch_data
    elif fusion_method in ("no", "intermediate"):
        output_dict = OrderedDict()
        output_dict["ego"] = model(batch_data["ego"])
        post_data = batch_data
    else:
        raise NotImplementedError(
            "Isaac CenterHead class-aware inference supports no, late, "
            "and intermediate fusion."
        )

    pred_box_tensor, pred_score, pred_label, gt_box_tensor, gt_label = (
        eval_utils_isaac.center_head_post_process_isaac(
            post_data, output_dict, postprocess_params
        )
    )
    debug_features = collect_model_debug_features(
        output_dict, include_co_feature=(fusion_method == "intermediate")
    )
    _add_center_head_prediction_debug(debug_features, output_dict)
    return {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "pred_label": pred_label,
        "gt_box_tensor": gt_box_tensor,
        "gt_label": gt_label,
        "debug_features": debug_features,
        "communication": _extract_communication_stats(output_dict),
    }


def _format_float_tag(value):
    if value is None:
        return "none"
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _evaluation_dataset_tag(dataset_dir):
    """Build a concise, filesystem-safe tag for evaluation outputs."""
    normalized = os.path.normpath(str(dataset_dir))
    basename = os.path.basename(normalized)
    if basename.lower() in {"test", "val", "validate", "validation"}:
        basename = os.path.basename(os.path.dirname(normalized))
    for prefix in ("IsaacSimOPV2V_", "IsaacSim_"):
        if basename.startswith(prefix):
            basename = basename[len(prefix):]
            break
    tag = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in basename
    ).strip("_")
    return tag or "custom"


def _prepare_evaluation_modality_assignment(hypes, evaluation_dataset_dir):
    """Extend a saved heter assignment for renamed high-overlap scenes."""
    heter_params = hypes.get("heter")
    if not isinstance(heter_params, dict):
        return None
    assignment_path = heter_params.get("assignment_path")
    if not assignment_path:
        return None

    resolved_assignment_path = os.path.abspath(os.path.expanduser(assignment_path))
    with open(resolved_assignment_path, "r") as assignment_file:
        modality_assignment = json.load(
            assignment_file, object_pairs_hook=OrderedDict
        )

    scenario_names = sorted(
        entry
        for entry in os.listdir(evaluation_dataset_dir)
        if os.path.isdir(os.path.join(evaluation_dataset_dir, entry))
    )
    missing_scenarios = [
        scenario_name
        for scenario_name in scenario_names
        if scenario_name not in modality_assignment
    ]
    if not missing_scenarios:
        return None

    copied_from = OrderedDict()
    exact_match_count = 0
    uniform_fallback_count = 0
    for scenario_name in missing_scenarios:
        source_name = scenario_name.replace(
            "__dual_case_highoverlap__", "__dual_case_distance__"
        )
        scenario_dir = os.path.join(evaluation_dataset_dir, scenario_name)
        dataset_cav_ids = sorted(
            entry
            for entry in os.listdir(scenario_dir)
            if os.path.isdir(os.path.join(scenario_dir, entry))
        )

        if source_name != scenario_name and source_name in modality_assignment:
            assignment_cav_ids = sorted(modality_assignment[source_name].keys())
            if dataset_cav_ids != assignment_cav_ids:
                raise ValueError(
                    "Cannot reuse heter modality assignment for scenario "
                    f"'{scenario_name}': dataset CAV IDs {dataset_cav_ids} differ "
                    f"from source '{source_name}' IDs {assignment_cav_ids}."
                )
            selected_assignment = modality_assignment[source_name]
            copied_from[scenario_name] = source_name
            exact_match_count += 1
        else:
            compatible_assignments = OrderedDict()
            for existing_assignment in modality_assignment.values():
                if sorted(existing_assignment.keys()) != dataset_cav_ids:
                    continue
                signature = tuple(
                    (cav_id, existing_assignment[cav_id])
                    for cav_id in dataset_cav_ids
                )
                compatible_assignments[signature] = existing_assignment
            if len(compatible_assignments) != 1:
                raise KeyError(
                    "The heter modality assignment has no exact source for "
                    f"evaluation scenario '{scenario_name}', and assignments "
                    f"for CAV IDs {dataset_cav_ids} are not uniform "
                    f"({len(compatible_assignments)} distinct settings)."
                )
            selected_assignment = next(iter(compatible_assignments.values()))
            copied_from[scenario_name] = "uniform_existing_assignment"
            uniform_fallback_count += 1

        modality_assignment[scenario_name] = OrderedDict(selected_assignment)

    temporary_assignment = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="inference_isaac_modality_assignment_",
        suffix=".json",
        delete=False,
    )
    try:
        json.dump(modality_assignment, temporary_assignment, indent=2)
        temporary_assignment.flush()
    finally:
        temporary_assignment.close()
    atexit.register(
        lambda path=temporary_assignment.name: (
            os.remove(path) if os.path.exists(path) else None
        )
    )
    heter_params["assignment_path"] = temporary_assignment.name
    info = OrderedDict(
        [
            ("original_assignment_path", resolved_assignment_path),
            ("copied_scene_count", len(copied_from)),
            ("exact_distance_match_count", exact_match_count),
            ("uniform_fallback_count", uniform_fallback_count),
            (
                "mapping_rule",
                "same-index distance scene; unique existing assignment fallback",
            ),
        ]
    )
    print(
        "Extended heter modality assignment for evaluation dataset: "
        f"{len(copied_from)} scene(s), rule={info['mapping_rule']}"
    )
    return info


def format_iou_metric_key(value):
    return f"{float(value):g}"


def map_key_for_iou(value):
    return f"mAP@{format_iou_metric_key(value)}"


def result_key_for_iou(value):
    return f"iou_{format_iou_metric_key(value)}"


def build_infer_info(opt, hypes):
    postprocess_params = hypes["postprocess"]
    score_threshold = postprocess_params.get("target_args", {}).get(
        "score_threshold",
        postprocess_params.get("anchor_args", {}).get("score_threshold"),
    )
    nms_thresh = postprocess_params.get("nms_thresh")
    return (
        f"{opt.fusion_method}{opt.note}"
        f"_score{_format_float_tag(score_threshold)}"
        f"_nms{_format_float_tag(nms_thresh)}"
    )


def init_object_result_stat():
    return {
        iou_thresh: {"tp": [], "fp": [], "gt": 0, "score": []}
        for iou_thresh in IOU_THRESHOLDS
    }


def calculate_ap_from_item_with_sort_option(result_item, global_sort_detections):
    if result_item['gt'] == 0:
        return None, [], []

    fp = np.asarray(result_item['fp'])
    tp = np.asarray(result_item['tp'])
    score = np.asarray(result_item['score'])
    if len(fp) == 0:
        return 0.0, [0.0, 1.0], [0.0, 0.0]

    if global_sort_detections:
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index]
        tp = tp[sorted_index]

    fp = np.cumsum(fp).tolist()
    tp = np.cumsum(tp).tolist()
    rec = [float(x) / result_item['gt'] for x in tp]
    prec = [float(t) / max(float(t + f), 1e-12) for t, f in zip(tp, fp)]

    rec_for_ap = [0.0] + rec + [1.0]
    prec_for_ap = [0.0] + prec + [0.0]
    for idx in range(len(prec_for_ap) - 2, -1, -1):
        prec_for_ap[idx] = max(prec_for_ap[idx], prec_for_ap[idx + 1])

    ap = 0.0
    for idx in range(1, len(rec_for_ap)):
        if rec_for_ap[idx] != rec_for_ap[idx - 1]:
            ap += (rec_for_ap[idx] - rec_for_ap[idx - 1]) * prec_for_ap[idx]
    return ap, rec_for_ap, prec_for_ap


def summarize_per_class_map(result_stat, label_names, global_sort_detections=False):
    summary = OrderedDict()
    if result_stat is None:
        return summary

    for label_name in label_names:
        class_summary = OrderedDict()
        for iou_thresh in IOU_THRESHOLDS:
            item = result_stat[iou_thresh].get(label_name)
            if item is None:
                ap = 0.0
            else:
                ap, _, _ = calculate_ap_from_item_with_sort_option(
                    item, global_sort_detections
                )
                ap = 0.0 if ap is None else float(ap)
            class_summary[map_key_for_iou(iou_thresh)] = ap
        summary[label_name] = class_summary
    return summary


def summarize_multiclass_map(result_stat, class_names, metric_name="Isaac multi-class",
                             global_sort_detections=False):
    summary = OrderedDict()
    per_class_map = summarize_per_class_map(
        result_stat, class_names, global_sort_detections
    )

    for iou_thresh in IOU_THRESHOLDS:
        threshold_key = map_key_for_iou(iou_thresh)
        class_values = []
        for class_name in class_names:
            item = result_stat[iou_thresh][class_name]
            if int(item["gt"]) <= 0:
                continue
            ap, _, _ = calculate_ap_from_item_with_sort_option(
                item, global_sort_detections
            )
            class_values.append(0.0 if ap is None else float(ap))
        summary[threshold_key] = float(np.mean(class_values)) if class_values else 0.0

    print(
        f"{metric_name} mAP: "
        f"mAP@0.3={summary['mAP@0.3']:.4f}, "
        f"mAP@0.5={summary['mAP@0.5']:.4f}, "
        f"mAP@0.7={summary['mAP@0.7']:.4f}"
    )
    return summary, per_class_map


def summarize_false_positive_metrics(result_stat, sample_count):
    summary = {}
    sample_count = max(int(sample_count), 1)

    for iou_thresh in IOU_THRESHOLDS:
        item = result_stat[iou_thresh]
        tp_count = int(np.sum(item["tp"]))
        fp_count = int(np.sum(item["fp"]))
        gt_count = int(item["gt"])
        pred_count = tp_count + fp_count
        precision = tp_count / max(pred_count, 1)
        recall = tp_count / max(gt_count, 1)
        false_positives_per_image = fp_count / sample_count

        summary[result_key_for_iou(iou_thresh)] = {
            "tp": tp_count,
            "fp": fp_count,
            "gt": gt_count,
            "pred": pred_count,
            "precision": precision,
            "recall": recall,
            "false_positives_per_image": false_positives_per_image,
        }

    fp50 = summary["iou_0.5"]["false_positives_per_image"]
    prec50 = summary["iou_0.5"]["precision"]
    print(
        "False-positive summary: "
        f"FPPI@0.5={fp50:.3f}, Precision@0.5={prec50:.3f}"
    )
    return summary


def _empty_paper_metrics(reason):
    return OrderedDict([("available", False), ("reason", reason)])


def summarize_multi_threshold_map(map_summary):
    if not isinstance(map_summary, dict) or map_summary.get("available") is False:
        return _empty_paper_metrics("Class-aware mAP summary is unavailable.")
    summary = OrderedDict()
    for iou_thresh in PAPER_MAP_THRESHOLDS:
        key = map_key_for_iou(iou_thresh)
        summary[key] = float(map_summary.get(key, 0.0))
    return summary


def summarize_size_group_map(per_class_map):
    if not isinstance(per_class_map, dict) or per_class_map.get("available") is False:
        return _empty_paper_metrics("Per-class mAP is unavailable.")

    summary = OrderedDict()
    for group_name, classes in PAPER_SIZE_GROUPS.items():
        present_classes = [name for name in classes if name in per_class_map]
        m03_values = [float(per_class_map[name].get("mAP@0.3", 0.0)) for name in present_classes]
        m05_values = [float(per_class_map[name].get("mAP@0.5", 0.0)) for name in present_classes]
        m03 = float(np.mean(m03_values)) if m03_values else 0.0
        m05 = float(np.mean(m05_values)) if m05_values else 0.0
        relative_drop = None if m03 <= 0.0 else float((m03 - m05) / m03)
        summary[group_name] = OrderedDict(
            [
                ("classes", list(classes)),
                ("present_classes", present_classes),
                ("mAP@0.3", m03),
                ("mAP@0.5", m05),
                ("relative_drop", relative_drop),
            ]
        )
    return summary


def init_tp_center_error_stat(class_names):
    return OrderedDict(
        [
            ("iou_threshold", 0.3),
            ("distances", []),
            ("per_class_count", {class_name: 0 for class_name in class_names}),
        ]
    )


def _boxes_to_numpy_or_empty(boxes):
    if boxes is None:
        return np.zeros((0, 8, 3), dtype=np.float32)
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return boxes


def _tensor_to_numpy_or_empty(values, dtype=None):
    if values is None:
        return np.zeros((0,), dtype=dtype or np.float32)
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if dtype is not None:
        values = values.astype(dtype)
    if values.size == 0:
        return np.zeros((0,), dtype=dtype or values.dtype)
    return values.reshape(-1)


def _bev_center_from_corners(corners):
    corners = np.asarray(corners, dtype=np.float32)
    return corners[:, :2].mean(axis=0)


def update_tp_center_error_stat(
    stat,
    pred_boxes,
    pred_score,
    pred_labels,
    gt_boxes,
    gt_labels,
    class_names,
    iou_thresh=0.3,
):
    if stat is None or pred_labels is None or gt_labels is None:
        return

    pred_boxes_np = _boxes_to_numpy_or_empty(pred_boxes)
    gt_boxes_np = _boxes_to_numpy_or_empty(gt_boxes)
    pred_scores_np = _tensor_to_numpy_or_empty(pred_score, dtype=np.float32)
    pred_labels_np = _tensor_to_numpy_or_empty(pred_labels, dtype=np.int64)
    gt_labels_np = _tensor_to_numpy_or_empty(gt_labels, dtype=np.int64)

    if pred_boxes_np.shape[0] == 0 or gt_boxes_np.shape[0] == 0:
        return

    pred_polygons_all = list(common_utils.convert_format(pred_boxes_np))
    gt_polygons_all = list(common_utils.convert_format(gt_boxes_np))

    for class_id, class_name in enumerate(class_names):
        det_indices = np.where(pred_labels_np == class_id)[0]
        gt_indices = np.where(gt_labels_np == class_id)[0]
        if det_indices.size == 0 or gt_indices.size == 0:
            continue
        det_indices = det_indices[np.argsort(-pred_scores_np[det_indices])]
        remaining_gt = list(gt_indices.tolist())
        for det_idx in det_indices:
            if not remaining_gt:
                break
            ious = common_utils.compute_iou(
                pred_polygons_all[int(det_idx)],
                [gt_polygons_all[int(gt_idx)] for gt_idx in remaining_gt],
            )
            if len(ious) == 0 or float(np.max(ious)) < iou_thresh:
                continue
            match_pos = int(np.argmax(ious))
            gt_idx = remaining_gt.pop(match_pos)
            pred_center = _bev_center_from_corners(pred_boxes_np[int(det_idx)])
            gt_center = _bev_center_from_corners(gt_boxes_np[int(gt_idx)])
            distance = float(np.linalg.norm(pred_center - gt_center))
            stat["distances"].append(distance)
            stat["per_class_count"][class_name] = stat["per_class_count"].get(class_name, 0) + 1


def summarize_tp_center_error_stat(stat):
    distances = np.asarray(stat.get("distances", []), dtype=np.float32)
    summary = OrderedDict([("iou_0.3", OrderedDict())])
    entry = summary["iou_0.3"]
    entry["unit"] = "meter"
    entry["matching"] = "class_wise_score_sorted_greedy"
    entry["count"] = int(distances.size)
    if distances.size == 0:
        entry["mean"] = None
        entry["median"] = None
    else:
        entry["mean"] = float(np.mean(distances))
        entry["median"] = float(np.median(distances))
    entry["per_class_count"] = dict(stat.get("per_class_count", {}))
    return summary


def init_classwise_ate_ase_stat(class_names, distance_threshold=INDOOR_TP_CENTER_DISTANCE):
    return OrderedDict(
        [
            ("distance_threshold", float(distance_threshold)),
            (
                "per_class",
                OrderedDict(
                    (
                        class_name,
                        {
                            "gt_count": 0,
                            "pred_count": 0,
                            "translation_errors": [],
                            "scale_errors": [],
                        },
                    )
                    for class_name in class_names
                ),
            ),
        ]
    )


def _box_size_from_corners(corners):
    """Return orientation-invariant (short side, long side, height) dimensions."""
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (8, 3):
        return None
    bottom = corners[np.argsort(corners[:, 2])[:4], :2]
    pairwise = []
    for first_idx in range(4):
        for second_idx in range(first_idx + 1, 4):
            pairwise.append(float(np.linalg.norm(bottom[first_idx] - bottom[second_idx])))
    pairwise.sort()
    if len(pairwise) != 6:
        return None
    short_side = float(np.mean(pairwise[:2]))
    long_side = float(np.mean(pairwise[2:4]))
    height = float(np.max(corners[:, 2]) - np.min(corners[:, 2]))
    dimensions = np.asarray([short_side, long_side, height], dtype=np.float32)
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        return None
    return dimensions


def _scale_error_from_corners(pred_corners, gt_corners):
    pred_size = _box_size_from_corners(pred_corners)
    gt_size = _box_size_from_corners(gt_corners)
    if pred_size is None or gt_size is None:
        return None
    intersection = float(np.prod(np.minimum(pred_size, gt_size)))
    pred_volume = float(np.prod(pred_size))
    gt_volume = float(np.prod(gt_size))
    union = pred_volume + gt_volume - intersection
    if union <= 0.0:
        return None
    return float(1.0 - intersection / union)


def update_classwise_ate_ase_stat(
    stat,
    pred_boxes,
    pred_score,
    pred_labels,
    gt_boxes,
    gt_labels,
    class_names,
):
    """Accumulate class-wise XY ATE and aligned-size 3D ASE at 0.5 m."""
    if stat is None or pred_labels is None or gt_labels is None:
        return

    pred_boxes_np = _boxes_to_numpy_or_empty(pred_boxes)
    gt_boxes_np = _boxes_to_numpy_or_empty(gt_boxes)
    pred_scores_np = _tensor_to_numpy_or_empty(pred_score, dtype=np.float32)
    pred_labels_np = _tensor_to_numpy_or_empty(pred_labels, dtype=np.int64)
    gt_labels_np = _tensor_to_numpy_or_empty(gt_labels, dtype=np.int64)
    distance_threshold = float(stat["distance_threshold"])

    for class_id, class_name in enumerate(class_names):
        class_stat = stat["per_class"][class_name]
        pred_indices = np.where(pred_labels_np == class_id)[0]
        gt_indices = np.where(gt_labels_np == class_id)[0]
        class_stat["pred_count"] += int(pred_indices.size)
        class_stat["gt_count"] += int(gt_indices.size)
        if pred_indices.size == 0 or gt_indices.size == 0:
            continue

        pred_indices = pred_indices[np.argsort(-pred_scores_np[pred_indices])]
        remaining_gt = list(gt_indices.tolist())
        for pred_idx in pred_indices:
            if not remaining_gt:
                break
            pred_center = _bev_center_from_corners(pred_boxes_np[int(pred_idx)])
            gt_centers = np.asarray(
                [_bev_center_from_corners(gt_boxes_np[int(gt_idx)]) for gt_idx in remaining_gt],
                dtype=np.float32,
            )
            distances = np.linalg.norm(gt_centers - pred_center[None, :], axis=1)
            match_pos = int(np.argmin(distances))
            translation_error = float(distances[match_pos])
            if translation_error > distance_threshold:
                continue
            gt_idx = remaining_gt.pop(match_pos)
            scale_error = _scale_error_from_corners(
                pred_boxes_np[int(pred_idx)], gt_boxes_np[int(gt_idx)]
            )
            class_stat["translation_errors"].append(translation_error)
            if scale_error is not None:
                class_stat["scale_errors"].append(scale_error)


def summarize_classwise_ate_ase_stat(stat):
    threshold = float(stat["distance_threshold"])
    summary = OrderedDict(
        [
            ("distance_threshold_m", threshold),
            ("matching", "class_wise_score_sorted_greedy_xy_center_distance"),
            ("ate_unit", "meter"),
            ("ase_definition", "1 - aligned orientation-invariant 3D size IoU"),
            ("per_class", OrderedDict()),
        ]
    )
    class_ate_means = []
    class_ase_means = []
    matched_classes = []
    excluded_no_match = []

    for class_name, class_stat in stat["per_class"].items():
        translation_errors = np.asarray(class_stat["translation_errors"], dtype=np.float32)
        scale_errors = np.asarray(class_stat["scale_errors"], dtype=np.float32)
        gt_count = int(class_stat["gt_count"])
        matched_count = int(translation_errors.size)
        entry = OrderedDict(
            [
                ("gt_count", gt_count),
                ("pred_count", int(class_stat["pred_count"])),
                ("matched_tp", matched_count),
                ("recall", float(matched_count / gt_count) if gt_count > 0 else None),
                ("ATE_mean", float(np.mean(translation_errors)) if matched_count else None),
                ("ATE_median", float(np.median(translation_errors)) if matched_count else None),
                ("ASE_mean", float(np.mean(scale_errors)) if scale_errors.size else None),
                ("ASE_median", float(np.median(scale_errors)) if scale_errors.size else None),
            ]
        )
        summary["per_class"][class_name] = entry
        if gt_count > 0 and matched_count > 0:
            matched_classes.append(class_name)
            class_ate_means.append(entry["ATE_mean"])
            if entry["ASE_mean"] is not None:
                class_ase_means.append(entry["ASE_mean"])
        elif gt_count > 0:
            excluded_no_match.append(class_name)

    summary["macro_mATE"] = (
        float(np.mean(class_ate_means)) if class_ate_means else None
    )
    summary["macro_mASE"] = (
        float(np.mean(class_ase_means)) if class_ase_means else None
    )
    summary["macro_averaged_classes"] = matched_classes
    summary["excluded_classes_without_matches"] = excluded_no_match
    summary["macro_policy"] = (
        "Unweighted mean of per-class errors over classes with GT and at least one matched TP; "
        "use the accompanying per-class recall/mAP to assess missed detections."
    )
    print(
        "Indoor class-wise localization: "
        f"mATE@{threshold:g}m={summary['macro_mATE']}, "
        f"mASE@{threshold:g}m={summary['macro_mASE']}"
    )
    return summary


def init_visibility_subset_stat(class_names):
    result_stat = OrderedDict()
    localization_stat = OrderedDict()
    distribution = OrderedDict()
    for subset_name in VISIBILITY_SUBSETS:
        result_stat[subset_name] = {
            iou_thresh: {
                class_name: {
                    "tp": [],
                    "fp": [],
                    "gt": 0,
                    "score": [],
                    "ignored_predictions": 0,
                }
                for class_name in class_names
            }
            for iou_thresh in VISIBILITY_IOU_THRESHOLDS
        }
        localization_stat[subset_name] = OrderedDict(
            (
                class_name,
                {
                    "gt_count": 0,
                    "translation_errors": [],
                    "scale_errors": [],
                },
            )
            for class_name in class_names
        )
        distribution[subset_name] = {
            "gt_count": 0,
            "frames_with_gt": 0,
            "per_class_gt_count": {
                class_name: 0 for class_name in class_names
            },
        }
    return OrderedDict(
        [
            ("result_stat", result_stat),
            ("localization_stat", localization_stat),
            ("distribution", distribution),
        ]
    )


def _normalize_visibility_object_id(object_id):
    return str(object_id)


def collect_isaac_visibility_membership(opencood_dataset, expanded_idx):
    """Read per-agent visible GT IDs without reloading image/LiDAR tensors."""
    sample_records = _get_opv2v_sample_records(opencood_dataset, expanded_idx)
    if not sample_records:
        raise RuntimeError("Visibility subsets require at least one Isaac agent record.")

    vis_info = _vis_sample_info(opencood_dataset, expanded_idx)
    ordered_cav_ids = _sorted_cav_ids(sample_records)
    ego_cav_id = vis_info.get("ego_cav_id")
    if ego_cav_id is None:
        ego_cav_id = ordered_cav_ids[0]
    if ego_cav_id not in sample_records:
        normalized_to_raw = {
            str(cav_id): cav_id for cav_id in ordered_cav_ids
        }
        ego_cav_id = normalized_to_raw.get(str(ego_cav_id), ego_cav_id)
    if ego_cav_id not in sample_records:
        raise KeyError(f"Visibility subset ego CAV {ego_cav_id} is missing.")

    per_agent_ids = OrderedDict()
    for cav_id in ordered_cav_ids:
        yaml_path = sample_records[cav_id].get("yaml")
        if not yaml_path or not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Visibility subset label YAML is missing for CAV {cav_id}: {yaml_path}"
            )
        params = yaml_utils.load_yaml(yaml_path)
        vehicles = params.get("vehicles", {}) or {}
        per_agent_ids[cav_id] = {
            _normalize_visibility_object_id(object_id)
            for object_id in vehicles.keys()
        }

    ego_ids = per_agent_ids[ego_cav_id]
    partner_cav_ids = [cav_id for cav_id in ordered_cav_ids if cav_id != ego_cav_id]
    partner_ids = set()
    for cav_id in partner_cav_ids:
        partner_ids.update(per_agent_ids[cav_id])

    membership = {}
    for object_id in ego_ids | partner_ids:
        if object_id in ego_ids and object_id in partner_ids:
            membership[object_id] = "shared"
        elif object_id in ego_ids:
            membership[object_id] = "ego_only"
        else:
            membership[object_id] = "partner_only"

    return membership, OrderedDict(
        [
            ("sample_idx", int(vis_info["sample_idx"])),
            ("expanded_idx", int(vis_info["expanded_idx"])),
            ("ego_cav_id", str(ego_cav_id)),
            ("partner_cav_ids", [str(cav_id) for cav_id in partner_cav_ids]),
        ]
    )


def generate_visibility_labels_for_evaluated_gt(
    data_dict,
    postprocess_params,
    membership,
):
    """Mirror Isaac GT generation and retain object IDs/subset membership."""
    gt_box3d_list = []
    gt_label_list = []
    object_id_list = []

    for cav_content in data_dict.values():
        transformation_matrix = cav_content["transformation_matrix_clean"]
        object_bbx_center = cav_content["object_bbx_center"]
        object_bbx_mask = cav_content["object_bbx_mask"].bool()
        object_ids = list(cav_content["object_ids"])
        class_ids = cav_content["label_dict"].get("object_class_ids")
        if class_ids is None:
            class_ids = torch.zeros_like(object_bbx_mask, dtype=torch.long)

        valid_boxes = object_bbx_center[object_bbx_mask]
        valid_classes = class_ids.to(object_bbx_mask.device).long()[object_bbx_mask]
        if valid_boxes.numel() == 0:
            continue
        if len(object_ids) != int(valid_boxes.shape[0]):
            raise RuntimeError(
                "Visibility subset object ID/GT count mismatch: "
                f"{len(object_ids)} IDs vs {int(valid_boxes.shape[0])} boxes."
            )

        corners = box_utils.boxes_to_corners_3d(
            valid_boxes, postprocess_params["order"]
        )
        projected_corners = box_utils.project_box3d(
            corners.float(), transformation_matrix
        )
        gt_box3d_list.append(projected_corners)
        gt_label_list.append(valid_classes)
        object_id_list.extend(
            _normalize_visibility_object_id(object_id)
            for object_id in object_ids
        )

    if not gt_box3d_list:
        device = next(iter(data_dict.values()))["object_bbx_center"].device
        return (
            torch.empty((0, 8, 3), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [],
            [],
        )

    gt_boxes = torch.vstack(gt_box3d_list)
    gt_labels = torch.cat(gt_label_list).to(gt_boxes.device)
    seen = set()
    selected_indices = []
    selected_object_ids = []
    for idx, object_id in enumerate(object_id_list):
        if object_id in seen:
            continue
        seen.add(object_id)
        selected_indices.append(idx)
        selected_object_ids.append(object_id)

    selected_indices_tensor = torch.as_tensor(
        selected_indices, dtype=torch.long, device=gt_boxes.device
    )
    gt_boxes = gt_boxes[selected_indices_tensor]
    gt_labels = gt_labels[selected_indices_tensor]
    range_mask = box_utils.get_mask_for_boxes_within_range_torch(
        gt_boxes, postprocess_params["gt_range"]
    )
    gt_boxes = gt_boxes[range_mask]
    gt_labels = gt_labels[range_mask]
    range_mask_np = range_mask.detach().cpu().numpy().astype(bool)
    selected_object_ids = [
        object_id
        for object_id, keep in zip(selected_object_ids, range_mask_np)
        if keep
    ]
    subset_labels = [membership.get(object_id) for object_id in selected_object_ids]
    return gt_boxes, gt_labels, selected_object_ids, subset_labels


def _verify_visibility_gt_alignment(reference_boxes, reference_labels, boxes, labels):
    reference_count = 0 if reference_boxes is None else int(reference_boxes.shape[0])
    if reference_count != int(boxes.shape[0]):
        raise RuntimeError(
            "Visibility subset GT count does not match the existing evaluator: "
            f"{int(boxes.shape[0])} vs {reference_count}."
        )
    if reference_count == 0:
        return
    if not torch.allclose(reference_boxes.float(), boxes.float(), atol=1e-4, rtol=1e-4):
        max_error = float(torch.max(torch.abs(reference_boxes.float() - boxes.float())))
        raise RuntimeError(
            "Visibility subset GT boxes are not aligned with the existing evaluator; "
            f"max corner error={max_error:.6g}."
        )
    if reference_labels is not None and not torch.equal(
        reference_labels.long(), labels.long()
    ):
        raise RuntimeError("Visibility subset GT class labels are not aligned.")


def _update_visibility_ap_item(
    result_item,
    det_boxes,
    det_scores,
    gt_boxes,
    gt_is_target,
    iou_thresh,
):
    gt_is_target = np.asarray(gt_is_target, dtype=bool)
    result_item["gt"] += int(np.sum(gt_is_target))
    if det_boxes is None or int(det_boxes.shape[0]) == 0:
        return

    det_boxes_np = _boxes_to_numpy_or_empty(det_boxes)
    det_scores_np = _tensor_to_numpy_or_empty(det_scores, dtype=np.float32)
    gt_boxes_np = _boxes_to_numpy_or_empty(gt_boxes)
    det_polygons = list(common_utils.convert_format(det_boxes_np))
    remaining_gt_polygons = list(common_utils.convert_format(gt_boxes_np))
    remaining_gt_targets = gt_is_target.tolist()

    for det_idx in np.argsort(-det_scores_np):
        score = float(det_scores_np[int(det_idx)])
        if not remaining_gt_polygons:
            result_item["tp"].append(0)
            result_item["fp"].append(1)
            result_item["score"].append(score)
            continue
        ious = common_utils.compute_iou(
            det_polygons[int(det_idx)], remaining_gt_polygons
        )
        if len(ious) == 0 or float(np.max(ious)) < iou_thresh:
            result_item["tp"].append(0)
            result_item["fp"].append(1)
            result_item["score"].append(score)
            continue
        match_pos = int(np.argmax(ious))
        is_target = bool(remaining_gt_targets.pop(match_pos))
        remaining_gt_polygons.pop(match_pos)
        if is_target:
            result_item["tp"].append(1)
            result_item["fp"].append(0)
            result_item["score"].append(score)
        else:
            result_item["ignored_predictions"] += 1


def update_visibility_subset_stat(
    stat,
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    subset_labels,
    class_names,
):
    if stat is None or pred_labels is None or gt_labels is None:
        return
    subset_labels = np.asarray(subset_labels, dtype=object)
    gt_labels_np = _tensor_to_numpy_or_empty(gt_labels, dtype=np.int64)
    if subset_labels.shape[0] != gt_labels_np.shape[0]:
        raise RuntimeError("Visibility subset labels are not aligned with GT labels.")

    unknown_mask = np.asarray([label is None for label in subset_labels], dtype=bool)
    if np.any(unknown_mask):
        raise RuntimeError(
            "Evaluated GT contains object IDs absent from all per-agent visibility labels."
        )

    for subset_name in VISIBILITY_SUBSETS:
        subset_mask = subset_labels == subset_name
        count = int(np.sum(subset_mask))
        distribution = stat["distribution"][subset_name]
        distribution["gt_count"] += count
        if count > 0:
            distribution["frames_with_gt"] += 1
        for class_id, class_name in enumerate(class_names):
            class_count = int(np.sum(subset_mask & (gt_labels_np == class_id)))
            distribution["per_class_gt_count"][class_name] += class_count
            stat["localization_stat"][subset_name][class_name][
                "gt_count"
            ] += class_count

    pred_labels_np = _tensor_to_numpy_or_empty(pred_labels, dtype=np.int64)
    for class_id, class_name in enumerate(class_names):
        gt_class_mask_np = gt_labels_np == class_id
        gt_class_mask = torch.as_tensor(
            gt_class_mask_np, dtype=torch.bool, device=gt_boxes.device
        )
        class_gt_boxes = gt_boxes[gt_class_mask]
        class_subset_labels = subset_labels[gt_class_mask_np]
        class_pred_boxes = None
        class_pred_scores = None
        class_pred_indices = np.where(pred_labels_np == class_id)[0]
        if pred_boxes is not None and class_pred_indices.size > 0:
            class_pred_indices_tensor = torch.as_tensor(
                class_pred_indices, dtype=torch.long, device=pred_boxes.device
            )
            class_pred_boxes = pred_boxes[class_pred_indices_tensor]
            class_pred_scores = pred_scores[class_pred_indices_tensor]

        for subset_name in VISIBILITY_SUBSETS:
            target_mask = class_subset_labels == subset_name
            for iou_thresh in VISIBILITY_IOU_THRESHOLDS:
                _update_visibility_ap_item(
                    stat["result_stat"][subset_name][iou_thresh][class_name],
                    class_pred_boxes,
                    class_pred_scores,
                    class_gt_boxes,
                    target_mask,
                    iou_thresh,
                )

            localization_item = stat["localization_stat"][subset_name][class_name]
            if class_pred_boxes is None or int(class_gt_boxes.shape[0]) == 0:
                continue
            pred_boxes_np = _boxes_to_numpy_or_empty(class_pred_boxes)
            pred_scores_np = _tensor_to_numpy_or_empty(
                class_pred_scores, dtype=np.float32
            )
            gt_boxes_np = _boxes_to_numpy_or_empty(class_gt_boxes)
            remaining_gt = list(range(gt_boxes_np.shape[0]))
            for pred_idx in np.argsort(-pred_scores_np):
                if not remaining_gt:
                    break
                pred_center = _bev_center_from_corners(pred_boxes_np[int(pred_idx)])
                gt_centers = np.asarray(
                    [_bev_center_from_corners(gt_boxes_np[idx]) for idx in remaining_gt],
                    dtype=np.float32,
                )
                distances = np.linalg.norm(gt_centers - pred_center[None, :], axis=1)
                match_pos = int(np.argmin(distances))
                translation_error = float(distances[match_pos])
                if translation_error > INDOOR_TP_CENTER_DISTANCE:
                    continue
                gt_idx = remaining_gt.pop(match_pos)
                if class_subset_labels[gt_idx] != subset_name:
                    continue
                scale_error = _scale_error_from_corners(
                    pred_boxes_np[int(pred_idx)], gt_boxes_np[int(gt_idx)]
                )
                localization_item["translation_errors"].append(translation_error)
                if scale_error is not None:
                    localization_item["scale_errors"].append(scale_error)


def summarize_visibility_subset_stat(
    stat, class_names, global_sort_detections=True
):
    total_union_gt = sum(
        int(stat["distribution"][subset_name]["gt_count"])
        for subset_name in VISIBILITY_SUBSETS
    )
    summary = OrderedDict()
    summary["protocol"] = OrderedDict(
        [
            ("unit", "object"),
            ("coordinate_frame", "ego"),
            ("partner_scope", "union_of_non_ego_agents"),
            (
                "subset_definition",
                OrderedDict(
                    [
                        ("ego_only", "ego_gt_ids - partner_gt_ids"),
                        ("partner_only", "partner_gt_ids - ego_gt_ids"),
                        ("shared", "ego_gt_ids intersect partner_gt_ids"),
                    ]
                ),
            ),
            ("id_matching", "isaac_global_object_id"),
            (
                "iou_ap",
                OrderedDict(
                    [
                        ("thresholds", list(VISIBILITY_IOU_THRESHOLDS)),
                        ("box_type", "rotated_bev"),
                        ("global_confidence_sort", bool(global_sort_detections)),
                        ("non_target_gt_policy", "ignore"),
                        ("background_prediction_policy", "false_positive"),
                    ]
                ),
            ),
            (
                "localization",
                OrderedDict(
                    [
                        ("matching", "class_wise_score_sorted_greedy_xy_center_distance"),
                        ("distance_threshold_m", INDOOR_TP_CENTER_DISTANCE),
                        ("mATE_unit", "meter"),
                        ("mASE_definition", "1 - aligned orientation-invariant 3D size IoU"),
                    ]
                ),
            ),
        ]
    )
    summary["distribution"] = OrderedDict([("total_union_gt", total_union_gt)])
    summary["subsets"] = OrderedDict()

    for subset_name in VISIBILITY_SUBSETS:
        distribution = stat["distribution"][subset_name]
        gt_count = int(distribution["gt_count"])
        distribution_entry = OrderedDict(
            [
                ("gt_count", gt_count),
                ("gt_ratio", float(gt_count / total_union_gt) if total_union_gt else 0.0),
                ("frames_with_gt", int(distribution["frames_with_gt"])),
                ("per_class_gt_count", dict(distribution["per_class_gt_count"])),
            ]
        )
        summary["distribution"][subset_name] = distribution_entry
        subset_summary = OrderedDict(
            [
                ("gt_count", gt_count),
                ("gt_ratio", distribution_entry["gt_ratio"]),
                ("detection", OrderedDict()),
                ("localization_0.5m", OrderedDict()),
                ("per_class", OrderedDict()),
            ]
        )

        for class_name in class_names:
            subset_summary["per_class"][class_name] = OrderedDict(
                [("gt_count", int(distribution["per_class_gt_count"][class_name]))]
            )

        for iou_thresh in VISIBILITY_IOU_THRESHOLDS:
            threshold_key = format_iou_metric_key(iou_thresh)
            macro_ap_values = []
            total_tp = 0
            total_fp = 0
            total_gt = 0
            total_ignored = 0
            for class_name in class_names:
                item = stat["result_stat"][subset_name][iou_thresh][class_name]
                ap, _, _ = calculate_ap_from_item_with_sort_option(
                    item, global_sort_detections
                )
                class_gt_count = int(item["gt"])
                class_tp = int(np.sum(item["tp"]))
                class_fp = int(np.sum(item["fp"]))
                if ap is not None:
                    macro_ap_values.append(float(ap))
                per_class_entry = subset_summary["per_class"][class_name]
                per_class_entry[f"mAP@{threshold_key}"] = (
                    None if ap is None else float(ap)
                )
                per_class_entry[f"recall@{threshold_key}"] = (
                    float(class_tp / class_gt_count) if class_gt_count > 0 else None
                )
                total_tp += class_tp
                total_fp += class_fp
                total_gt += class_gt_count
                total_ignored += int(item["ignored_predictions"])

            map_value = float(np.mean(macro_ap_values)) if macro_ap_values else None
            subset_summary["detection"][f"mAP@{threshold_key}"] = map_value
            subset_summary["detection"][f"iou_{threshold_key}"] = OrderedDict(
                [
                    ("tp", total_tp),
                    ("fp", total_fp),
                    ("fn", max(total_gt - total_tp, 0)),
                    ("ignored_predictions", total_ignored),
                    ("recall", float(total_tp / total_gt) if total_gt else None),
                    (
                        "precision",
                        float(total_tp / (total_tp + total_fp))
                        if total_gt > 0 and total_tp + total_fp > 0 else None,
                    ),
                ]
            )

        class_ate_means = []
        class_ase_means = []
        total_localization_matches = 0
        for class_name in class_names:
            localization_item = stat["localization_stat"][subset_name][class_name]
            translation_errors = np.asarray(
                localization_item["translation_errors"], dtype=np.float32
            )
            scale_errors = np.asarray(
                localization_item["scale_errors"], dtype=np.float32
            )
            matched_tp = int(translation_errors.size)
            total_localization_matches += matched_tp
            ate_mean = float(np.mean(translation_errors)) if matched_tp else None
            ase_mean = float(np.mean(scale_errors)) if scale_errors.size else None
            per_class_entry = subset_summary["per_class"][class_name]
            per_class_entry["matched_tp_0.5m"] = matched_tp
            per_class_entry["ATE_mean"] = ate_mean
            per_class_entry["ATE_median"] = (
                float(np.median(translation_errors)) if matched_tp else None
            )
            per_class_entry["ASE_mean"] = ase_mean
            per_class_entry["ASE_median"] = (
                float(np.median(scale_errors)) if scale_errors.size else None
            )
            if ate_mean is not None:
                class_ate_means.append(ate_mean)
            if ase_mean is not None:
                class_ase_means.append(ase_mean)

        subset_summary["localization_0.5m"] = OrderedDict(
            [
                ("matched_tp", total_localization_matches),
                (
                    "recall",
                    float(total_localization_matches / gt_count) if gt_count else None,
                ),
                (
                    "macro_mATE",
                    float(np.mean(class_ate_means)) if class_ate_means else None,
                ),
                (
                    "macro_mASE",
                    float(np.mean(class_ase_means)) if class_ase_means else None,
                ),
            ]
        )
        summary["subsets"][subset_name] = subset_summary

    return summary


def classify_isaac_case_type(opencood_dataset, sample_idx):
    meta = _isaac_case_metadata(opencood_dataset, sample_idx)
    text = " ".join(
        str(meta.get(key, ""))
        for key in ("scenario_name", "scenario_tag")
    ).lower()
    if "distance" in text:
        return "distance", meta
    if any(token in text for token in ("occlusion", "occluded", "shadow")):
        return "occlusion", meta
    return None, meta


def init_case_type_result_stat(class_names):
    return OrderedDict(
        (
            case_type,
            OrderedDict(
                [
                    ("num_samples", 0),
                    ("result_stat", eval_utils_isaac.init_multiclass_result_stat(class_names, IOU_THRESHOLDS)),
                ]
            ),
        )
        for case_type in PAPER_CASE_TYPES
    )


def summarize_case_type_map(case_type_result_stat, class_names, global_sort_detections=False):
    summary = OrderedDict()
    available_any = False
    for case_type, case_data in case_type_result_stat.items():
        num_samples = int(case_data.get("num_samples", 0))
        entry = OrderedDict([("num_samples", num_samples)])
        if num_samples <= 0:
            entry["available"] = False
            entry["reason"] = f"No samples classified as {case_type}."
            summary[case_type] = entry
            continue
        map_summary, _ = summarize_multiclass_map(
            case_data["result_stat"],
            class_names,
            metric_name=f"Isaac {case_type} case",
            global_sort_detections=global_sort_detections,
        )
        entry["mAP@0.3"] = float(map_summary.get("mAP@0.3", 0.0))
        entry["mAP@0.5"] = float(map_summary.get("mAP@0.5", 0.0))
        summary[case_type] = entry
        available_any = True

    if not available_any:
        summary["available"] = False
        summary["reason"] = "No scenario names matched distance/occlusion case-type keywords."
    return summary


def build_paper_metrics(
    map_summary,
    per_class_map,
    tp_center_error_stat,
    classwise_ate_ase_stat,
    case_type_result_stat,
    class_names,
    opt,
    hypes,
    resume_epoch,
    global_sort_detections=False,
):
    postprocess_params = hypes["postprocess"]
    score_threshold = postprocess_params.get("target_args", {}).get(
        "score_threshold",
        postprocess_params.get("anchor_args", {}).get("score_threshold"),
    )
    paper_metrics = OrderedDict()
    paper_metrics["multi_threshold_map"] = summarize_multi_threshold_map(map_summary)
    paper_metrics["size_group_map"] = summarize_size_group_map(per_class_map)
    paper_metrics["tp_center_error"] = summarize_tp_center_error_stat(
        tp_center_error_stat
    )
    paper_metrics["classwise_ate_ase"] = summarize_classwise_ate_ase_stat(
        classwise_ate_ase_stat
    )
    paper_metrics["case_type_map"] = summarize_case_type_map(
        case_type_result_stat,
        class_names,
        global_sort_detections=global_sort_detections,
    )
    paper_metrics["table_row"] = OrderedDict(
        [
            ("method_name", os.path.basename(opt.model_dir.rstrip(os.sep))),
            ("model_dir", opt.model_dir),
            ("fusion_method", opt.fusion_method),
            ("checkpoint_mode", opt.checkpoint_mode),
            ("epoch", int(resume_epoch)),
            ("score_threshold", None if score_threshold is None else float(score_threshold)),
            ("nms_thresh", float(postprocess_params.get("nms_thresh"))),
            ("global_sort_detections", bool(global_sort_detections)),
        ]
    )
    return paper_metrics


def to_plain_yaml_object(obj):
    if isinstance(obj, OrderedDict):
        return {k: to_plain_yaml_object(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: to_plain_yaml_object(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_yaml_object(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_plain_yaml_object(v) for v in obj]
    return obj


def save_isaac_eval_summary(
    result_stat,
    multiclass_result_stat,
    class_names,
    save_path,
    infer_info,
    sample_count,
    *,
    global_sort_detections,
    tp_center_error_stat,
    classwise_ate_ase_stat,
    case_type_result_stat,
    opt,
    hypes,
    resume_epoch,
    communication_summary,
    visibility_subset_stat,
):
    summary = OrderedDict(
        [
            (
                "definition",
                (
                    "Predicted and GT 3D boxes are evaluated in the ego BEV/3D "
                    "coordinate frame using rotated-box IoU thresholds."
                ),
            ),
            ("infer_info", infer_info),
            ("num_samples", int(sample_count)),
            ("global_sort_detections", bool(global_sort_detections)),
        ]
    )

    map_summary, per_class_map = summarize_multiclass_map(
        multiclass_result_stat,
        class_names,
        global_sort_detections=global_sort_detections,
    )
    summary["mAP"] = map_summary
    summary["per_class_mAP"] = per_class_map
    summary["false_positive"] = summarize_false_positive_metrics(
        result_stat, sample_count
    )
    selected_dataset_dir = (
        hypes.get("test_dir")
        if opt.eval_split == "test"
        else hypes.get("validate_dir")
    )
    summary["evaluation_dataset"] = OrderedDict(
        [
            ("split", opt.eval_split),
            ("directory", selected_dataset_dir),
            ("overridden", bool(opt.eval_dataset_dir)),
            (
                "tag",
                _evaluation_dataset_tag(selected_dataset_dir)
                if selected_dataset_dir
                else None,
            ),
        ]
    )
    modality_assignment_info = getattr(
        opt, "eval_modality_assignment_info", None
    )
    if modality_assignment_info is not None:
        summary["evaluation_dataset"]["modality_assignment"] = (
            modality_assignment_info
        )
    if opt.random_pair:
        summary["random_pair"] = OrderedDict(
            [
                ("enabled", True),
                ("seed", int(opt.random_pair_seed)),
                ("pairing_digest", getattr(opt, "random_pair_digest", None)),
                ("ego_policy", "original_test_split_ego"),
                ("partner_policy", "one_non_ego_agent_from_another_test_sample"),
                ("self_pairing", False),
            ]
        )
    visibility_summary = summarize_visibility_subset_stat(
        visibility_subset_stat,
        class_names,
        global_sort_detections=global_sort_detections,
    )
    summary["visibility_subset_metrics"] = visibility_summary
    if communication_summary is not None:
        summary["communication"] = communication_summary
    summary["paper_metrics"] = build_paper_metrics(
        map_summary,
        per_class_map,
        tp_center_error_stat,
        classwise_ate_ase_stat,
        case_type_result_stat,
        class_names,
        opt,
        hypes,
        resume_epoch,
        global_sort_detections=global_sort_detections,
    )

    output_path = os.path.join(save_path, f"eval_all_isaac_{infer_info}.yaml")
    summary = to_plain_yaml_object(summary)
    yaml_utils.save_yaml(summary, output_path)
    print(f"Saved unified Isaac evaluation summary to {output_path}")
    return summary


def test_parser():
    parser = argparse.ArgumentParser(
        description="InCoP IsaacSim inference and evaluation"
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='checkpoint directory containing config.yaml and net_epoch*.pth')
    parser.add_argument('--checkpoint_mode', type=str,
                        default='bestval',
                        choices=['bestval', 'latest'],
                        help='checkpoint selection: bestval uses net_epoch_bestval_at*.pth; latest uses the largest net_epoch*.pth')
    parser.add_argument('--fusion_method', type=str,
                        default='no',
                        choices=['no', 'late', 'intermediate'],
                        help='IsaacSim fusion mode: no, late, or intermediate')
    parser.add_argument('--eval_split', type=str, default='test',
                        choices=['test', 'val'],
                        help='dataset split for inference; default uses test_dir, val uses validate_dir')
    parser.add_argument('--eval_dataset_dir', type=str, default='',
                        help='optional dataset directory override for this evaluation only; the checkpoint config.yaml is not modified')
    parser.add_argument('--save_vis_interval', type=int, default=200,
                        help='interval of saving BEV visualization; <=0 disables visualization')
    parser.add_argument('--save_demo_output', action='store_true',
                        help='save per-frame Isaac visualization plus ego RGB 3D box projections')
    parser.add_argument('--save_full_vis_output', action='store_true',
                        help='save Isaac BEV/RGB/LiDAR visualizations for every frame')
    parser.add_argument('--stream_video_output', type=str, default='',
                        help='stream in-memory 3-panel RGB/BEV frames directly to this MP4 via ffmpeg')
    parser.add_argument('--video_only', action='store_true',
                        help='skip AP/statistics and all per-frame image files; requires --stream_video_output')
    parser.add_argument('--video_frame_stride', type=int, default=1,
                        help='evaluate every Nth dataset frame in streaming video mode (default: 1)')
    parser.add_argument('--video_fps', type=float, default=10.0,
                        help='streaming video output frame rate (default: 10)')
    parser.add_argument('--video_compare_fusion', action='store_true',
                        help='one-command 2x2 ego-only versus intermediate-fusion video; '
                             'automatically selects intermediate fusion, video-only mode, '
                             'and <model_dir>/video_compare_fusion.mp4 unless an output path is provided')
    parser.add_argument('--split_output_by_case', action='store_true',
                        help='split visualization output folders by Isaac case/scenario')
    parser.add_argument('--range', type=str, default="",
                        help="optional Isaac front range as x_min,y_min,x_max,y_max; defaults to the saved yaml range")
    parser.add_argument('--score_threshold', type=float, default=0.25,
                        help="postprocess target score_threshold; defaults to 0.25 for IsaacSim evaluation")
    parser.add_argument('--nms_thresh', type=float, default=0.15,
                        help="postprocess.nms_thresh; defaults to 0.15 for IsaacSim evaluation")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="optional limit for quick Isaac visualization/debug runs")
    parser.add_argument('--sample_indices', type=str, default="",
                        help="optional comma-separated dataset indices or ranges, e.g. 1600 or 1600,1800:1810")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker count; use 0 if multiprocessing data loading stalls')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='DataLoader prefetch factor when num_workers > 0')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='keep DataLoader workers alive across the full inference pass')
    parser.add_argument('--profile_timing', action='store_true',
                        help='print coarse per-stage timing at the end of inference')
    parser.add_argument('--all_ego', action='store_true',
                        help='legacy/debug mode: evaluate every CAV once as ego; default keeps robot 0 as ego')
    parser.add_argument('--random_pair', action='store_true',
                        help='keep each original ego but replace its partner with a non-ego agent from another sample in the same evaluation split')
    parser.add_argument('--random_pair_seed', type=int, default=303,
                        help='fixed seed used to build the deterministic random-pair assignment (default: 303)')
    parser.add_argument('--ego_only', action='store_true',
                        help='evaluate an intermediate-fusion checkpoint with only the ego agent; partner agents are removed before batch collation')
    parser.add_argument('--global_sort_detections', type=parse_bool_arg,
                        default=True, metavar='{true,false}',
                        help='whether to globally sort detections by confidence before AP calculation (default: true)')
    parser.add_argument('--pose_noise_pos_std', type=float, default=0.0,
                        help='Gaussian pose translation noise std in meters, same convention as OPV2V/CoAlign')
    parser.add_argument('--pose_noise_rot_std', type=float, default=0.0,
                        help='Gaussian pose yaw noise std in degrees, same convention as OPV2V/CoAlign')
    parser.add_argument('--pose_noise_pos_mean', type=float, default=0.0,
                        help='Gaussian pose translation noise mean in meters')
    parser.add_argument('--pose_noise_rot_mean', type=float, default=0.0,
                        help='Gaussian pose yaw noise mean in degrees')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def load_selected_saved_model(saved_path, model, checkpoint_mode):
    if checkpoint_mode == 'bestval':
        return train_utils.load_saved_model(saved_path, model)

    candidates = []
    for filename in os.listdir(saved_path):
        if not (filename.startswith('net_epoch') and filename.endswith('.pth')):
            continue
        if filename.startswith('net_epoch_bestval_at'):
            continue
        epoch_text = filename[len('net_epoch'):-len('.pth')]
        if not epoch_text.isdigit():
            continue
        candidates.append((int(epoch_text), os.path.join(saved_path, filename)))

    if not candidates:
        raise FileNotFoundError(
            f"No latest checkpoint matching net_epoch*.pth found in {saved_path}"
        )

    epoch, checkpoint_path = max(candidates, key=lambda item: item[0])
    print(f"resuming latest checkpoint at epoch {epoch}")
    loaded_state_dict = torch.load(checkpoint_path, map_location='cpu')
    train_utils.check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)
    return epoch, model


def build_isaac_multicav_lidar_for_vis(opencood_dataset, idx, ego_lidar=None,
                                       base_data_dict=None):
    """Load one non-ego CAV lidar and project it into ego frame for BEV plots."""
    if base_data_dict is None:
        try:
            base_data_dict = opencood_dataset.retrieve_base_data(idx)
        except Exception as exc:
            print(f"Skip Isaac multi-CAV lidar visualization at sample {idx}: {exc}")
            return ego_lidar if ego_lidar is not None else np.zeros((0, 4), dtype=np.float32)

    sample_records = _get_opv2v_sample_records(opencood_dataset, idx)

    ego_id = _find_ego_id(base_data_dict)
    if ego_id is None:
        return ego_lidar if ego_lidar is not None else np.zeros((0, 4), dtype=np.float32)

    ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
    if ego_lidar is None:
        ego_lidar = np.asarray(base_data_dict[ego_id].get("lidar_np", []), dtype=np.float32)
        if ego_lidar.size == 0:
            ego_lidar_path = sample_records.get(ego_id, {}).get("lidar")
            if ego_lidar_path and os.path.exists(ego_lidar_path):
                ego_lidar = pcd_utils.pcd_to_np(ego_lidar_path)
    if ego_lidar is None or np.asarray(ego_lidar).size == 0:
        ego_lidar = np.zeros((0, 4), dtype=np.float32)

    extra_lidar = []
    for cav_id, cav_content in base_data_dict.items():
        if cav_id == ego_id:
            continue
        lidar_np = np.asarray(cav_content.get("lidar_np", []))
        if lidar_np.size == 0:
            lidar_path = sample_records.get(cav_id, {}).get("lidar")
            if lidar_path and os.path.exists(lidar_path):
                lidar_np = pcd_utils.pcd_to_np(lidar_path)
        if lidar_np.size == 0:
            continue
        transform = x1_to_x2(cav_content["params"]["lidar_pose"], ego_pose)
        projected_xyz = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transform)
        projected_lidar = np.zeros((projected_xyz.shape[0], max(4, lidar_np.shape[1])),
                                   dtype=np.float32)
        projected_lidar[:, :3] = projected_xyz
        if lidar_np.shape[1] > 3:
            projected_lidar[:, 3] = lidar_np[:, 3]
        extra_lidar.append(projected_lidar[:, :4])
        break

    if not extra_lidar:
        return ego_lidar
    return [ego_lidar] + extra_lidar


def _sorted_cav_ids(base_data_dict):
    def sort_key(cav_id):
        try:
            return (0, int(cav_id))
        except (TypeError, ValueError):
            return (1, str(cav_id))

    return sorted(base_data_dict.keys(), key=sort_key)


def _resolve_opv2v_sample_dataset_and_idx(opencood_dataset, idx):
    """Map an all-ego expanded index back to the underlying raw sample."""
    if hasattr(opencood_dataset, "expanded_sample_info"):
        info = opencood_dataset.expanded_sample_info(idx)
        base_dataset = getattr(opencood_dataset, "dataset", opencood_dataset)
        return base_dataset, info["sample_idx"]
    return opencood_dataset, idx


def _locate_opv2v_sample(opencood_dataset, idx):
    opencood_dataset, idx = _resolve_opv2v_sample_dataset_and_idx(
        opencood_dataset, idx
    )
    scenario_index = 0
    for i, ele in enumerate(opencood_dataset.len_record):
        if idx < ele:
            scenario_index = i
            break
    scenario_database = opencood_dataset.scenario_database[scenario_index]
    timestamp_index = idx if scenario_index == 0 else idx - opencood_dataset.len_record[scenario_index - 1]
    timestamp_key = opencood_dataset.return_timestamp_key(
        scenario_database, timestamp_index
    )
    return scenario_database, timestamp_key


def _get_opv2v_sample_records(opencood_dataset, idx):
    if hasattr(opencood_dataset, "random_pair_sample_records"):
        return opencood_dataset.random_pair_sample_records(idx)
    scenario_database, timestamp_key = _locate_opv2v_sample(opencood_dataset, idx)
    records = OrderedDict()
    for cav_id, cav_content in scenario_database.items():
        if cav_id == "ego":
            continue
        records[cav_id] = cav_content[timestamp_key]
    return records


def _depth_to_rgb(depth_values, min_depth=2.0, max_depth=30.0):
    """Small jet-like colormap without adding a matplotlib dependency here."""
    t = (depth_values - min_depth) / max(max_depth - min_depth, 1e-6)
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def render_lidar_projection_black(cav_content, camera_id=0):
    """Project a CAV's lidar into its RGB camera frame on a black canvas."""
    camera_data = cav_content.get("camera_data", [])
    if len(camera_data) <= camera_id:
        return None

    rgb = camera_data[camera_id].convert("RGB")
    img_w, img_h = rgb.size
    params = cav_content.get("params", {})
    camera_key = f"camera{camera_id}"
    if camera_key not in params:
        return None

    lidar_np = np.asarray(cav_content.get("lidar_np", []), dtype=np.float32)
    if lidar_np.size == 0:
        return Image.fromarray(np.zeros((img_h, img_w, 3), dtype=np.uint8))
    pc = lidar_np[:, :3]

    intrinsic = np.asarray(params[camera_key]["intrinsic"], dtype=np.float32)
    camera_to_lidar = np.asarray(params[camera_key]["extrinsic"], dtype=np.float32)
    lidar_to_camera = np.linalg.inv(camera_to_lidar)

    pts_lidar = np.hstack((pc, np.ones((pc.shape[0], 1), dtype=np.float32)))
    pts_cam = (lidar_to_camera @ pts_lidar.T).T
    x_cv, y_cv, z_cv = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid = z_cv > 0.2
    if not np.any(valid):
        return Image.fromarray(np.zeros((img_h, img_w, 3), dtype=np.uint8))

    x_cv, y_cv, z_cv = x_cv[valid], y_cv[valid], z_cv[valid]
    u = (intrinsic[0, 0] * x_cv / z_cv) + intrinsic[0, 2]
    v = (intrinsic[1, 1] * y_cv / z_cv) + intrinsic[1, 2]
    img_valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    if not np.any(img_valid):
        return Image.fromarray(np.zeros((img_h, img_w, 3), dtype=np.uint8))

    u = u[img_valid].astype(np.int32)
    v = v[img_valid].astype(np.int32)
    colors = _depth_to_rgb(z_cv[img_valid])

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        uu = np.clip(u + du, 0, img_w - 1)
        vv = np.clip(v + dv, 0, img_h - 1)
        canvas[vv, uu] = colors
    return Image.fromarray(canvas)


ISAAC_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _box_tensor_to_numpy(box_tensor):
    if box_tensor is None:
        return None
    if isinstance(box_tensor, torch.Tensor):
        box_tensor = box_tensor.detach().cpu().numpy()
    boxes = np.asarray(box_tensor, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    if boxes.ndim == 2 and boxes.shape == (8, 3):
        boxes = boxes[None, ...]
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
        return None
    return boxes


def render_3d_boxes_on_rgb(
    cav_content,
    box_tensor,
    color,
    camera_id=0,
    width=3,
    labels=None,
    class_names=None,
    class_colors=None,
):
    camera_data = cav_content.get("camera_data", [])
    if len(camera_data) <= camera_id:
        return None
    rgb = camera_data[camera_id].convert("RGB").copy()
    boxes = _box_tensor_to_numpy(box_tensor)
    if boxes is None:
        # A detector may represent "no prediction" as None rather than an
        # empty tensor. Keep the camera frame visible and simply draw no boxes.
        return rgb

    img_w, img_h = rgb.size
    if boxes.shape[0] == 0:
        return rgb

    params = cav_content.get("params", {})
    camera_key = f"camera{camera_id}"
    if camera_key not in params:
        return None

    intrinsic = np.asarray(params[camera_key]["intrinsic"], dtype=np.float32)
    camera_to_lidar = np.asarray(params[camera_key]["extrinsic"], dtype=np.float32)
    lidar_to_camera = np.linalg.inv(camera_to_lidar)
    draw = ImageDraw.Draw(rgb)

    labels_np = None
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels_np = np.asarray(labels).reshape(-1)

    for box_idx, corners in enumerate(boxes):
        box_color = color
        class_text = None
        if labels_np is not None and box_idx < len(labels_np):
            class_id = int(labels_np[box_idx])
            if (
                class_names is not None
                and class_colors is not None
                and 0 <= class_id < len(class_names)
                and class_id < len(class_colors)
            ):
                box_color = tuple(int(value) for value in class_colors[class_id])
                class_text = str(class_names[class_id])

        pts_lidar = np.hstack((corners[:, :3], np.ones((8, 1), dtype=np.float32)))
        pts_cam = (lidar_to_camera @ pts_lidar.T).T
        z = pts_cam[:, 2]
        if np.count_nonzero(z > 0.2) < 2:
            continue

        z_safe = np.maximum(z, 1e-5)
        u = (intrinsic[0, 0] * pts_cam[:, 0] / z_safe) + intrinsic[0, 2]
        v = (intrinsic[1, 1] * pts_cam[:, 1] / z_safe) + intrinsic[1, 2]
        u = np.clip(u, -2 * img_w, 3 * img_w)
        v = np.clip(v, -2 * img_h, 3 * img_h)

        for start, end in ISAAC_BOX_EDGES:
            if z[start] <= 0.2 or z[end] <= 0.2:
                continue
            draw.line(
                [(float(u[start]), float(v[start])), (float(u[end]), float(v[end]))],
                fill=box_color,
                width=width,
            )

        if class_text:
            visible = z > 0.2
            text_x = int(np.clip(np.min(u[visible]), 2, img_w - 2))
            text_y = int(np.clip(np.min(v[visible]), 2, img_h - 2))
            text_bbox = draw.textbbox((text_x, text_y), class_text)
            draw.rectangle(
                (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2),
                fill=(0, 0, 0),
            )
            draw.text((text_x, text_y), class_text, fill=box_color)
    return rgb


def save_isaac_ego_3d_projection_views(cav_content, infer_result, prefix):
    if infer_result is None or not cav_content.get("ego", False):
        return
    pred_rgb = render_3d_boxes_on_rgb(
        cav_content,
        infer_result.get("pred_box_tensor"),
        color=tuple(int(value) for value in simple_vis_isaac.PREDICTION_FALLBACK_COLOR),
        labels=infer_result.get("pred_label"),
        class_names=simple_vis_isaac.PREDICTION_CLASS_NAMES,
        class_colors=simple_vis_isaac.PREDICTION_CLASS_COLORS,
    )
    if pred_rgb is not None:
        pred_rgb.save(f"{prefix}_ego_pred_3d_project.png")

    gt_rgb = render_3d_boxes_on_rgb(
        cav_content, infer_result.get("gt_box_tensor"), color=(0, 255, 0)
    )
    if gt_rgb is not None:
        gt_rgb.save(f"{prefix}_ego_gt_3d_project.png")


def _find_ego_id(base_data_dict):
    for cav_id, cav_content in base_data_dict.items():
        if cav_content.get("ego", False):
            return cav_id
    sorted_ids = _sorted_cav_ids(base_data_dict)
    return sorted_ids[0] if sorted_ids else None


class IsaacAllEgoDataset:
    """Expand Isaac samples so every CAV is evaluated once as ego."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_refs = self._build_sample_refs()
        if not self.sample_refs:
            raise RuntimeError("Isaac all-ego evaluation found no CAV samples.")
        cav_counts = [len(cav_ids) for _, cav_ids in self._base_sample_cavs()]
        unique_counts = sorted(set(cav_counts))
        print(
            "Isaac all-ego evaluation enabled: "
            f"{len(self.dataset)} base samples -> {len(self.sample_refs)} ego samples; "
            f"CAV counts per base sample: {unique_counts}"
        )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.sample_refs)

    def _base_sample_cavs(self):
        base_refs = []
        for sample_idx in range(len(self.dataset)):
            scenario_database, _ = _locate_opv2v_sample(self.dataset, sample_idx)
            cav_ids = [
                cav_id for cav_id in _sorted_cav_ids(scenario_database)
                if cav_id != "ego"
            ]
            base_refs.append((sample_idx, cav_ids))
        return base_refs

    def _build_sample_refs(self):
        sample_refs = []
        for sample_idx, cav_ids in self._base_sample_cavs():
            for cav_id in cav_ids:
                sample_refs.append((sample_idx, cav_id))
        return sample_refs

    def expanded_sample_info(self, expanded_idx):
        sample_idx, ego_cav_id = self.sample_refs[expanded_idx]
        return {"sample_idx": sample_idx, "ego_cav_id": ego_cav_id}

    def retrieve_base_data(self, expanded_idx):
        sample_idx, ego_cav_id = self.sample_refs[expanded_idx]
        base_data = self.dataset.retrieve_base_data(sample_idx)
        return self._force_ego(base_data, ego_cav_id)

    @staticmethod
    def _force_ego(base_data, ego_cav_id):
        if ego_cav_id not in base_data:
            raise KeyError(f"Requested ego CAV {ego_cav_id} is missing from sample.")
        ordered = OrderedDict()
        ordered[ego_cav_id] = base_data[ego_cav_id]
        for cav_id in _sorted_cav_ids(base_data):
            if cav_id != ego_cav_id:
                ordered[cav_id] = base_data[cav_id]
        for cav_id, cav_content in ordered.items():
            cav_content["ego"] = cav_id == ego_cav_id
        return ordered

    def __getitem__(self, expanded_idx):
        sample_idx, ego_cav_id = self.sample_refs[expanded_idx]
        forced_base_data = self.retrieve_base_data(expanded_idx)
        original_retrieve_base_data = self.dataset.retrieve_base_data
        self.dataset.retrieve_base_data = lambda _: forced_base_data
        try:
            item = self.dataset[sample_idx]
        finally:
            self.dataset.retrieve_base_data = original_retrieve_base_data
        if item is not None and "ego" in item:
            item["ego"]["base_sample_idx"] = sample_idx
            item["ego"]["ego_cav_id"] = ego_cav_id
        return item

    def collate_batch_test(self, batch):
        return self.dataset.collate_batch_test(batch)


class IsaacRandomPairDataset:
    """Keep the original ego frame and inject one deterministic random partner.

    The source partner is always taken from a different sample in the same
    evaluation split.  Pair assignments are built once from a private RNG, so
    they do not depend on global NumPy state or DataLoader worker scheduling.
    """

    def __init__(self, dataset, seed=303):
        self.dataset = dataset
        self.seed = int(seed)
        self.sample_refs = self._build_sample_refs()
        digest_payload = "\n".join(
            f"{sample_idx}|{ego_cav_id}|{partner_sample_idx}|{partner_source_cav_id}|{partner_output_cav_id}"
            for (
                sample_idx,
                ego_cav_id,
                partner_sample_idx,
                partner_source_cav_id,
                partner_output_cav_id,
            ) in self.sample_refs
        )
        self.pairing_digest = hashlib.sha256(
            digest_payload.encode("utf-8")
        ).hexdigest()[:16]
        preview = [
            f"{sample_idx}->{partner_sample_idx}:{partner_source_cav_id}"
            for (
                sample_idx,
                _,
                partner_sample_idx,
                partner_source_cav_id,
                _,
            ) in self.sample_refs[:5]
        ]
        print(
            "Isaac random-pair evaluation enabled: "
            f"{len(self.sample_refs)} ego samples, seed={self.seed}, "
            f"digest={self.pairing_digest}, preview={preview}"
        )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.sample_refs)

    def _sample_cav_ids(self, sample_idx):
        scenario_database, _ = _locate_opv2v_sample(self.dataset, sample_idx)
        cav_ids = [
            cav_id
            for cav_id in _sorted_cav_ids(scenario_database)
            if cav_id != "ego"
        ]
        if not cav_ids:
            raise RuntimeError(f"Isaac sample {sample_idx} has no CAV records.")
        return cav_ids[0], cav_ids[1:]

    @staticmethod
    def _derangement(values, rng):
        values = np.asarray(values, dtype=np.int64)
        if values.size < 2:
            raise RuntimeError(
                "Isaac random-pair evaluation requires at least two samples "
                "with non-ego partner agents."
            )
        for _ in range(1000):
            shuffled = rng.permutation(values)
            if np.all(shuffled != values):
                return shuffled.tolist()
        # Deterministic fallback; the random rotation is still seed-controlled.
        shift = int(rng.randint(1, values.size))
        return np.roll(values, shift).tolist()

    def _build_sample_refs(self):
        rng = np.random.RandomState(self.seed)
        cav_layouts = [
            self._sample_cav_ids(sample_idx)
            for sample_idx in range(len(self.dataset))
        ]
        eligible_partner_samples = [
            sample_idx
            for sample_idx, (_, partner_ids) in enumerate(cav_layouts)
            if partner_ids
        ]
        if len(eligible_partner_samples) < 2:
            raise RuntimeError(
                "Isaac random-pair evaluation requires at least two samples "
                "containing a non-ego partner agent."
            )

        all_samples_are_eligible = (
            len(eligible_partner_samples) == len(self.dataset)
        )
        if all_samples_are_eligible:
            partner_sample_indices = self._derangement(
                list(range(len(self.dataset))), rng
            )
        else:
            partner_sample_indices = []
            for sample_idx in range(len(self.dataset)):
                candidates = [
                    candidate
                    for candidate in eligible_partner_samples
                    if candidate != sample_idx
                ]
                if not candidates:
                    raise RuntimeError(
                        f"Isaac sample {sample_idx} has no valid random partner source."
                    )
                partner_sample_indices.append(
                    int(candidates[int(rng.randint(len(candidates)))])
                )

        sample_refs = []
        for sample_idx, partner_sample_idx in enumerate(partner_sample_indices):
            ego_cav_id, target_partner_ids = cav_layouts[sample_idx]
            _, source_partner_ids = cav_layouts[partner_sample_idx]
            if not target_partner_ids:
                # Preserve a stable, non-colliding output key when the target
                # sample itself has no partner slot.
                partner_output_cav_id = f"random_partner_{partner_sample_idx}"
            else:
                partner_output_cav_id = target_partner_ids[0]
            partner_source_cav_id = source_partner_ids[
                int(rng.randint(len(source_partner_ids)))
            ]
            sample_refs.append(
                (
                    sample_idx,
                    ego_cav_id,
                    partner_sample_idx,
                    partner_source_cav_id,
                    partner_output_cav_id,
                )
            )
        return sample_refs

    def expanded_sample_info(self, expanded_idx):
        (
            sample_idx,
            ego_cav_id,
            partner_sample_idx,
            partner_source_cav_id,
            _,
        ) = self.sample_refs[expanded_idx]
        return {
            "sample_idx": sample_idx,
            "ego_cav_id": ego_cav_id,
            "partner_sample_idx": partner_sample_idx,
            "partner_cav_id": partner_source_cav_id,
        }

    def retrieve_base_data(self, expanded_idx):
        (
            sample_idx,
            ego_cav_id,
            partner_sample_idx,
            partner_source_cav_id,
            partner_output_cav_id,
        ) = self.sample_refs[expanded_idx]
        ego_base_data = self.dataset.retrieve_base_data(sample_idx)
        partner_base_data = self.dataset.retrieve_base_data(partner_sample_idx)
        if ego_cav_id not in ego_base_data:
            raise KeyError(
                f"Original ego CAV {ego_cav_id} is missing from sample {sample_idx}."
            )
        if partner_source_cav_id not in partner_base_data:
            raise KeyError(
                "Random partner CAV "
                f"{partner_source_cav_id} is missing from sample {partner_sample_idx}."
            )

        ordered = OrderedDict()
        ordered[ego_cav_id] = ego_base_data[ego_cav_id]
        ordered[partner_output_cav_id] = partner_base_data[partner_source_cav_id]
        for cav_id, cav_content in ordered.items():
            cav_content["ego"] = cav_id == ego_cav_id
        return ordered

    def random_pair_sample_records(self, expanded_idx):
        (
            sample_idx,
            ego_cav_id,
            partner_sample_idx,
            partner_source_cav_id,
            partner_output_cav_id,
        ) = self.sample_refs[expanded_idx]
        ego_records = _get_opv2v_sample_records(self.dataset, sample_idx)
        partner_records = _get_opv2v_sample_records(
            self.dataset, partner_sample_idx
        )
        records = OrderedDict()
        records[ego_cav_id] = ego_records[ego_cav_id]
        records[partner_output_cav_id] = partner_records[partner_source_cav_id]
        return records

    def __getitem__(self, expanded_idx):
        sample_idx = self.sample_refs[expanded_idx][0]
        forced_base_data = self.retrieve_base_data(expanded_idx)
        original_retrieve_base_data = self.dataset.retrieve_base_data
        self.dataset.retrieve_base_data = lambda _: forced_base_data
        try:
            item = self.dataset[sample_idx]
        finally:
            self.dataset.retrieve_base_data = original_retrieve_base_data
        if item is not None and "ego" in item:
            info = self.expanded_sample_info(expanded_idx)
            item["ego"]["base_sample_idx"] = sample_idx
            item["ego"]["ego_cav_id"] = info["ego_cav_id"]
            item["ego"]["random_partner_sample_idx"] = info[
                "partner_sample_idx"
            ]
            item["ego"]["random_partner_cav_id"] = info["partner_cav_id"]
        return item

    def collate_batch_test(self, batch):
        return self.dataset.collate_batch_test(batch)


def _vis_sample_info(opencood_dataset, expanded_idx):
    """Return raw case index and selected ego for human-readable vis paths."""
    if hasattr(opencood_dataset, "expanded_sample_info"):
        info = opencood_dataset.expanded_sample_info(expanded_idx)
        return {
            "expanded_idx": int(expanded_idx),
            "sample_idx": int(info["sample_idx"]),
            "ego_cav_id": info["ego_cav_id"],
        }
    return {
        "expanded_idx": int(expanded_idx),
        "sample_idx": int(expanded_idx),
        "ego_cav_id": None,
    }


def _cav_id_to_path_tag(cav_id):
    if cav_id is None:
        return None
    return str(cav_id).replace(os.sep, "_").replace(" ", "_")


def _isaac_vis_dir_and_index(vis_save_path_root, opencood_dataset, expanded_idx):
    info = _vis_sample_info(opencood_dataset, expanded_idx)
    sample_idx = info["sample_idx"]
    sample_vis_dir = os.path.join(vis_save_path_root, f"{sample_idx:05d}")
    ego_tag = _cav_id_to_path_tag(info["ego_cav_id"])
    if ego_tag is not None:
        sample_vis_dir = os.path.join(sample_vis_dir, f"ego_{ego_tag}")
    return sample_vis_dir, sample_idx, info


def _isaac_bev_vis_filename(bev_index, vis_sample_info):
    ego_tag = _cav_id_to_path_tag(vis_sample_info.get("ego_cav_id"))
    if ego_tag is None:
        return "bev_%05d.png" % bev_index
    return "bev_%05d_%s.png" % (bev_index, ego_tag)


def _unwrap_isaac_dataset(opencood_dataset):
    dataset = opencood_dataset
    visited = set()
    while not hasattr(dataset, "scenario_database") and hasattr(dataset, "dataset"):
        dataset_id = id(dataset)
        if dataset_id in visited:
            break
        visited.add(dataset_id)
        dataset = dataset.dataset
    return dataset


def _isaac_case_metadata(opencood_dataset, sample_idx):
    dataset = _unwrap_isaac_dataset(opencood_dataset)
    meta = OrderedDict()
    meta["sample_idx"] = int(sample_idx)
    meta["scenario_index"] = None
    meta["scenario_name"] = None
    meta["scenario_tag"] = "case_unknown"
    meta["timestamp_index"] = None
    meta["timestamp_key"] = None

    if not hasattr(dataset, "scenario_database") or not hasattr(dataset, "len_record"):
        return meta

    scenario_index = 0
    for idx, end_idx in enumerate(dataset.len_record):
        if sample_idx < end_idx:
            scenario_index = idx
            break
    previous_end = 0 if scenario_index == 0 else dataset.len_record[scenario_index - 1]
    timestamp_index = int(sample_idx - previous_end)
    scenario_database = dataset.scenario_database[scenario_index]
    timestamp_key = dataset.return_timestamp_key(scenario_database, timestamp_index)
    scenario_folder = None
    if hasattr(dataset, "scenario_folders") and scenario_index < len(dataset.scenario_folders):
        scenario_folder = dataset.scenario_folders[scenario_index]
    scenario_name = os.path.basename(scenario_folder) if scenario_folder else f"case_{scenario_index:03d}"
    scenario_tag = f"case_{scenario_index:03d}_{_cav_id_to_path_tag(scenario_name)}"

    meta["scenario_index"] = int(scenario_index)
    meta["scenario_name"] = scenario_name
    meta["scenario_tag"] = scenario_tag
    meta["timestamp_index"] = timestamp_index
    meta["timestamp_key"] = str(timestamp_key)
    return meta


def _isaac_case_output_root(base_root, opencood_dataset, sample_idx, split_by_case):
    if not split_by_case:
        return base_root
    case_meta = _isaac_case_metadata(opencood_dataset, sample_idx)
    return os.path.join(base_root, case_meta["scenario_tag"])


def save_isaac_agent_rgb_lidar_views(opencood_dataset, idx, vis_save_path_root, bev_index,
                                     base_data_dict=None, only_cav_id=None,
                                     infer_result=None):
    """Save per-agent RGB, lidar projection, and ego 3D box projection views."""
    if base_data_dict is None:
        try:
            base_data_dict = opencood_dataset.retrieve_base_data(idx)
        except Exception as exc:
            print(f"Skip Isaac RGB/LiDAR camera visualization at sample {idx}: {exc}")
            return None

    sample_records = _get_opv2v_sample_records(opencood_dataset, idx)
    for agent_idx, cav_id in enumerate(_sorted_cav_ids(base_data_dict)[:2]):
        if only_cav_id is not None and cav_id != only_cav_id:
            continue
        cav_content = base_data_dict[cav_id]
        sample_record = sample_records.get(cav_id, {})
        if hasattr(opencood_dataset, "_normalize_camera_params"):
            opencood_dataset._normalize_camera_params(cav_content["params"], 0)

        if not cav_content.get("camera_data"):
            camera_files = sample_record.get("cameras", [])
            if camera_files and os.path.exists(camera_files[0]):
                cav_content["camera_data"] = [Image.open(camera_files[0]).convert("RGB")]

        if "lidar_np" not in cav_content or np.asarray(cav_content["lidar_np"]).size == 0:
            lidar_path = sample_record.get("lidar")
            if lidar_path and os.path.exists(lidar_path):
                cav_content["lidar_np"] = pcd_utils.pcd_to_np(lidar_path)

        camera_data = cav_content.get("camera_data", [])
        if not camera_data:
            continue

        prefix = os.path.join(vis_save_path_root, f"bev_{bev_index:05d}_{agent_idx}")
        label_yaml = sample_record.get("yaml")
        if label_yaml and os.path.exists(label_yaml):
            shutil.copyfile(label_yaml, f"{prefix}.yaml")
        elif label_yaml:
            print(
                f"Skip Isaac label YAML for sample {idx} cav {cav_id}: "
                f"{label_yaml} not found"
            )
        camera_data[0].convert("RGB").save(f"{prefix}_RGB.png")
        lidar_projection = render_lidar_projection_black(cav_content, camera_id=0)
        if lidar_projection is not None:
            lidar_projection.save(f"{prefix}_LiDAR.png")
        save_isaac_ego_3d_projection_views(cav_content, infer_result, prefix)


    return base_data_dict


class FFmpegStreamVideoWriter:
    """Write fixed-size RGB PIL frames to one persistent ffmpeg process."""

    def __init__(self, output_path, fps):
        self.output_path = os.path.abspath(os.path.expanduser(output_path))
        self.fps = float(fps)
        self.process = None
        self.frame_size = None
        self.frame_count = 0
        self.closed = False
        atexit.register(self.close)

    def _start(self, frame):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.frame_size = frame.size
        width, height = self.frame_size
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.output_path,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )
        print(
            "Streaming video enabled: "
            f"{self.output_path}, size={width}x{height}, fps={self.fps:g}"
        )

    def write(self, frame):
        if self.closed:
            raise RuntimeError("Cannot write to a closed streaming video.")
        frame = frame.convert("RGB")
        if self.process is None:
            self._start(frame)
        if frame.size != self.frame_size:
            raise ValueError(
                f"Streaming frame size changed from {self.frame_size} to {frame.size}"
            )
        try:
            self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            return_code = self.process.poll()
            raise RuntimeError(
                f"ffmpeg video pipe closed unexpectedly (return code {return_code})"
            ) from exc
        self.frame_count += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with return code {return_code}: {self.output_path}"
            )
        print(
            f"Streaming video complete: {self.output_path} "
            f"({self.frame_count} frame(s))"
        )


_STREAM_VIDEO_LABEL_FONT = None
_STREAM_VIDEO_LEGEND_FONT = None


def _get_stream_video_label_font():
    """Load one reusable, presentation-sized label font."""
    global _STREAM_VIDEO_LABEL_FONT
    if _STREAM_VIDEO_LABEL_FONT is None:
        font_path = "DejaVuSans-Bold.ttf"
        try:
            _STREAM_VIDEO_LABEL_FONT = ImageFont.truetype(font_path, 20)
        except OSError:
            _STREAM_VIDEO_LABEL_FONT = ImageFont.load_default()
    return _STREAM_VIDEO_LABEL_FONT


def _get_stream_video_legend_font():
    """Load one reusable font for the BEV class-color legend."""
    global _STREAM_VIDEO_LEGEND_FONT
    if _STREAM_VIDEO_LEGEND_FONT is None:
        font_path = "DejaVuSans.ttf"
        try:
            _STREAM_VIDEO_LEGEND_FONT = ImageFont.truetype(font_path, 16)
        except OSError:
            _STREAM_VIDEO_LEGEND_FONT = ImageFont.load_default()
    return _STREAM_VIDEO_LEGEND_FONT


def _fit_stream_video_tile(image, label, width=640, height=400):
    """Letterbox one PIL/numpy image and add a large green panel label."""
    if image is None:
        image = Image.new("RGB", (width, height), (0, 0, 0))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    else:
        image = image.convert("RGB")

    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), (0, 0, 0))
    left = (width - image.width) // 2
    top = (height - image.height) // 2
    tile.paste(image, (left, top))

    draw = ImageDraw.Draw(tile)
    font = _get_stream_video_label_font()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = max(12, width - text_width - 18)
    y = 14
    draw.rectangle(
        (x - 8, y - 6, x + text_width + 8, y + text_height + 7),
        fill=(0, 0, 0),
    )
    draw.text((x, y), label, font=font, fill=(0, 255, 0))
    return tile


def _fit_stream_video_bev_tile_with_legend(
    image, label, width=640, height=400
):
    """Place the BEV at the left and its box-color legend at the right."""
    if image is None:
        image = Image.new("RGB", (400, height), (0, 0, 0))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    else:
        image = image.convert("RGB")

    map_region_left = 8
    map_region_width = 400
    image.thumbnail((map_region_width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), (0, 0, 0))
    image_left = map_region_left + (map_region_width - image.width) // 2
    image_top = (height - image.height) // 2
    tile.paste(image, (image_left, image_top))

    draw = ImageDraw.Draw(tile)
    panel_font = _get_stream_video_label_font()
    text_bbox = draw.textbbox((0, 0), label, font=panel_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    # Match the other three tiles: right-align every panel title to the same
    # 18 px inset from the full 640 px tile, independent of content layout.
    title_x = max(12, width - text_width - 18)
    title_y = 14
    draw.rectangle(
        (
            title_x - 8,
            title_y - 6,
            title_x + text_width + 8,
            title_y + text_height + 7,
        ),
        fill=(0, 0, 0),
    )
    draw.text(
        (title_x, title_y),
        label,
        font=panel_font,
        fill=(0, 255, 0),
    )

    divider_x = map_region_left + map_region_width + 8
    draw.line((divider_x, 50, divider_x, height - 18), fill=(70, 70, 70), width=1)

    legend_font = _get_stream_video_legend_font()
    legend_x = divider_x + 14
    draw.text(
        (legend_x, 54),
        "Box colors",
        font=legend_font,
        fill=(235, 235, 235),
    )
    legend_entries = [("ground truth", (0, 255, 0))]
    legend_entries.extend(
        (
            class_name.replace("_", " "),
            tuple(int(value) for value in class_color),
        )
        for class_name, class_color in zip(
            simple_vis_isaac.PREDICTION_CLASS_NAMES,
            simple_vis_isaac.PREDICTION_CLASS_COLORS,
        )
    )
    swatch_size = 18
    legend_y = 84
    row_height = 37
    for entry_idx, (class_name, class_color) in enumerate(legend_entries):
        row_y = legend_y + entry_idx * row_height
        draw.rectangle(
            (
                legend_x,
                row_y,
                legend_x + swatch_size,
                row_y + swatch_size,
            ),
            fill=class_color,
            outline=(235, 235, 235),
            width=1,
        )
        draw.text(
            (legend_x + swatch_size + 10, row_y),
            class_name,
            font=legend_font,
            fill=(235, 235, 235),
        )
    return tile


def _ensure_stream_camera_data(opencood_dataset, idx, base_data_dict):
    """Load only missing camera data needed by the streaming panels."""
    sample_records = _get_opv2v_sample_records(opencood_dataset, idx)
    for cav_id in _sorted_cav_ids(base_data_dict):
        cav_content = base_data_dict[cav_id]
        if hasattr(opencood_dataset, "_normalize_camera_params"):
            opencood_dataset._normalize_camera_params(cav_content["params"], 0)
        if cav_content.get("camera_data"):
            continue
        camera_files = sample_records.get(cav_id, {}).get("cameras", [])
        if camera_files and os.path.exists(camera_files[0]):
            cav_content["camera_data"] = [
                Image.open(camera_files[0]).convert("RGB")
            ]
    return base_data_dict


def build_isaac_stream_video_frame(
    opencood_dataset,
    idx,
    base_data_dict,
    infer_result,
    vis_lidar,
    pc_range,
    left_hand,
    fusion_method,
    ego_only_infer_result=None,
):
    """Build one in-memory legacy row or 2x2 fusion-comparison frame."""
    base_data_dict = _ensure_stream_camera_data(
        opencood_dataset, idx, base_data_dict
    )
    ego_id = _find_ego_id(base_data_dict)
    if ego_id is None:
        raise RuntimeError(f"No ego CAV found for streaming frame {idx}")
    ego_content = base_data_dict[ego_id]

    intermediate_pred_rgb = render_3d_boxes_on_rgb(
        ego_content,
        infer_result.get("pred_box_tensor"),
        color=tuple(
            int(value) for value in simple_vis_isaac.PREDICTION_FALLBACK_COLOR
        ),
        labels=infer_result.get("pred_label"),
        class_names=simple_vis_isaac.PREDICTION_CLASS_NAMES,
        class_colors=simple_vis_isaac.PREDICTION_CLASS_COLORS,
    )
    ego_only_pred_rgb = None
    if ego_only_infer_result is not None:
        ego_only_pred_rgb = render_3d_boxes_on_rgb(
            ego_content,
            ego_only_infer_result.get("pred_box_tensor"),
            color=tuple(
                int(value) for value in simple_vis_isaac.PREDICTION_FALLBACK_COLOR
            ),
            labels=ego_only_infer_result.get("pred_label"),
            class_names=simple_vis_isaac.PREDICTION_CLASS_NAMES,
            class_colors=simple_vis_isaac.PREDICTION_CLASS_COLORS,
        )
    gt_rgb = render_3d_boxes_on_rgb(
        ego_content,
        infer_result.get("gt_box_tensor"),
        color=(0, 255, 0),
    )
    intermediate_bev_rgb = simple_vis_isaac.visualize(
        infer_result,
        vis_lidar,
        pc_range,
        save_path=None,
        method="bev",
        left_hand=left_hand,
        return_image=True,
    )

    partner_rgb = None
    for cav_id in _sorted_cav_ids(base_data_dict):
        if cav_id == ego_id:
            continue
        camera_data = base_data_dict[cav_id].get("camera_data", [])
        if camera_data:
            partner_rgb = camera_data[0].convert("RGB")
            break

    if ego_only_infer_result is not None:
        images_and_labels = (
            (ego_only_pred_rgb, "Ego Agent (No Fusion)"),
            (intermediate_pred_rgb, "Ego Agent (Intermediate Fusion)"),
            (partner_rgb, "Collaborative Agent"),
            (intermediate_bev_rgb, "Ego Agent Intermediate BEV"),
        )
        grid_rows, grid_columns = 2, 2
    else:
        fusion_label = {
            "no": "No Fusion",
            "late": "Late Fusion",
            "intermediate": "Intermediate Fusion",
        }[fusion_method]
        if fusion_method == "no":
            images_and_labels = (
                (gt_rgb, "Ego GT"),
                (intermediate_pred_rgb, "No Fusion Prediction"),
                (intermediate_bev_rgb, "No Fusion BEV"),
            )
        else:
            images_and_labels = (
                (partner_rgb, "Collaborative RGB"),
                (intermediate_pred_rgb, f"{fusion_label} Prediction"),
                (intermediate_bev_rgb, f"{fusion_label} BEV"),
            )
        grid_rows, grid_columns = 1, 3

    if ego_only_infer_result is not None:
        tiles = [
            _fit_stream_video_tile(image, label)
            for image, label in images_and_labels[:3]
        ]
        bev_image, bev_label = images_and_labels[3]
        tiles.append(
            _fit_stream_video_bev_tile_with_legend(bev_image, bev_label)
        )
    else:
        tiles = [
            _fit_stream_video_tile(image, label)
            for image, label in images_and_labels
        ]
    tile_width = tiles[0].width
    tile_height = tiles[0].height
    frame = Image.new(
        "RGB",
        (grid_columns * tile_width, grid_rows * tile_height),
        (0, 0, 0),
    )
    for tile_idx, tile in enumerate(tiles):
        row_idx = tile_idx // grid_columns
        column_idx = tile_idx % grid_columns
        frame.paste(tile, (column_idx * tile_width, row_idx * tile_height))
    return frame


def main():
    opt = apply_cli_shortcuts(test_parser())

    if opt.video_frame_stride < 1:
        raise ValueError("--video_frame_stride must be >= 1")
    if opt.video_fps <= 0:
        raise ValueError("--video_fps must be > 0")
    if opt.video_only and not opt.stream_video_output:
        raise ValueError("--video_only requires --stream_video_output")
    if opt.video_compare_fusion and opt.ego_only:
        raise ValueError(
            "--video_compare_fusion internally runs strict ego-only and full "
            "intermediate inference; do not also pass --ego_only"
        )
    if opt.video_only:
        opt.save_demo_output = False
        opt.save_full_vis_output = False
        opt.save_vis_interval = 0
        print(
            "Video-only mode: AP/statistics, feature visualization, and "
            "per-frame image files are disabled."
        )
    elif opt.save_demo_output:
        opt.save_full_vis_output = True

    if opt.video_compare_fusion:
        opt.note += '_video_compare'
        print('Video fusion comparison enabled: strict ego-only + full intermediate.')

    assert opt.fusion_method in ['late', 'intermediate', 'no']

    if opt.random_pair and opt.all_ego:
        raise ValueError('--random_pair and --all_ego are mutually exclusive')
    if opt.random_pair and opt.ego_only:
        raise ValueError('--random_pair and --ego_only are mutually exclusive')

    hypes = yaml_utils.load_yaml(None, opt)
    if not is_isaac_center_head(hypes):
        loss_name = hypes.get("loss", {}).get("core_method")
        raise ValueError(
            "InCoP inference supports center_head_loss configs only; "
            f"got {loss_name!r}."
        )
    class_names = hypes.get("postprocess", {}).get("class_names", []) or []
    if len(class_names) <= 1:
        raise ValueError(
            "InCoP inference requires a multi-class postprocess.class_names "
            "configuration."
        )

    if opt.ego_only:
        if opt.fusion_method != 'intermediate':
            raise ValueError('--ego_only is intended for --fusion_method intermediate')
        original_comm_range = float(hypes.get('comm_range', 0.0))
        hypes['comm_range'] = 0.0
        opt.note += '_ego_only'
        print(
            'Ego-only intermediate inference enabled: '
            f'comm_range {original_comm_range:g} -> 0 m'
        )

    if opt.score_threshold is not None:
        if 'target_args' in hypes['postprocess']:
            hypes['postprocess']['target_args']['score_threshold'] = opt.score_threshold
        hypes['postprocess']['anchor_args']['score_threshold'] = opt.score_threshold
    if opt.nms_thresh is not None:
        hypes['postprocess']['nms_thresh'] = opt.nms_thresh

    if 'heter' in hypes and opt.range:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        range_values = [float(v.strip()) for v in opt.range.split(',') if v.strip()]
        is_isaac_dataset = any(
            "IsaacSim" in hypes.get(key, "")
            for key in ("test_dir", "validate_dir")
        )
        if is_isaac_dataset and len(range_values) != 4:
            raise ValueError(
                "IsaacSim --range must be x_min,y_min,x_max,y_max "
                "to preserve the front-only lidar range."
            )
        if len(range_values) == 2:
            x_min, x_max = -range_values[0], range_values[0]
            y_min, y_max = -range_values[1], range_values[1]
        elif len(range_values) == 4:
            x_min, y_min, x_max, y_max = range_values
        else:
            raise ValueError("--range should be either x_half,y_half or x_min,y_min,x_max,y_max")
        opt.note += (
            f"_r{_format_float_tag(x_min)}to{_format_float_tag(x_max)}"
            f"_y{_format_float_tag(y_min)}to{_format_float_tag(y_max)}"
        )

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

    if opt.eval_dataset_dir:
        requested_eval_dir = os.path.expanduser(opt.eval_dataset_dir)
        requested_eval_dir = os.path.abspath(requested_eval_dir)
        if not os.path.isdir(requested_eval_dir):
            raise FileNotFoundError(
                f"--eval_dataset_dir does not exist or is not a directory: "
                f"{requested_eval_dir}"
            )
        dataset_tag = _evaluation_dataset_tag(requested_eval_dir)
        opt.eval_dataset_dir = requested_eval_dir
        opt.note += f"_data_{dataset_tag}"
        if opt.eval_split == 'test':
            hypes['test_dir'] = requested_eval_dir
        else:
            hypes['validate_dir'] = requested_eval_dir
        print(
            "Evaluation dataset override enabled: "
            f"split={opt.eval_split}, directory={requested_eval_dir}, "
            f"output_tag=data_{dataset_tag}"
        )

    if opt.eval_split == 'test':
        if 'test_dir' not in hypes or not hypes['test_dir']:
            raise ValueError("--eval_split test requires test_dir in the loaded config")
        selected_eval_dir = hypes['test_dir']
        hypes['validate_dir'] = selected_eval_dir
    else:
        if 'validate_dir' not in hypes or not hypes['validate_dir']:
            raise ValueError("--eval_split val requires validate_dir in the loaded config")
        selected_eval_dir = hypes['validate_dir']
        opt.note += '_val'

    if (
        opt.eval_split == 'test'
        and ("OPV2V" in selected_eval_dir or "v2xsim" in selected_eval_dir)
    ):
        assert "test" in selected_eval_dir

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    selected_eval_dir_upper = selected_eval_dir.upper()
    left_hand = (
        "OPV2V" in selected_eval_dir_upper
        or "V2XSET" in selected_eval_dir_upper
        # The converted real-world split follows the same lidar convention as
        # the Isaac/OPV2V data even though its directory name omits "OPV2V".
        or "REAL_WORLD" in selected_eval_dir_upper
    )

    print(f"Inference split: {opt.eval_split}, dataset dir: {selected_eval_dir}")
    print(f"Left hand visualizing: {left_hand}")

    opt.eval_modality_assignment_info = (
        _prepare_evaluation_modality_assignment(hypes, selected_eval_dir)
    )

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = load_selected_saved_model(
        saved_path, model, opt.checkpoint_mode
    )
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    pose_noise_enabled = any(
        abs(value) > 0.0
        for value in (
            opt.pose_noise_pos_std,
            opt.pose_noise_rot_std,
            opt.pose_noise_pos_mean,
            opt.pose_noise_rot_mean,
        )
    )
    noise_setting = OrderedDict()
    noise_setting["add_noise"] = pose_noise_enabled
    noise_setting["args"] = {
        "pos_std": opt.pose_noise_pos_std,
        "rot_std": opt.pose_noise_rot_std,
        "pos_mean": opt.pose_noise_pos_mean,
        "rot_mean": opt.pose_noise_rot_mean,
    }
    hypes["noise_setting"] = noise_setting
    if pose_noise_enabled:
        opt.note += (
            f"_noise_t{_format_float_tag(opt.pose_noise_pos_std)}"
            f"_r{_format_float_tag(opt.pose_noise_rot_std)}"
        )
        print(
            "Isaac pose noise enabled: "
            f"pos_std={opt.pose_noise_pos_std}m, "
            f"rot_std={opt.pose_noise_rot_std}deg, "
            f"pos_mean={opt.pose_noise_pos_mean}m, "
            f"rot_mean={opt.pose_noise_rot_mean}deg"
        )
    else:
        print("Isaac pose noise disabled: pos_std=0m, rot_std=0deg")

    video_ego_hypes = None
    if opt.video_compare_fusion:
        video_ego_hypes = copy.deepcopy(hypes)
        original_comm_range = float(video_ego_hypes.get("comm_range", 0.0))
        video_ego_hypes["comm_range"] = 0.0
        print(
            "Video comparison ego-only dataset: "
            f"comm_range {original_comm_range:g} -> 0 m"
        )

    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)
    if opt.random_pair and "IsaacSim" not in selected_eval_dir:
        raise ValueError('--random_pair is currently supported only for IsaacSim datasets')
    if "IsaacSim" in selected_eval_dir and opt.random_pair:
        opencood_dataset = IsaacRandomPairDataset(
            opencood_dataset, seed=opt.random_pair_seed
        )
        opt.random_pair_digest = opencood_dataset.pairing_digest
        opt.note += f"_random_pair_seed{opt.random_pair_seed}"
    elif "IsaacSim" in selected_eval_dir and opt.all_ego:
        opencood_dataset = IsaacAllEgoDataset(opencood_dataset)
        opt.note += "_all_ego"
    elif "IsaacSim" in selected_eval_dir:
        print("Isaac fixed-ego inference enabled: robot 0 is ego, robot 1 is collaborative.")

    video_ego_dataset = None
    if opt.video_compare_fusion:
        video_ego_dataset = build_dataset(
            video_ego_hypes, visualize=False, train=False
        )
        if "IsaacSim" in selected_eval_dir and opt.all_ego:
            video_ego_dataset = IsaacAllEgoDataset(video_ego_dataset)
        if len(video_ego_dataset) != len(opencood_dataset):
            raise RuntimeError(
                "Video comparison dataset length mismatch: "
                f"intermediate={len(opencood_dataset)}, "
                f"ego_only={len(video_ego_dataset)}"
            )

    sample_indices = parse_sample_indices(opt.sample_indices)
    data_loader_dataset = opencood_dataset
    iteration_indices = None
    if sample_indices is not None:
        dataset_len = len(opencood_dataset)
        invalid = [idx for idx in sample_indices if idx >= dataset_len]
        if invalid:
            raise ValueError(
                f"sample_indices out of range for dataset length {dataset_len}: {invalid}"
            )
        if opt.stream_video_output and opt.video_frame_stride > 1:
            sample_indices = sample_indices[::opt.video_frame_stride]
        if opt.max_samples is not None:
            sample_indices = sample_indices[:opt.max_samples]
        iteration_indices = sample_indices
        data_loader_dataset = Subset(opencood_dataset, sample_indices)
        print(f"Isaac selected sample indices: {sample_indices}")
    elif opt.stream_video_output and opt.video_frame_stride > 1:
        iteration_indices = list(
            range(0, len(opencood_dataset), opt.video_frame_stride)
        )
        if opt.max_samples is not None:
            iteration_indices = iteration_indices[:opt.max_samples]
        data_loader_dataset = Subset(opencood_dataset, iteration_indices)
        print(
            "Streaming video frame selection: "
            f"stride={opt.video_frame_stride}, "
            f"{len(iteration_indices)}/{len(opencood_dataset)} samples"
        )

    video_ego_loader_dataset = video_ego_dataset
    if video_ego_dataset is not None and iteration_indices is not None:
        video_ego_loader_dataset = Subset(
            video_ego_dataset, iteration_indices
        )

    data_loader_kwargs = {
        "batch_size": 1,
        "num_workers": opt.num_workers,
        "collate_fn": opencood_dataset.collate_batch_test,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if opt.num_workers > 0:
        data_loader_kwargs["prefetch_factor"] = opt.prefetch_factor
        data_loader_kwargs["persistent_workers"] = opt.persistent_workers
    data_loader = DataLoader(data_loader_dataset, **data_loader_kwargs)

    video_ego_data_loader = None
    if video_ego_loader_dataset is not None:
        video_ego_loader_kwargs = dict(data_loader_kwargs)
        video_ego_loader_kwargs["collate_fn"] = (
            video_ego_dataset.collate_batch_test
        )
        video_ego_data_loader = DataLoader(
            video_ego_loader_dataset, **video_ego_loader_kwargs
        )
    
    # Create the dictionary for evaluation
    result_stat = init_object_result_stat()
    multiclass_result_stat = None
    tp_center_error_stat = None
    classwise_ate_ase_stat = None
    visibility_subset_stat = None
    case_type_result_stat = None
    if not opt.video_only:
        multiclass_result_stat = eval_utils_isaac.init_multiclass_result_stat(
            class_names, IOU_THRESHOLDS
        )
        tp_center_error_stat = init_tp_center_error_stat(class_names)
        classwise_ate_ase_stat = init_classwise_ate_ase_stat(class_names)
        visibility_subset_stat = init_visibility_subset_stat(class_names)
        case_type_result_stat = init_case_type_result_stat(class_names)

    infer_info = build_infer_info(opt, hypes)
    if opt.global_sort_detections:
        infer_info += '_global_sort'
    # Keep targeted/limited debug runs from overwriting the full-evaluation
    # summary and visualization directory that use the same checkpoint.
    if sample_indices is not None:
        shown_indices = "_".join(str(idx) for idx in sample_indices[:8])
        if len(sample_indices) > 8:
            shown_indices += f"_plus{len(sample_indices) - 8}"
        infer_info += f"_selected_{shown_indices}"
    elif opt.max_samples is not None:
        infer_info += f"_first{int(opt.max_samples)}"
    evaluated_samples = 0
    communication_records = []
    stream_video_writer = (
        FFmpegStreamVideoWriter(opt.stream_video_output, opt.video_fps)
        if opt.stream_video_output
        else None
    )
    print(f"Isaac BEV/debug range: {hypes['postprocess']['gt_range']}")
    print(
        "BEV visualization: "
        f"interval={opt.save_vis_interval}"
    )
    print(
        "Feature visualization: follows --save_vis_interval, "
        "method=channel_l2, cmap=viridis, includes ego/co/raw/overlay and "
        "proposed-module internals under module_features/"
    )
    if opt.save_full_vis_output:
        print("Isaac full visualization export: BEV/RGB/LiDAR views will be saved for every frame")
    ego_dropout_args = (
        hypes.get("model", {}).get("args", {}).get("ego_feature_dropout", {})
        or {}
    )
    if ego_dropout_args.get("enabled", False):
        dropout_prob = float(ego_dropout_args.get("prob", 0.0))
        print(
            "Ego feature dropout: configured for training "
            f"p={dropout_prob:.3f}; "
            "disabled during inference/eval"
        )
    timing = defaultdict(float)

    def sync_for_timing():
        if opt.profile_timing and torch.cuda.is_available():
            torch.cuda.synchronize()

    video_ego_data_iterator = (
        iter(video_ego_data_loader)
        if video_ego_data_loader is not None
        else None
    )
    iter_end_time = time.perf_counter()
    for loader_idx, batch_data in enumerate(data_loader):
        i = (
            iteration_indices[loader_idx]
            if iteration_indices is not None
            else loader_idx
        )
        iter_start_time = time.perf_counter()
        timing["data_loading"] += iter_start_time - iter_end_time
        if (
            iteration_indices is None
            and opt.max_samples is not None
            and loader_idx >= opt.max_samples
        ):
            break
        video_ego_batch_data = None
        if video_ego_data_iterator is not None:
            try:
                video_ego_batch_data = next(video_ego_data_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "Video comparison ego-only loader ended before the "
                    "intermediate loader"
                ) from exc
            if (batch_data is None) != (video_ego_batch_data is None):
                raise RuntimeError(
                    "Video comparison loaders returned mismatched samples at "
                    f"dataset index {i}"
                )
        if loader_idx == 0 or i % 50 == 0:
            print(f"{infer_info}_{i}")
        if batch_data is None:
            iter_end_time = time.perf_counter()
            continue
        vis_sample_info = _vis_sample_info(opencood_dataset, i)
        visibility_membership = None
        if visibility_subset_stat is not None:
            visibility_membership, _ = (
                collect_isaac_visibility_membership(opencood_dataset, i)
            )
        vis_case_idx = vis_sample_info["sample_idx"]
        save_feature_vis = (
            opt.save_vis_interval > 0
            and (sample_indices is not None or vis_case_idx % opt.save_vis_interval == 0)
        )
        with torch.no_grad():
            stage_time = time.perf_counter()
            batch_data = train_utils.to_device(batch_data, device)
            if video_ego_batch_data is not None:
                video_ego_batch_data = train_utils.to_device(
                    video_ego_batch_data, device
                )
                video_record_len = video_ego_batch_data.get(
                    "ego", {}
                ).get("record_len")
                if video_record_len is None:
                    raise RuntimeError(
                        "Video comparison ego-only batch is missing record_len"
                    )
                if not torch.all(video_record_len == 1):
                    raise RuntimeError(
                        "Video comparison ego-only validation failed: "
                        f"expected record_len=1, got "
                        f"{video_record_len.detach().cpu().tolist()}"
                    )
                if loader_idx == 0:
                    print(
                        "Video comparison ego-only validation passed: "
                        "record_len=[1]"
                    )
            if opt.ego_only:
                record_len = batch_data.get('ego', {}).get('record_len')
                if record_len is None:
                    raise RuntimeError(
                        '--ego_only requires batch_data["ego"]["record_len"]'
                    )
                if not torch.all(record_len == 1):
                    raise RuntimeError(
                        '--ego_only validation failed: expected record_len=1, '
                        f'got {record_len.detach().cpu().tolist()}'
                    )
                if loader_idx == 0:
                    print('Ego-only validation passed: record_len=[1]')
            sync_for_timing()
            timing["to_device"] += time.perf_counter() - stage_time

            stage_time = time.perf_counter()
            setattr(model, "save_feature_debug", save_feature_vis)
            infer_result = inference_isaac_center_head(
                batch_data, model, hypes["postprocess"], opt.fusion_method
            )

            video_ego_infer_result = None
            if video_ego_batch_data is not None:
                setattr(model, "save_feature_debug", False)
                video_ego_infer_result = inference_isaac_center_head(
                    video_ego_batch_data,
                    model,
                    video_ego_hypes["postprocess"],
                    "intermediate",
                )
            setattr(model, "save_feature_debug", False)
            sync_for_timing()
            timing["model_postprocess"] += time.perf_counter() - stage_time

            stage_time = time.perf_counter()
            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            pred_label = infer_result.get("pred_label")
            gt_label = infer_result.get("gt_label")
            if (
                not opt.video_only
                and isinstance(infer_result.get("communication"), dict)
            ):
                communication_records.append(infer_result["communication"])
            class_gt_box_tensor = gt_box_tensor
            evaluated_samples += 1

            if not opt.video_only:
                for iou_thresh in IOU_THRESHOLDS:
                    eval_utils.caluclate_tp_fp(
                        pred_box_tensor,
                        pred_score,
                        gt_box_tensor,
                        result_stat,
                        iou_thresh,
                    )
            if not opt.video_only and gt_label is None:
                class_gt_box_tensor, gt_label = (
                    eval_utils_isaac.generate_gt_bbx_with_classes_isaac(
                        batch_data, hypes["postprocess"]
                    )
                )

            if multiclass_result_stat is not None and gt_label is not None:
                for iou_thresh in IOU_THRESHOLDS:
                    eval_utils_isaac.calculate_tp_fp_multiclass_isaac(
                        pred_box_tensor,
                        pred_score,
                        pred_label,
                        class_gt_box_tensor,
                        gt_label,
                        multiclass_result_stat,
                        iou_thresh,
                        class_names,
                    )
                update_tp_center_error_stat(
                    tp_center_error_stat,
                    pred_box_tensor,
                    pred_score,
                    pred_label,
                    class_gt_box_tensor,
                    gt_label,
                    class_names,
                    iou_thresh=0.3,
                )
                update_classwise_ate_ase_stat(
                    classwise_ate_ase_stat,
                    pred_box_tensor,
                    pred_score,
                    pred_label,
                    class_gt_box_tensor,
                    gt_label,
                    class_names,
                )
                visibility_gt_boxes, visibility_gt_labels, _, subset_labels = (
                    generate_visibility_labels_for_evaluated_gt(
                        batch_data,
                        hypes["postprocess"],
                        visibility_membership,
                    )
                )
                _verify_visibility_gt_alignment(
                    class_gt_box_tensor,
                    gt_label,
                    visibility_gt_boxes,
                    visibility_gt_labels,
                )
                update_visibility_subset_stat(
                    visibility_subset_stat,
                    pred_box_tensor,
                    pred_score,
                    pred_label,
                    visibility_gt_boxes,
                    visibility_gt_labels,
                    subset_labels,
                    class_names,
                )
                if case_type_result_stat is not None:
                    case_type, _ = classify_isaac_case_type(
                        opencood_dataset, vis_sample_info["sample_idx"]
                    )
                    if case_type in case_type_result_stat:
                        case_type_result_stat[case_type]["num_samples"] += 1
                        for iou_thresh in IOU_THRESHOLDS:
                            eval_utils_isaac.calculate_tp_fp_multiclass_isaac(
                                pred_box_tensor,
                                pred_score,
                                pred_label,
                                class_gt_box_tensor,
                                gt_label,
                                case_type_result_stat[case_type]["result_stat"],
                                iou_thresh,
                                class_names,
                            )
            timing["bev_eval"] += time.perf_counter() - stage_time

            stage_time = time.perf_counter()
            isaac_base_data_for_vis = None
            infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            vis_sample_info = _vis_sample_info(opencood_dataset, i)
            vis_case_idx = vis_sample_info["sample_idx"]
            save_bev_vis = (
                (
                    opt.save_full_vis_output
                    or (
                        opt.save_vis_interval > 0
                        and (sample_indices is not None or vis_case_idx % opt.save_vis_interval == 0)
                    )
                )
                and (pred_box_tensor is not None or gt_box_tensor is not None)
            )
            if stream_video_writer is not None:
                if isaac_base_data_for_vis is None:
                    isaac_base_data_for_vis = opencood_dataset.retrieve_base_data(i)
                stream_vis_lidar = build_isaac_multicav_lidar_for_vis(
                    opencood_dataset,
                    i,
                    ego_lidar=None,
                    base_data_dict=isaac_base_data_for_vis,
                )
                stream_frame = build_isaac_stream_video_frame(
                    opencood_dataset,
                    i,
                    isaac_base_data_for_vis,
                    infer_result,
                    stream_vis_lidar,
                    hypes["postprocess"]["gt_range"],
                    left_hand,
                    opt.fusion_method,
                    ego_only_infer_result=video_ego_infer_result,
                )
                stream_video_writer.write(stream_frame)
            if save_bev_vis:
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                vis_save_path_root = _isaac_case_output_root(
                    vis_save_path_root,
                    opencood_dataset,
                    vis_sample_info["sample_idx"],
                    opt.split_output_by_case,
                )
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)
                sample_vis_dir, bev_vis_index, vis_sample_info = (
                    _isaac_vis_dir_and_index(
                        vis_save_path_root, opencood_dataset, i
                    )
                )
                os.makedirs(sample_vis_dir, exist_ok=True)
                if isaac_base_data_for_vis is None:
                    try:
                        isaac_base_data_for_vis = opencood_dataset.retrieve_base_data(i)
                    except Exception as exc:
                        isaac_base_data_for_vis = None
                        print(f"Skip Isaac per-agent RGB/LiDAR visualization at sample {i}: {exc}")
                 
                vis_save_path = os.path.join(
                    sample_vis_dir,
                    _isaac_bev_vis_filename(bev_vis_index, vis_sample_info),
                )
                vis_lidar = build_isaac_multicav_lidar_for_vis(
                                    opencood_dataset,
                                    i,
                                    ego_lidar=None,
                                    base_data_dict=isaac_base_data_for_vis)
                simple_vis_isaac.visualize(infer_result,
                                    vis_lidar,
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                if "IsaacSim" in hypes.get("test_dir", ""):
                    save_isaac_agent_rgb_lidar_views(
                        opencood_dataset,
                        i,
                        sample_vis_dir,
                        bev_vis_index,
                        base_data_dict=isaac_base_data_for_vis,
                        only_cav_id=vis_sample_info.get("ego_cav_id"),
                        infer_result=infer_result,
                    )
            if save_feature_vis and infer_result.get("debug_features"):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                vis_save_path_root = _isaac_case_output_root(
                    vis_save_path_root,
                    opencood_dataset,
                    vis_sample_info["sample_idx"],
                    opt.split_output_by_case,
                )
                os.makedirs(vis_save_path_root, exist_ok=True)
                sample_vis_dir, bev_vis_index, vis_sample_info = (
                    _isaac_vis_dir_and_index(
                        vis_save_path_root, opencood_dataset, i
                    )
                )
                os.makedirs(sample_vis_dir, exist_ok=True)
                save_isaac_feature_debug_views(
                    infer_result["debug_features"],
                    sample_vis_dir,
                    bev_vis_index,
                    opt,
                )
                save_isaac_method_debug_views(
                    infer_result["debug_features"],
                    sample_vis_dir,
                    bev_vis_index,
                    opt,
                )
                save_isaac_module_feature_views(
                    infer_result["debug_features"],
                    sample_vis_dir,
                    bev_vis_index,
                    opt,
                )
            timing["visualization"] += time.perf_counter() - stage_time
            timing["loop_total"] += time.perf_counter() - iter_start_time
            iter_end_time = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if stream_video_writer is not None:
        stream_video_writer.close()

    if opt.video_only:
        print(
            "Video-only inference complete: evaluation summaries were "
            "intentionally skipped."
        )
    elif opt.max_samples is not None and all(v['gt'] == 0 for v in result_stat.values()):
        print("Skip AP calculation: no GT boxes were seen in this limited Isaac debug run.")
    else:
        communication_summary = summarize_communication_stats(
            communication_records
        )
        save_isaac_eval_summary(
            result_stat,
            multiclass_result_stat,
            class_names,
            opt.model_dir,
            infer_info,
            evaluated_samples,
            global_sort_detections=opt.global_sort_detections,
            tp_center_error_stat=tp_center_error_stat,
            classwise_ate_ase_stat=classwise_ate_ase_stat,
            case_type_result_stat=case_type_result_stat,
            opt=opt,
            hypes=hypes,
            resume_epoch=resume_epoch,
            communication_summary=communication_summary,
            visibility_subset_stat=visibility_subset_stat,
        )
        if communication_summary is not None:
            print(
                "Communication: "
                f"{communication_summary['mean_per_sample_KB']:.3f} KB/sample "
                f"({communication_summary['mean_per_sample_MB']:.6f} MB/sample), "
                f"total {communication_summary['total_MB']:.3f} MB"
            )


    if opt.profile_timing:
        sample_count = max(evaluated_samples, 1)
        print("Isaac inference timing profile:")
        for key in (
            "data_loading",
            "to_device",
            "model_postprocess",
            "bev_eval",
            "visualization",
            "loop_total",
        ):
            seconds = timing.get(key, 0.0)
            print(f"  {key}: {seconds:.2f}s total, {seconds / sample_count:.4f}s/sample")

if __name__ == '__main__':
    main()
