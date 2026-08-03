# -*- coding: utf-8 -*-
"""Utility functions for anchor-free CenterHead box decoding."""

import torch


def _decode_args_from_output(output_dict, fallback_params=None):
    decode_args = output_dict.get("center_head_decode_args", {})
    fallback_params = {} if fallback_params is None else fallback_params
    anchor_args = fallback_params.get("anchor_args", {})

    lidar_range = decode_args.get(
        "lidar_range",
        anchor_args.get("cav_lidar_range", fallback_params.get("gt_range")),
    )
    if lidar_range is None:
        raise KeyError("CenterHead decoding needs lidar_range or cav_lidar_range.")

    return (
        lidar_range,
        decode_args.get("min_size", 0.05),
        decode_args.get("max_size", 8.0),
    )


def decode_center_boxes_at_indices(
    pred_dict,
    cell_inds,
    lidar_range,
    min_size=0.05,
    max_size=8.0,
    batch_idx=0,
):
    """Decode selected CenterHead cells to boxes in hwl order.

    Args:
        pred_dict: Raw CenterHead output dict with hm/center/center_z/dim/rot.
        cell_inds: Flat H*W cell indices. Class-expanded heatmap indices should
            be converted to cell indices before calling this function.
        lidar_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        min_size: Lower clamp for decoded box dimensions.
        max_size: Upper clamp for decoded box dimensions.
        batch_idx: Batch element to decode.

    Returns:
        Tensor with shape [N, 7] in [x, y, z, h, w, l, yaw] order.
    """
    hm = pred_dict["hm"]
    _, _, height, width = hm.shape
    device = hm.device
    dtype = hm.dtype

    if cell_inds.numel() == 0:
        return torch.empty((0, 7), device=device, dtype=dtype)

    cell_inds = cell_inds.to(device=device, dtype=torch.long)
    ys = torch.div(cell_inds, width, rounding_mode="floor")
    xs = cell_inds - ys * width

    x_min, y_min, _, x_max, y_max, _ = lidar_range
    stride_x = (x_max - x_min) / float(width)
    stride_y = (y_max - y_min) / float(height)

    flat_center_x = pred_dict["center"][batch_idx, 0].reshape(-1)
    flat_center_y = pred_dict["center"][batch_idx, 1].reshape(-1)
    flat_z = pred_dict["center_z"][batch_idx, 0].reshape(-1)
    flat_h = pred_dict["dim"][batch_idx, 0].reshape(-1)
    flat_w = pred_dict["dim"][batch_idx, 1].reshape(-1)
    flat_l = pred_dict["dim"][batch_idx, 2].reshape(-1)
    flat_rot_cos = pred_dict["rot"][batch_idx, 0].reshape(-1)
    flat_rot_sin = pred_dict["rot"][batch_idx, 1].reshape(-1)

    center_x = (
        xs.to(dtype=dtype) + flat_center_x.index_select(0, cell_inds)
    ) * stride_x + x_min
    center_y = (
        ys.to(dtype=dtype) + flat_center_y.index_select(0, cell_inds)
    ) * stride_y + y_min
    center_z = flat_z.index_select(0, cell_inds)

    dim_h = torch.exp(
        torch.clamp(flat_h.index_select(0, cell_inds), min=-4.0, max=2.0)
    )
    dim_w = torch.exp(
        torch.clamp(flat_w.index_select(0, cell_inds), min=-4.0, max=2.0)
    )
    dim_l = torch.exp(
        torch.clamp(flat_l.index_select(0, cell_inds), min=-4.0, max=2.0)
    )
    dim_h = torch.clamp(dim_h, min=min_size, max=max_size)
    dim_w = torch.clamp(dim_w, min=min_size, max=max_size)
    dim_l = torch.clamp(dim_l, min=min_size, max=max_size)

    yaw = torch.atan2(
        flat_rot_sin.index_select(0, cell_inds),
        flat_rot_cos.index_select(0, cell_inds),
    )

    return torch.stack(
        [center_x, center_y, center_z, dim_h, dim_w, dim_l, yaw],
        dim=-1,
    )


def decode_center_boxes_for_output(output_dict, cell_inds, fallback_params=None):
    lidar_range, min_size, max_size = _decode_args_from_output(
        output_dict, fallback_params
    )
    return decode_center_boxes_at_indices(
        output_dict["center_head_preds"],
        cell_inds,
        lidar_range,
        min_size=min_size,
        max_size=max_size,
    )
