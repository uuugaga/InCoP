# -*- coding: utf-8 -*-
"""DSVT-style LiDAR encoders for OpenCOOD heterogeneous models.

The main encoder is a DSVT-Pillar variant adapted from the official DSVT
implementation. It keeps the paper's dynamic sparse set partition, rotated
sets, shifted/hybrid windows, q/k position embedding, and DSVTBlock residual
layout, while avoiding OpenPCDet-specific CUDA ops so it can run in HEAL.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class DSVTPositionEmbedding(nn.Module):
    """Learned relative window position embedding used by DSVT."""

    def __init__(self, input_channels: int, channels: int):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.position_embedding_head(xy)


class DSVTSetAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, feedforward_channels: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.channels = channels
        self.norm1 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, feedforward_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_channels, channels)
        self.activation = nn.GELU()
        self.norm2 = nn.LayerNorm(channels)

    def _scatter_set_features(
        self,
        src: torch.Tensor,
        set_update: torch.Tensor,
        set_inds: torch.Tensor,
        set_masks: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(src)
        output_count = torch.zeros((src.shape[0], 1), dtype=src.dtype, device=src.device)
        flat_inds = set_inds.reshape(-1)
        valid = ~set_masks.reshape(-1)
        valid_inds = flat_inds[valid]
        valid_update = set_update.reshape(-1, self.channels)[valid].to(output.dtype)
        output.index_add_(0, valid_inds, valid_update)
        output_count.index_add_(
            0,
            valid_inds,
            torch.ones((valid_update.shape[0], 1), dtype=src.dtype, device=src.device),
        )
        return torch.where(output_count > 0, output / output_count.clamp_min(1.0), torch.zeros_like(src))

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        set_inds: torch.Tensor,
        set_masks: torch.Tensor,
        chunk_sets: int,
    ) -> torch.Tensor:
        if set_inds.numel() == 0:
            return src

        if chunk_sets <= 0 or set_inds.shape[0] <= chunk_sets:
            set_features = src[set_inds]
            set_pos = pos[set_inds]
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features
            set_update = self.attn(
                query,
                key,
                value,
                key_padding_mask=set_masks,
                need_weights=False,
            )[0]
            attn_out = self._scatter_set_features(src, set_update, set_inds, set_masks)
            src = src + attn_out
            src = self.norm1(src)
            ffn_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + ffn_out)
            return src

        attn_out = torch.zeros_like(src)
        attn_count = torch.zeros((src.shape[0], 1), dtype=src.dtype, device=src.device)

        for start in range(0, set_inds.shape[0], chunk_sets):
            inds = set_inds[start:start + chunk_sets]
            masks = set_masks[start:start + chunk_sets]
            set_features = src[inds]
            set_pos = pos[inds]
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features
            set_update = self.attn(
                query,
                key,
                value,
                key_padding_mask=masks,
                need_weights=False,
            )[0]

            flat_inds = inds.reshape(-1)
            valid = ~masks.reshape(-1)
            valid_inds = flat_inds[valid]
            valid_update = set_update.reshape(-1, self.channels)[valid].to(attn_out.dtype)
            attn_out.index_add_(0, valid_inds, valid_update)
            attn_count.index_add_(
                0,
                valid_inds,
                torch.ones((valid_update.shape[0], 1), dtype=src.dtype, device=src.device),
            )

        attn_out = torch.where(attn_count > 0, attn_out / attn_count.clamp_min(1.0), torch.zeros_like(src))
        src = src + attn_out
        src = self.norm1(src)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + ffn_out)
        return src


class DSVTEncoderLayer(nn.Module):
    def __init__(self, channels: int, num_heads: int, feedforward_channels: int, dropout: float):
        super().__init__()
        self.win_attn = DSVTSetAttention(channels, num_heads, feedforward_channels, dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        set_inds: torch.Tensor,
        set_masks: torch.Tensor,
        chunk_sets: int,
    ) -> torch.Tensor:
        identity = src
        src = self.win_attn(src, pos, set_inds, set_masks, chunk_sets)
        return self.norm(src + identity)


class DSVTBlock(nn.Module):
    """A DSVT block with two rotated-set encoder layers."""

    def __init__(self, channels: int, num_heads: int, feedforward_channels: int, dropout: float):
        super().__init__()
        self.encoder_list = nn.ModuleList(
            [
                DSVTEncoderLayer(channels, num_heads, feedforward_channels, dropout),
                DSVTEncoderLayer(channels, num_heads, feedforward_channels, dropout),
            ]
        )

    def forward(
        self,
        src: torch.Tensor,
        pos_list: Tuple[torch.Tensor, torch.Tensor],
        set_inds_list: Tuple[torch.Tensor, torch.Tensor],
        set_masks_list: Tuple[torch.Tensor, torch.Tensor],
        chunk_sets: int,
    ) -> torch.Tensor:
        output = src
        for axis_id, layer in enumerate(self.encoder_list):
            output = layer(
                output,
                pos_list[axis_id],
                set_inds_list[axis_id],
                set_masks_list[axis_id],
                chunk_sets,
            )
        return output


class DsvtPillar(nn.Module):
    """DSVT-style sparse pillar encoder returning dense BEV features.

    Input follows OpenCOOD's heter encoder convention:
    ``data_dict[f"inputs_{modality_name}"]`` contains voxel tensors.
    Output is ``B x C x H x W`` and can feed the existing BEV backbone.
    """

    def __init__(self, args):
        super().__init__()
        self.lidar_range = np.asarray(args["lidar_range"], dtype=np.float32)
        self.voxel_size = np.asarray(args["voxel_size"], dtype=np.float32)
        grid_size = (self.lidar_range[3:6] - self.lidar_range[0:3]) / self.voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        pillar_cfg = dict(args["pillar_vfe"])
        self.pillar_vfe = PillarVFE(
            pillar_cfg,
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        in_channels = pillar_cfg["num_filters"][-1]
        self.channels = int(args.get("d_model", in_channels))
        self.input_proj = nn.Linear(in_channels, self.channels) if in_channels != self.channels else nn.Identity()

        if "window_shapes" in args:
            self.window_shapes = [tuple(int(v) for v in shape) for shape in args["window_shapes"]]
        else:
            base_window = tuple(int(v) for v in args.get("window_shape", [12, 12]))
            self.window_shapes = [
                base_window,
                tuple(int(v * 2) for v in base_window),
            ]
        self.set_size = int(args.get("set_size", 36))
        self.num_blocks = int(args.get("num_blocks", 4))
        self.chunk_sets = int(args.get("chunk_sets", 128))
        if "shifts" in args:
            self.shift_list = [tuple(int(v) for v in shift) for shift in args["shifts"]]
        else:
            self.shift_list = [
                (0, 0),
                tuple(int(v) for v in args.get("shift", [self.window_shapes[0][0] // 2, self.window_shapes[0][1] // 2])),
            ]
        assert len(self.window_shapes) == len(self.shift_list)

        num_heads = int(args.get("num_heads", 4))
        feedforward_channels = int(args.get("feedforward_channels", self.channels * 2))
        dropout = float(args.get("dropout", 0.0))
        self.pos_embed_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DSVTPositionEmbedding(2, self.channels),
                        DSVTPositionEmbedding(2, self.channels),
                    ]
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.blocks = nn.ModuleList(
            [DSVTBlock(self.channels, num_heads, feedforward_channels, dropout) for _ in range(self.num_blocks)]
        )
        self.residual_norms = nn.ModuleList(
            [nn.LayerNorm(self.channels) for _ in range(self.num_blocks)]
        )

        scatter_cfg = dict(args["point_pillar_scatter"])
        scatter_cfg["grid_size"] = grid_size
        scatter_cfg["num_features"] = self.channels
        self.scatter = PointPillarScatter(scatter_cfg)

    @torch.no_grad()
    def _window_info(
        self,
        coords: torch.Tensor,
        window_shape: Tuple[int, int],
        shift_xy: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        win_x, win_y = window_shape
        shift_x, shift_y = shift_xy
        batch = coords[:, 0].long()
        y = coords[:, 2].long()
        x = coords[:, 3].long()

        sx = x + shift_x
        sy = y + shift_y
        wx = torch.div(sx, win_x, rounding_mode="floor")
        wy = torch.div(sy, win_y, rounding_mode="floor")
        ix = torch.remainder(sx, win_x)
        iy = torch.remainder(sy, win_y)

        max_wx = int(math.ceil(float(self.scatter.nx) / float(win_x))) + 1
        max_wy = int(math.ceil(float(self.scatter.ny) / float(win_y))) + 1
        window_key = (batch * max_wx + wx) * max_wy + wy
        return window_key, ix, iy, batch

    @torch.no_grad()
    def _make_sets(
        self,
        coords: torch.Tensor,
        window_shape: Tuple[int, int],
        shift_xy: Tuple[int, int],
        axis_order: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        window_key, ix, iy, _ = self._window_info(coords, window_shape, shift_xy)
        if window_key.numel() == 0:
            empty_inds = coords.new_zeros((0, self.set_size))
            return empty_inds, torch.ones((0, self.set_size), dtype=torch.bool, device=coords.device)

        if axis_order == 0:
            local_sort_key = iy * window_shape[0] + ix
        else:
            local_sort_key = ix * window_shape[1] + iy

        sort_scale = int(window_shape[0] * window_shape[1]) + 1
        sorted_order = torch.argsort(window_key * sort_scale + local_sort_key)
        sorted_window_key = window_key[sorted_order]
        _, counts = torch.unique_consecutive(sorted_window_key, return_counts=True)
        set_counts = torch.div(counts + self.set_size - 1, self.set_size, rounding_mode="floor")
        total_sets = int(set_counts.sum().item())
        if total_sets == 0:
            empty_inds = coords.new_zeros((0, self.set_size))
            return empty_inds, torch.ones((0, self.set_size), dtype=torch.bool, device=coords.device)

        starts = torch.cumsum(counts, dim=0) - counts
        set_offsets = torch.cumsum(set_counts, dim=0) - set_counts
        window_ids = torch.repeat_interleave(torch.arange(counts.shape[0], device=coords.device), set_counts)
        set_ids_in_window = (
            torch.arange(total_sets, device=coords.device)
            - torch.repeat_interleave(set_offsets, set_counts)
        )

        arange_set = torch.arange(self.set_size, device=coords.device)
        denominator = (set_counts[window_ids] * self.set_size).unsqueeze(1)
        dense_select = torch.div(
            (set_ids_in_window.unsqueeze(1) * self.set_size + arange_set.unsqueeze(0))
            * counts[window_ids].unsqueeze(1),
            denominator,
            rounding_mode="floor",
        )
        dense_select = torch.minimum(dense_select, (counts[window_ids] - 1).unsqueeze(1))
        set_inds = sorted_order[starts[window_ids].unsqueeze(1) + dense_select]

        set_masks = dense_select == torch.roll(dense_select, shifts=1, dims=1)
        set_masks[:, 0] = False
        return set_inds, set_masks

    def _positional_features(
        self,
        coords: torch.Tensor,
        window_shape: Tuple[int, int],
        shift_xy: Tuple[int, int],
        pos_layer: DSVTPositionEmbedding,
    ) -> torch.Tensor:
        win_x, win_y = window_shape
        _, ix, iy, _ = self._window_info(coords, window_shape, shift_xy)
        x = ix.float() - float(win_x) / 2.0
        y = iy.float() - float(win_y) / 2.0
        pos = torch.stack([x, y], dim=-1)
        return pos_layer(pos)

    def forward(self, data_dict, modality_name):
        inputs = data_dict[f"inputs_{modality_name}"]
        voxel_features = inputs["voxel_features"]
        voxel_coords = inputs["voxel_coords"]
        voxel_num_points = inputs["voxel_num_points"]
        if voxel_coords.numel() == 0:
            batch_size = int(data_dict.get("batch_size", 1))
            return voxel_features.new_zeros((batch_size, self.channels, self.scatter.ny, self.scatter.nx))

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        features = self.input_proj(batch_dict["pillar_features"])
        coords = voxel_coords.long()
        set_cache = {}

        for block_id in range(self.num_blocks):
            window_id = block_id % len(self.window_shapes)
            window_shape = self.window_shapes[window_id]
            shift_xy = self.shift_list[window_id]
            cache_key = (window_shape, shift_xy)
            if cache_key not in set_cache:
                cached_set_inds = []
                cached_set_masks = []
                for axis_order in range(2):
                    set_inds, set_masks = self._make_sets(coords, window_shape, shift_xy, axis_order)
                    cached_set_inds.append(set_inds)
                    cached_set_masks.append(set_masks)
                set_cache[cache_key] = (cached_set_inds, cached_set_masks)
            set_inds_list, set_masks_list = set_cache[cache_key]
            pos_list = []
            for axis_order in range(2):
                pos_list.append(
                    self._positional_features(
                        coords,
                        window_shape,
                        shift_xy,
                        self.pos_embed_layers[block_id][axis_order],
                    )
                )

            residual = features
            features = self.blocks[block_id](
                features,
                (pos_list[0], pos_list[1]),
                (set_inds_list[0], set_inds_list[1]),
                (set_masks_list[0], set_masks_list[1]),
                self.chunk_sets,
            )
            features = self.residual_norms[block_id](features + residual)

        batch_dict["pillar_features"] = features
        batch_dict["voxel_coords"] = coords
        batch_dict = self.scatter(batch_dict)
        return batch_dict["spatial_features"]


# Backward-compatible aliases for older IsaacSim logs/configs.
IsaacDSVTPositionEmbedding = DSVTPositionEmbedding
IsaacDSVTSetAttention = DSVTSetAttention
IsaacDSVTEncoderLayer = DSVTEncoderLayer
IsaacDSVTBlock = DSVTBlock
IsaacDsvtPillar = DsvtPillar
