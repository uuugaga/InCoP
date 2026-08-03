# -*- coding: utf-8 -*-
"""CoBEVT dense BEV communication and fused axial (swap) attention."""

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

from opencood.models.fuse_modules.communication import (
    CommunicationStatsMixin, SpatialDownsampleAdapter,
)
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.swap_fusion_modules import SwapFusionBlockMask
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class CoBEVT(CommunicationStatsMixin, nn.Module):
    """Detection-side CoBEVT Swap Fusion.

    CoBEVT uses sparse *attention axes*, but its paper does not specify sparse
    network messages.  Consequently this baseline correctly counts and sends a
    dense aligned BEV feature map.
    """

    def __init__(self, args):
        super(CoBEVT, self).__init__()
        args = args or {}
        input_dim = int(args.get("input_dim", 64))
        model_dim = int(args.get("model_dim", input_dim))
        self.window_size = int(args.get("window_size", 4))
        depth = int(args.get("depth", 3))
        mlp_dim = int(args.get("mlp_dim", model_dim * 4))
        agent_size = int(args.get("agent_size", args.get("max_cav", 2)))
        drop_out = float(args.get("drop_out", 0.0))
        dim_head = int(args.get("dim_head", 32))
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.input_proj = (
            nn.Identity() if input_dim == model_dim
            else nn.Conv2d(input_dim, model_dim, kernel_size=1, bias=False)
        )
        self.spatial_downsample_stages = int(
            args.get("spatial_downsample_stages", 1)
        )
        self.spatial_adapter = SpatialDownsampleAdapter(
            input_dim, self.spatial_downsample_stages
        )
        self.layers = nn.ModuleList([
            SwapFusionBlockMask(
                model_dim, mlp_dim, dim_head, self.window_size, agent_size, drop_out
            ) for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            Reduce("b m d h w -> b d h w", "mean"),
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            Rearrange("b h w d -> b d h w"),
        )
        self.output_proj = (
            nn.Identity() if model_dim == input_dim
            else nn.Conv2d(model_dim, input_dim, kernel_size=1, bias=False)
        )
        self._init_communication_stats(args, method="CoBEVT")

    @staticmethod
    def _length(value):
        return int(value.detach().cpu().item()) if torch.is_tensor(value) else int(value)

    def forward(self, x, record_len, affine_matrix, **kwargs):
        del kwargs
        output_size = x.shape[-2:]
        x = self.spatial_adapter.encode(x)
        _, _, raw_h, raw_w = x.shape
        batch_size, max_cav = affine_matrix.shape[:2]
        regroup_feature, mask = Regroup(x, record_len, max_cav)
        aligned = []
        for batch_idx in range(batch_size):
            aligned.append(warp_affine_simple(
                regroup_feature[batch_idx], affine_matrix[batch_idx, 0], (raw_h, raw_w)
            ))
        x = torch.stack(aligned)
        x = self.input_proj(x.flatten(0, 1)).reshape(
            batch_size, max_cav, self.model_dim, raw_h, raw_w
        )
        pad_h = (self.window_size - raw_h % self.window_size) % self.window_size
        pad_w = (self.window_size - raw_w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height, width = x.shape[-2:]
        com_mask = repeat(
            mask.unsqueeze(1).unsqueeze(2).unsqueeze(3),
            "b h w c l -> b (h new_h) (w new_w) c l",
            new_h=height, new_w=width,
        )
        for layer in self.layers:
            x = layer(x, mask=com_mask)
        fused = self.mlp_head(x)[..., :raw_h, :raw_w]
        fused = self.output_proj(fused)
        fused = self.spatial_adapter.decode(fused, output_size)

        feature_elements = dense_elements = message_count = 0
        per_sample = []
        for length in record_len:
            collaborators = max(self._length(length) - 1, 0)
            sample_elements = collaborators * self.input_dim * raw_h * raw_w
            feature_elements += sample_elements
            dense_elements += sample_elements
            message_count += collaborators
            per_sample.append(sample_elements)
        self._record_communication(
            feature_elements=feature_elements,
            dense_feature_elements=dense_elements,
            message_count=message_count,
            batch_size=batch_size,
            mode="dense_bev",
            per_sample_feature_elements=per_sample,
            notes="CoBEVT swap attention is computationally sparse/axial; network transmission remains a dense BEV map.",
        )
        return fused
