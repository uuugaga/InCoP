# -*- coding: utf-8 -*-
"""V2X-ViT fusion with progressive channel compression and metadata."""

import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.fuse_modules.communication import (
    CommunicationStatsMixin, SpatialDownsampleAdapter,
)
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class V2XViTFusion(CommunicationStatsMixin, nn.Module):
    """V2X-ViT's compressed transmission, STTF, and heterogeneous transformer."""

    requires_fusion_context = True

    def __init__(self, args):
        super(V2XViTFusion, self).__init__()
        from opencood.models.sub_modules.v2xvit_basic import V2XTransformer

        args = args or {}
        input_dim = int(args.get("input_dim", args.get("feat_dim", 64)))
        model_dim = int(args.get("model_dim", 256))
        compression_rate = int(args.get("compression_rate", 32))
        compressed_dim = int(args.get("compressed_dim", max(1, input_dim // compression_rate)))
        if min(input_dim, model_dim, compressed_dim) <= 0:
            raise ValueError("V2X-ViT channel dimensions must be positive.")
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.compressed_dim = compressed_dim
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
        # Progressive 1x1 channel encoder/decoder described in V2X-ViT.
        middle_dim = max(compressed_dim, input_dim // 4)
        self.compressor = nn.Sequential(
            nn.Conv2d(input_dim, middle_dim, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(middle_dim, compressed_dim, kernel_size=1),
        )
        self.decompressor = nn.Sequential(
            nn.Conv2d(compressed_dim, middle_dim, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(middle_dim, input_dim, kernel_size=1),
        )
        self.fusion_net = V2XTransformer(args["transformer"])
        window_sizes = args["transformer"]["encoder"][
            "pwindow_att_config"
        ].get("window_size", [1])
        self.window_multiple = max(int(value) for value in window_sizes)
        self.output_proj = (
            nn.Identity() if model_dim == input_dim
            else nn.Conv2d(model_dim, input_dim, kernel_size=1, bias=False)
        )
        self._init_communication_stats(args, method="V2X-ViT")

    @staticmethod
    def _length(value):
        return int(value.detach().cpu().item()) if torch.is_tensor(value) else int(value)

    def _prior_encoding(self, context, batch_size, max_cav, device, dtype):
        prior = None
        if isinstance(context, dict):
            prior = context.get("v2x_prior_encoding", context.get("prior_encoding"))
        if prior is None:
            return torch.zeros(batch_size, max_cav, 3, device=device, dtype=dtype)
        prior = prior.to(device=device, dtype=dtype)
        if prior.ndim == 2:
            output = torch.zeros(batch_size, max_cav, 3, device=device, dtype=dtype)
            offset = 0
            record_len = context.get("record_len") if isinstance(context, dict) else None
            if record_len is None:
                raise ValueError("Flattened V2X prior encoding requires record_len in fusion_context.")
            for batch_idx, length in enumerate(record_len):
                count = self._length(length)
                output[batch_idx, :count] = prior[offset:offset + count, :3]
                offset += count
            return output
        return prior[:, :max_cav, :3]

    def forward(self, x, record_len, affine_matrix, fusion_context=None, **kwargs):
        del kwargs
        output_size = x.shape[-2:]
        message = self.spatial_adapter.encode(x)
        compressed = self.compressor(message)
        decoded = self.decompressor(compressed)
        _, _, height, width = decoded.shape
        projected = self.input_proj(decoded)
        batch_size, max_cav = affine_matrix.shape[:2]
        regroup_feature, mask = Regroup(projected, record_len, max_cav)
        prior = self._prior_encoding(
            fusion_context, batch_size, max_cav, decoded.device, decoded.dtype
        )
        prior_map = prior[..., None, None].expand(-1, -1, -1, height, width)

        pairwise = None
        if isinstance(fusion_context, dict):
            pairwise = fusion_context.get("pairwise_t_matrix")
        if torch.is_tensor(pairwise) and pairwise.ndim == 5:
            spatial_correction = pairwise[:, 0, :max_cav].to(
                device=decoded.device, dtype=decoded.dtype
            )
        else:
            # The shared OpenCOOD affine matrix already maps agents to ego.
            aligned = []
            for batch_idx in range(batch_size):
                aligned.append(warp_affine_simple(
                    regroup_feature[batch_idx], affine_matrix[batch_idx, 0], (height, width)
                ))
            regroup_feature = torch.stack(aligned)
            spatial_correction = torch.eye(
                4, device=decoded.device, dtype=decoded.dtype
            ).view(1, 1, 4, 4).expand(batch_size, max_cav, -1, -1)

        transformer_input = torch.cat([regroup_feature, prior_map], dim=2)
        pad_h = (self.window_multiple - height % self.window_multiple) % self.window_multiple
        pad_w = (self.window_multiple - width % self.window_multiple) % self.window_multiple
        if pad_h or pad_w:
            transformer_input = F.pad(transformer_input, (0, pad_w, 0, pad_h))
        transformer_input = transformer_input.permute(0, 1, 3, 4, 2)
        fused = self.fusion_net(transformer_input, mask, spatial_correction)
        fused = fused[:, :height, :width].permute(0, 3, 1, 2)
        fused = self.output_proj(fused)
        fused = self.spatial_adapter.decode(fused, output_size)

        feature_elements = dense_elements = metadata_elements = message_count = 0
        per_sample = []
        for length in record_len:
            collaborators = max(self._length(length) - 1, 0)
            sample_elements = collaborators * self.compressed_dim * height * width
            feature_elements += sample_elements
            dense_elements += collaborators * self.input_dim * height * width
            # velocity, delay, agent type, and a 4x4 extrinsic matrix.
            metadata_elements += collaborators * 19
            message_count += collaborators
            per_sample.append(sample_elements)
        self._record_communication(
            feature_elements=feature_elements,
            metadata_elements=metadata_elements,
            dense_feature_elements=dense_elements,
            message_count=message_count,
            batch_size=batch_size,
            mode="progressive_channel_compressed_dense_bev_with_metadata",
            per_sample_feature_elements=per_sample,
            notes="Payload after the progressive 1x1 encoder; metadata counts 3 priors and 16 extrinsic values per collaborator.",
        )
        return fused
