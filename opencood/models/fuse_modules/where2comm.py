# -*- coding: utf-8 -*-
"""Where2comm spatially sparse communication and confidence-aware fusion."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.fuse_modules.common import regroup
from opencood.models.fuse_modules.communication import (
    CommunicationStatsMixin, SpatialDownsampleAdapter,
)
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


def _gaussian_kernel(kernel_size, sigma, device, dtype):
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype)
    coordinates = coordinates - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, kernel_size, kernel_size)


class Where2commAttention(nn.Module):
    """Per-location multi-agent attention with confidence as an attention prior."""

    def __init__(self, dim, heads=8, dropout=0.0, max_agents=8):
        super(Where2commAttention, self).__init__()
        if dim % heads != 0:
            raise ValueError("Where2comm model_dim must be divisible by heads.")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.agent_embedding = nn.Embedding(max_agents, dim)
        self.sensor_embedding = nn.Embedding(3, dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, features, confidence, valid, sensor_types=None):
        # features [N,C,H,W], confidence/valid [N,1,H,W]
        num_agents, channels, height, width = features.shape
        tokens = features.permute(2, 3, 0, 1).reshape(height * width, num_agents, channels)
        agent_ids = torch.arange(num_agents, device=features.device)
        position = self.agent_embedding(agent_ids)
        if sensor_types is None:
            sensor_types = torch.zeros(num_agents, device=features.device, dtype=torch.long)
        sensor_types = sensor_types.to(device=features.device, dtype=torch.long).clamp_(0, 2)
        tokens = tokens + position.unsqueeze(0) + self.sensor_embedding(sensor_types).unsqueeze(0)

        query_token = tokens[:, 0:1]
        query = self.to_q(query_token).view(-1, 1, self.heads, self.head_dim).transpose(1, 2)
        key = self.to_k(tokens).view(-1, num_agents, self.heads, self.head_dim).transpose(1, 2)
        value = self.to_v(tokens).view(-1, num_agents, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        quality = confidence.permute(2, 3, 0, 1).reshape(height * width, 1, num_agents)
        valid_tokens = valid.permute(2, 3, 0, 1).reshape(height * width, 1, num_agents) > 0.5
        logits = logits + torch.log(quality.clamp_min(1.0e-6)).unsqueeze(1)
        logits = logits.masked_fill(~valid_tokens.unsqueeze(1), torch.finfo(logits.dtype).min)
        attention = self.dropout(torch.softmax(logits, dim=-1))
        output = torch.matmul(attention, value).transpose(1, 2).reshape(-1, 1, channels)
        output = self.norm1(query_token + self.to_out(output))
        output = self.norm2(output + self.ffn(output))
        return output[:, 0].reshape(height, width, channels).permute(2, 0, 1)


class Where2commFusion(CommunicationStatsMixin, nn.Module):
    """One-round Where2comm with request/confidence spatial message selection."""

    requires_confidence_maps = True
    requires_fusion_context = True

    def __init__(self, args):
        super(Where2commFusion, self).__init__()
        if isinstance(args, int):
            args = {"input_dim": args}
        args = args or {}
        input_dim = int(args.get("input_dim", args.get("feat_dim", 64)))
        model_dim = int(args.get("model_dim", 256))
        heads = int(args.get("heads", 8))
        self.threshold = float(args.get("threshold", args.get("thre", 0.01)))
        self.communication_rounds = int(args.get("communication_rounds", 1))
        if self.communication_rounds < 1:
            raise ValueError("Where2comm communication_rounds must be at least one.")
        self.topk_ratio = args.get("topk_ratio", None)
        self.gaussian_smooth = bool(args.get("gaussian_smooth", True))
        self.gaussian_kernel_size = int(args.get("gaussian_kernel_size", 5))
        self.gaussian_sigma = float(args.get("gaussian_sigma", 1.0))
        if self.gaussian_kernel_size % 2 != 1 or self.gaussian_kernel_size <= 0:
            raise ValueError("Where2comm Gaussian kernel size must be a positive odd integer.")
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
        self.attention = Where2commAttention(
            model_dim, heads=heads, dropout=float(args.get("dropout", 0.0)),
            max_agents=int(args.get("max_agents", args.get("agent_size", 8))),
        )
        self.output_proj = (
            nn.Identity() if model_dim == input_dim
            else nn.Conv2d(model_dim, input_dim, kernel_size=1, bias=False)
        )
        self.model_dim = model_dim
        self.input_dim = input_dim
        self._init_communication_stats(args, method="Where2comm")

    @staticmethod
    def _length(value):
        return int(value.detach().cpu().item()) if torch.is_tensor(value) else int(value)

    def _smooth(self, confidence):
        if not self.gaussian_smooth:
            return confidence
        kernel = _gaussian_kernel(
            self.gaussian_kernel_size, self.gaussian_sigma,
            confidence.device, confidence.dtype,
        )
        return F.conv2d(confidence, kernel, padding=self.gaussian_kernel_size // 2)

    def _selection_mask(self, score):
        if self.topk_ratio is None:
            return score > self.threshold
        ratio = float(self.topk_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("Where2comm topk_ratio must be in (0, 1].")
        flat = score.flatten()
        count = max(1, int(math.ceil(flat.numel() * ratio)))
        threshold = torch.topk(flat, count, sorted=False).values.min()
        return score >= threshold

    def forward(
        self, x, record_len, affine_matrix, confidence_maps=None,
        fusion_context=None, **kwargs,
    ):
        del kwargs
        output_size = x.shape[-2:]
        message = self.spatial_adapter.encode(x)
        _, _, height, width = message.shape
        if confidence_maps is None:
            confidence_maps = torch.sigmoid(x.detach().abs().mean(dim=1, keepdim=True))
            confidence_source = "feature-energy fallback"
        else:
            confidence_maps = torch.sigmoid(confidence_maps)
            confidence_source = "shared detection head"
        if confidence_maps.shape[-2:] != (height, width):
            confidence_maps = F.interpolate(
                confidence_maps, size=(height, width), mode="bilinear", align_corners=False
            )
        confidence_maps = self._smooth(confidence_maps.amax(dim=1, keepdim=True))
        split_features = regroup(message, record_len)
        split_confidence = regroup(confidence_maps, record_len)
        out = []
        feature_elements = index_elements = metadata_elements = message_count = 0
        dense_elements = 0
        per_sample = []

        sensor_type_all = None
        if isinstance(fusion_context, dict):
            sensor_type_all = fusion_context.get("sensor_type_ids")

        offset = 0
        for batch_idx in range(affine_matrix.shape[0]):
            num_agents = self._length(record_len[batch_idx])
            transforms = affine_matrix[batch_idx, 0, :num_agents]
            aligned = warp_affine_simple(split_features[batch_idx], transforms, (height, width))
            aligned_conf = warp_affine_simple(
                split_confidence[batch_idx], transforms, (height, width)
            ).clamp_(0.0, 1.0)
            request = 1.0 - aligned_conf[0:1]
            # The communicated object is an explicit sparse payload
            # (selected feature vectors, y/x coordinates).  A dense tensor is
            # reconstructed only on the receiver side for attention.
            sparse = torch.zeros_like(aligned)
            sparse[0] = aligned[0]
            valid = torch.ones(num_agents, 1, height, width, device=x.device, dtype=x.dtype)
            sample_feature_elements = 0
            for source_idx in range(1, num_agents):
                selection_score = aligned_conf[source_idx, 0]
                if self.communication_rounds > 1:
                    selection_score = selection_score * request[0, 0]
                selected = self._selection_mask(selection_score)
                coordinates = selected.nonzero(as_tuple=False)
                payload = aligned[
                    source_idx, :, coordinates[:, 0], coordinates[:, 1]
                ].transpose(0, 1).contiguous()
                receiver_map = torch.zeros_like(aligned[source_idx])
                receiver_map[
                    :, coordinates[:, 0], coordinates[:, 1]
                ] = payload.transpose(0, 1)
                sparse[source_idx] = receiver_map
                valid[source_idx, 0] = selected.to(x.dtype)
                selected_count = int(payload.shape[0])
                sample_feature_elements += selected_count * self.input_dim
                index_elements += selected_count * 2
                # Receiver request map is a control message sent once per collaborator.
                metadata_elements += height * width
                message_count += 1
            feature_elements += sample_feature_elements
            dense_elements += max(num_agents - 1, 0) * self.input_dim * height * width
            per_sample.append(sample_feature_elements)
            sensor_types = None
            if torch.is_tensor(sensor_type_all):
                sensor_types = sensor_type_all[offset:offset + num_agents]
            out.append(self.attention(
                self.input_proj(sparse), aligned_conf, valid, sensor_types
            ))
            offset += num_agents

        self._record_communication(
            feature_elements=feature_elements,
            index_elements=index_elements,
            metadata_elements=metadata_elements,
            dense_feature_elements=dense_elements,
            message_count=message_count,
            batch_size=len(out),
            mode="request_driven_spatially_sparse_bev",
            rounds=self.communication_rounds,
            per_sample_feature_elements=per_sample,
            notes=(f"{self.communication_rounds}-round confidence/request selection; confidence source: " + confidence_source
                   + ". Sender payload contains selected feature vectors and two int32 coordinates; dense workspace is reconstructed at the receiver."),
        )
        fused = self.output_proj(torch.stack(out))
        return self.spatial_adapter.decode(fused, output_size)
