# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from scipy.optimize import linear_sum_assignment

from opencood.models.sub_modules.base_transformer import FeedForward, PreNormResidual
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.common import regroup
from opencood.models.fuse_modules.communication import (
    CommunicationStatsMixin, SpatialDownsampleAdapter,
)


class AFFSqueezeExcitation(nn.Module):
    """Squeeze-excitation block used by the ERMVP AFF feature extractor."""

    def __init__(self, inp, oup, expansion=0.25):
        super(AFFSqueezeExcitation, self).__init__()
        hidden_dim = max(int(inp * expansion), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y


class AFFFeatureExtract(nn.Module):
    """Depthwise context extractor from ERMVP's AFF global branch."""

    def __init__(self, dim, strides, keep_dim=False):
        super(AFFFeatureExtract, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            AFFSqueezeExcitation(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        self.keep_dim = keep_dim
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=strides, padding=1)

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class AFFAttention(nn.Module):
    """Window attention over agents and local BEV patches."""

    def __init__(self, dim=256, dim_head=32, dropout=0.0, agent_size=2, window_size=4):
        super(AFFAttention, self).__init__()
        if dim % dim_head != 0:
            raise ValueError('AFFAttention dim must be divisible by dim_head.')
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1),
            self.heads,
        )
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))

    def forward(self, x, mask=None):
        batch, agent_size, height, width, window_height, window_width, _, device, heads = (
            *x.shape,
            x.device,
            self.heads,
        )
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), (q, k, v))
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        if mask is not None:
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            sim = sim.masked_fill(mask.unsqueeze(1) == 0, -float('inf'))
        attn = self.attend(sim)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)', l=agent_size, w1=window_height, w2=window_width)
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d', b=batch, x=height, y=width)


class AFFWindowAttentionGlobal(nn.Module):
    """Official ERMVP global-query attention with a shape-safe implementation.

    The release code applies two stride-2 feature extractors for a 4x4
    attention window, performs self-attention on the resulting global tokens,
    averages them across windows, and shares the query with every local
    window.  The published release has an invalid rearrange at that point.
    Here the same intended computation is implemented explicitly: one global
    descriptor is generated per agent and local window, globally refined,
    averaged over windows, and expanded over the P x P query positions.
    """

    def __init__(
        self, dim=256, dim_head=32, dropout=0.0, agent_size=2,
        window_size=4, downsample_stages=None,
    ):
        super(AFFWindowAttentionGlobal, self).__init__()
        if dim % dim_head != 0:
            raise ValueError('AFFWindowAttentionGlobal dim must be divisible by dim_head.')
        self.head_dim = dim_head
        self.num_heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]
        self.agent_size = agent_size
        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1),
            self.num_heads,
        )
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 2)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        if downsample_stages is None:
            downsample_stages = int(round(math.log(max(window_size, 1), 2)))
        self.downsample_stages = int(downsample_stages)
        if self.downsample_stages < 0:
            raise ValueError("ERMVP global-query downsample stages cannot be negative.")
        self.to_q_global = nn.Sequential(*[
            AFFFeatureExtract(dim, (2, 2), keep_dim=False)
            for _ in range(self.downsample_stages)
        ])

    def forward(self, x, mask=None):
        batch, agent_size, height, width, window_height, window_width, _, device, heads = (
            *x.shape,
            x.device,
            self.num_heads,
        )
        del device
        if agent_size != self.agent_size:
            raise ValueError(
                f"ERMVP AFF expected {self.agent_size} agents, got {agent_size}."
            )

        # Eq. (4) / official FeatExtract: obtain one descriptor for every
        # agent and local window.
        global_map = rearrange(
            x, 'b m x y w1 w2 d -> (b m) d (x w1) (y w2)'
        )
        global_map = self.to_q_global(global_map)
        if global_map.shape[-2:] != (height, width):
            global_map = F.adaptive_avg_pool2d(global_map, (height, width))
        global_tokens = rearrange(
            global_map, '(b m) d x y -> b (x y) m d',
            b=batch, m=agent_size,
        )

        # Official global-token refinement and mean over all spatial windows.
        global_attention = torch.matmul(
            global_tokens, global_tokens.transpose(-2, -1)
        )
        global_attention = self.softmax(global_attention * (global_tokens.shape[-1] ** -0.5))
        global_tokens = torch.matmul(global_attention, global_tokens)
        shared_query = global_tokens.mean(dim=1)

        # The shared per-agent query is used at all P x P query positions and
        # repeated for every local window, matching the release intent.
        query_tokens = shared_query[:, :, None, :].expand(
            -1, -1, window_height * window_width, -1
        )
        query_tokens = query_tokens.reshape(
            batch, agent_size * window_height * window_width, -1
        )
        query_tokens = query_tokens[:, None].expand(
            -1, height * width, -1, -1
        ).reshape(
            batch * height * width,
            agent_size * window_height * window_width,
            -1,
        )

        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        tokens = x.shape[1]
        head_dim = x.shape[-1] // self.num_heads
        kv = self.qkv(x).reshape(x.shape[0], tokens, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = query_tokens.reshape(
            x.shape[0], tokens, self.num_heads, head_dim
        ).permute(0, 2, 1, 3)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table(self.relative_position_index)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -float('inf'))
        attn = self.softmax(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)', l=agent_size, w1=window_height, w2=window_width)
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d', b=batch, x=height, y=width)


class AFFFusionBlockMask(nn.Module):
    """ERMVP AFF block with local window attention and global-query attention."""

    def __init__(
        self, input_dim, mlp_dim, dim_head, window_size, agent_size,
        drop_out, global_downsample_stages=None,
    ):
        super(AFFFusionBlockMask, self).__init__()
        self.window_size = window_size
        self.window_attention = PreNormResidual(input_dim, AFFAttention(input_dim, dim_head, drop_out, agent_size, window_size))
        self.window_ffd = PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out))
        self.window_global_attention = PreNormResidual(
            input_dim,
            AFFWindowAttentionGlobal(
                input_dim, dim_head, drop_out, agent_size, window_size,
                downsample_stages=global_downsample_stages,
            ),
        )
        self.window_global_ffd = PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out))

    def forward(self, x, mask):
        mask_swap = rearrange(mask, 'b (x w1) (y w2) e l -> b x y w1 w2 e l', w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d', w1=self.window_size, w2=self.window_size)
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')

        mask_swap = rearrange(mask, 'b (x w1) (y w2) e l -> b x y w1 w2 e l', w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d', w1=self.window_size, w2=self.window_size)
        x = self.window_global_attention(x, mask=mask_swap)
        x = self.window_global_ffd(x)
        return rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')


class AFFFusionEncoder(nn.Module):
    """AFF encoder adapted from ERMVP for dense BEV feature fusion."""

    def __init__(self, args):
        super(AFFFusionEncoder, self).__init__()
        self.depth = int(args.get('depth', 1))
        input_dim = int(args['input_dim'])
        mlp_dim = int(args.get('mlp_dim', input_dim))
        agent_size = int(args.get('agent_size', 2))
        window_size = int(args.get('window_size', 4))
        drop_out = float(args.get('drop_out', 0.0))
        dim_head = int(args.get('dim_head', 32))
        global_downsample_stages = args.get('global_query_downsample_stages')
        self.layers = nn.ModuleList([
            AFFFusionBlockMask(
                input_dim, mlp_dim, dim_head, window_size, agent_size,
                drop_out, global_downsample_stages,
            )
            for _ in range(self.depth)
        ])
        self.mlp_head = nn.Sequential(
            Reduce('b m d h w -> b d h w', 'mean'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w'),
        )

    def forward(self, x, mask):
        for stage in self.layers:
            x = stage(x, mask=mask)
        return self.mlp_head(x)


class AFFFusion(nn.Module):
    """ERMVP AFF fusion wrapper for OpenCOOD intermediate-fusion tensors."""

    def __init__(self, args):
        super(AFFFusion, self).__init__()
        self.max_cav = int(args.get('agent_size', args.get('max_cav', 2)))
        self.window_size = int(args.get('window_size', 4))
        self.encoder = AFFFusionEncoder(args)

    def _pad_to_window(self, x):
        height, width = x.shape[-2:]
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h)), height, width

    def forward(self, x, record_len, affine_matrix):
        _, channels, height, width = x.shape
        batch_size, max_cav = affine_matrix.shape[:2]
        regroup_feature, mask = Regroup(x, record_len, max_cav)
        regroup_feature_new = []
        for b in range(batch_size):
            ego_idx = 0
            regroup_feature_new.append(
                warp_affine_simple(regroup_feature[b], affine_matrix[b, ego_idx], (height, width))
            )
        regroup_feature = torch.stack(regroup_feature_new)
        regroup_feature, raw_h, raw_w = self._pad_to_window(regroup_feature)
        padded_h, padded_w = regroup_feature.shape[-2:]
        com_mask = mask[:, None, None, None, :].repeat(1, padded_h, padded_w, 1, 1)
        fused = self.encoder(regroup_feature, com_mask)
        return fused[..., :raw_h, :raw_w]


def _index_points(points, indices):
    """Batch-aware indexing used by the official ERMVP DPC-KNN sampler."""
    batch = points.shape[0]
    view_shape = [batch] + [1] * (indices.ndim - 1)
    repeat_shape = [1] + list(indices.shape[1:])
    batch_indices = torch.arange(
        batch, device=points.device, dtype=torch.long
    ).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, indices]


def _cluster_dpc_knn(features, cluster_count, neighbors=10):
    """Official release DPC-KNN grouping, made safe for short token lists."""
    with torch.no_grad():
        batch, token_count, channels = features.shape
        neighbors = min(max(int(neighbors), 1), token_count)
        cluster_count = min(max(int(cluster_count), 1), token_count)
        distance_matrix = torch.cdist(features, features) / math.sqrt(channels)
        nearest_distance = torch.topk(
            distance_matrix, k=neighbors, dim=-1, largest=False
        ).values
        density = (-(nearest_distance ** 2).mean(dim=-1)).exp()
        density = density + torch.rand_like(density) * 1.0e-6
        higher_density = density[:, None, :] > density[:, :, None]
        maximum_distance = distance_matrix.flatten(1).amax(dim=-1)[:, None, None]
        distance_to_parent = torch.where(
            higher_density, distance_matrix, maximum_distance
        ).amin(dim=-1)
        center_score = distance_to_parent * density
        center_indices = torch.topk(
            center_score, k=cluster_count, dim=-1
        ).indices
        center_distances = _index_points(distance_matrix, center_indices)
        assignment = center_distances.argmin(dim=1)
        batch_indices = torch.arange(
            batch, device=features.device
        )[:, None].expand(batch, cluster_count)
        cluster_indices = torch.arange(
            cluster_count, device=features.device
        )[None, :].expand(batch, cluster_count)
        assignment[
            batch_indices.reshape(-1), center_indices.reshape(-1)
        ] = cluster_indices.reshape(-1)
    return assignment


class ERMVPFeatureMergeSampler(nn.Module):
    """Official-first ERMVP filter/merge sampler.

    Filtering and DPC-KNN follow the public release.  Two evident release
    wiring errors are corrected using Eqs. (2)-(3) of the paper:

    * actual confidence scores, rather than spatial indices, weight merging;
    * cluster assignments are returned so all L retained positions can be
      reconstructed from the K transmitted representatives.
    """

    def __init__(
        self, channels, topk_ratio=0.2, cluster_sample_ratio=0.2,
        neighbors=10,
    ):
        super(ERMVPFeatureMergeSampler, self).__init__()
        self.channels = int(channels)
        self.topk_ratio = float(topk_ratio)
        self.cluster_sample_ratio = float(cluster_sample_ratio)
        self.neighbors = int(neighbors)
        if not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError("ERMVP topk_ratio must be in (0, 1].")
        if not 0.0 < self.cluster_sample_ratio <= 1.0:
            raise ValueError("ERMVP cluster_sample_ratio must be in (0, 1].")
        self.score_pred_net = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, 1, kernel_size=1),
        )
        self.norm_feature = nn.LayerNorm(
            self.channels, elementwise_affine=False
        )

    def forward(self, feature):
        channels, height, width = feature.shape
        if channels != self.channels:
            raise ValueError(
                f"ERMVP sampler expected {self.channels} channels, got {channels}."
            )
        confidence = self.score_pred_net(feature.unsqueeze(0)).sigmoid()[0, 0]
        retained_count = max(
            1, int(height * width * self.topk_ratio)
        )
        retained_count = min(retained_count, height * width)
        selected_scores, selected_indices = torch.topk(
            confidence.flatten(), k=retained_count, sorted=False
        )

        flat_feature = feature.flatten(1).transpose(0, 1)
        selected_feature = flat_feature[selected_indices]
        grouping_feature = (
            self.norm_feature(selected_feature) * selected_scores[:, None]
        )
        cluster_count = max(
            1, int(math.ceil(retained_count * self.cluster_sample_ratio))
        )
        cluster_count = min(cluster_count, retained_count)
        assignment = _cluster_dpc_knn(
            grouping_feature.unsqueeze(0),
            cluster_count,
            self.neighbors,
        )[0]

        # Paper Eq. (3): confidence-weighted merging of the original vectors.
        weight_sums = torch.zeros(
            cluster_count, device=feature.device, dtype=feature.dtype
        ).index_add_(0, assignment, selected_scores.to(feature.dtype))
        weight_sums = weight_sums.clamp_min(1.0e-6)
        feature_sums = torch.zeros(
            cluster_count, channels, device=feature.device, dtype=feature.dtype
        ).index_add_(
            0,
            assignment,
            selected_feature * selected_scores.to(feature.dtype)[:, None],
        )
        merged_feature = feature_sums / weight_sums[:, None]
        selected_y = torch.div(
            selected_indices, width, rounding_mode="floor"
        )
        selected_x = selected_indices.remainder(width)
        selected_coordinates = torch.stack(
            [selected_y, selected_x], dim=1
        )
        return {
            "merged_feature": merged_feature.contiguous(),
            "selected_coordinates": selected_coordinates,
            "cluster_assignment": assignment,
            "confidence": confidence,
            "selected_scores": selected_scores,
            "retained_count": retained_count,
            "cluster_count": cluster_count,
        }

    @staticmethod
    def reconstruct(payload, output_shape):
        """Reconstruct all retained positions from transmitted representatives."""
        channels, height, width = output_shape
        receiver_feature = payload["merged_feature"][
            payload["cluster_assignment"]
        ]
        receiver_map = receiver_feature.new_zeros(channels, height, width)
        coordinates = payload["selected_coordinates"]
        receiver_map[:, coordinates[:, 0], coordinates[:, 1]] = (
            receiver_feature.transpose(0, 1)
        )
        return receiver_map


class ERMVPFeatureSpatialCalibration(nn.Module):
    """Paper-complete FSC for the module missing from the public release.

    Informative sparse regions are matched by minimum-cost bipartite
    assignment, geometrically verified with RANSAC/SVD, and subjected to the
    paper's error-modulation decision between original and calibrated maps.
    """

    def __init__(
        self, max_matches=128, ransac_iterations=16,
        tolerance_pixels=2.5, position_cost_weight=0.1,
    ):
        super(ERMVPFeatureSpatialCalibration, self).__init__()
        self.max_matches = int(max_matches)
        self.ransac_iterations = int(ransac_iterations)
        self.tolerance_pixels = float(tolerance_pixels)
        self.position_cost_weight = float(position_cost_weight)

    @staticmethod
    def _rigid(source, target):
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)
        source_centered = source - source_mean
        target_centered = target - target_mean
        covariance = source_centered.transpose(0, 1) @ target_centered
        u, _, vh = torch.linalg.svd(covariance)
        rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
        if torch.det(rotation) < 0:
            vh = vh.clone()
            vh[-1] = -vh[-1]
            rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
        translation = target_mean[0] - rotation @ source_mean[0]
        return rotation, translation

    @staticmethod
    def _apply_transform(points, rotation, translation):
        return points @ rotation.transpose(0, 1) + translation

    def _consensus_transform(self, source, target):
        if source.shape[0] < 2:
            identity = torch.eye(2, device=source.device, dtype=source.dtype)
            zero = torch.zeros(2, device=source.device, dtype=source.dtype)
            return identity, zero, torch.tensor(0.0, device=source.device, dtype=source.dtype)
        best_inliers = None
        best_count = -1
        best_error = float("inf")
        iterations = max(self.ransac_iterations, 1)
        for _ in range(iterations):
            sample = torch.randperm(source.shape[0], device=source.device)[:2]
            rotation, translation = self._rigid(source[sample], target[sample])
            residual = torch.linalg.vector_norm(
                self._apply_transform(source, rotation, translation) - target,
                dim=1,
            )
            inliers = residual <= self.tolerance_pixels
            count = int(inliers.sum().detach().cpu().item())
            inlier_error = (
                float(residual[inliers].mean().detach().cpu().item())
                if count > 0 else float("inf")
            )
            if count > best_count or (count == best_count and inlier_error < best_error):
                best_count = count
                best_error = inlier_error
                best_inliers = inliers
        if best_inliers is None or int(best_inliers.sum().item()) < 2:
            best_inliers = torch.ones(source.shape[0], device=source.device, dtype=torch.bool)
        rotation, translation = self._rigid(source[best_inliers], target[best_inliers])
        error = torch.linalg.vector_norm(
            self._apply_transform(
                source[best_inliers], rotation, translation
            ) - target[best_inliers],
            dim=1,
        ).mean()
        return rotation, translation, error

    @staticmethod
    def _pixel_inverse_theta(rotation, translation, height, width):
        transform = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
        transform[:2, :2] = rotation
        transform[:2, 2] = translation
        inverse = torch.linalg.inv(transform)
        pixel_to_norm = torch.tensor([
            [2.0 / width, 0.0, 1.0 / width - 1.0],
            [0.0, 2.0 / height, 1.0 / height - 1.0],
            [0.0, 0.0, 1.0],
        ], device=rotation.device, dtype=rotation.dtype)
        norm_to_pixel = torch.linalg.inv(pixel_to_norm)
        return (pixel_to_norm @ inverse @ norm_to_pixel)[:2]

    @staticmethod
    def _indices_to_points(indices, width, dtype):
        return torch.stack([
            indices.remainder(width),
            torch.div(indices, width, rounding_mode="floor"),
        ], dim=1).to(dtype)

    def _informative_indices(self, feature, count=None, require_nonzero=False):
        energy = feature.abs().sum(dim=0).flatten()
        if require_nonzero:
            available = int((energy > 0).sum().detach().cpu().item())
        else:
            available = energy.numel()
        if count is None:
            count = self.max_matches
        count = min(int(count), self.max_matches, available)
        if count <= 0:
            return torch.empty(
                0, device=feature.device, dtype=torch.long
            )
        return torch.topk(energy, count, sorted=False).indices

    def _linear_assignment(self, ego, source, source_idx, ego_idx):
        channels, height, width = source.shape
        source_desc = F.normalize(
            source.flatten(1)[:, source_idx].transpose(0, 1), dim=1
        )
        ego_desc = F.normalize(
            ego.flatten(1)[:, ego_idx].transpose(0, 1), dim=1
        )
        feature_cost = 1.0 - source_desc @ ego_desc.transpose(0, 1)
        source_points = self._indices_to_points(
            source_idx, width, source.dtype
        )
        ego_points = self._indices_to_points(ego_idx, width, ego.dtype)
        diagonal = math.sqrt(height * height + width * width)
        position_cost = torch.cdist(source_points, ego_points) / max(diagonal, 1.0)
        cost = feature_cost + self.position_cost_weight * position_cost
        source_match, ego_match = linear_sum_assignment(
            cost.detach().float().cpu().numpy()
        )
        source_match = torch.as_tensor(
            source_match, device=source.device, dtype=torch.long
        )
        ego_match = torch.as_tensor(
            ego_match, device=source.device, dtype=torch.long
        )
        return source_points[source_match], ego_points[ego_match]

    def _overlap_ratio(self, ego, candidate, sample_count):
        _, height, width = candidate.shape
        source_idx = self._informative_indices(
            candidate, count=sample_count, require_nonzero=True
        )
        if source_idx.numel() == 0:
            return candidate.new_tensor(0.0)
        ego_idx = self._informative_indices(
            ego, count=max(int(source_idx.numel()), 1)
        )
        ego_mask = candidate.new_zeros(1, 1, height, width)
        ego_mask.view(-1)[ego_idx] = 1.0
        ego_mask = F.max_pool2d(
            ego_mask, kernel_size=3, stride=1, padding=1
        )[0, 0]
        source_y = torch.div(source_idx, width, rounding_mode="floor")
        source_x = source_idx.remainder(width)
        return ego_mask[source_y, source_x].mean()

    def forward(self, ego, source):
        _, height, width = source.shape
        source_idx = self._informative_indices(
            source, require_nonzero=True
        )
        if source_idx.numel() < 2:
            zero = torch.tensor(
                0.0, device=source.device, dtype=source.dtype
            )
            return source, zero, False
        ego_idx = self._informative_indices(
            ego, count=int(source_idx.numel())
        )
        source_points, ego_points = self._linear_assignment(
            ego, source, source_idx, ego_idx
        )
        rotation, translation, error = self._consensus_transform(source_points, ego_points)
        theta = self._pixel_inverse_theta(rotation, translation, height, width)
        calibrated = warp_affine_simple(source.unsqueeze(0), theta.unsqueeze(0), (height, width))[0]
        original_overlap = self._overlap_ratio(
            ego, source, int(source_idx.numel())
        )
        calibrated_overlap = self._overlap_ratio(
            ego, calibrated, int(source_idx.numel())
        )
        use_calibrated = bool(
            calibrated_overlap.detach().cpu().item()
            >= original_overlap.detach().cpu().item()
        )
        selected = calibrated if use_calibrated else source
        return selected, error, use_calibrated


class ERMVPAccuracyEnhancedInteraction(nn.Module):
    """Paper Eq. (5) local Q-K-V cross-attention plus split attention."""

    def __init__(self, channels, dim_head=32, kernel_size=3):
        super(ERMVPAccuracyEnhancedInteraction, self).__init__()
        if channels % dim_head != 0:
            raise ValueError("ERMVP AEI channels must be divisible by dim_head.")
        if kernel_size % 2 != 1:
            raise ValueError("ERMVP AEI kernel size must be odd.")
        self.channels = int(channels)
        self.dim_head = int(dim_head)
        self.heads = self.channels // self.dim_head
        self.scale = self.dim_head ** -0.5
        self.kernel_size = int(kernel_size)
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1, bias=False)
        hidden = max(channels // 4, 1)
        self.split_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
            nn.GELU(), nn.Conv2d(hidden, channels * 2, 1),
        )

    @staticmethod
    def _shift(feature, offset_y, offset_x):
        height, width = feature.shape[-2:]
        radius = max(abs(offset_y), abs(offset_x))
        padded = F.pad(feature, (radius, radius, radius, radius))
        start_y = radius + offset_y
        start_x = radius + offset_x
        return padded[
            :, :, :, start_y:start_y + height, start_x:start_x + width
        ]

    def _local_cross_attention(self, fused, ego):
        batch, _, height, width = fused.shape
        query = self.to_q(ego).reshape(
            batch, self.heads, self.dim_head, height, width
        )
        key = self.to_k(fused).reshape(
            batch, self.heads, self.dim_head, height, width
        )
        value = self.to_v(fused).reshape(
            batch, self.heads, self.dim_head, height, width
        )
        radius = self.kernel_size // 2
        offsets = [
            (offset_y, offset_x)
            for offset_y in range(-radius, radius + 1)
            for offset_x in range(-radius, radius + 1)
        ]
        valid = fused.new_ones(batch, 1, 1, height, width)
        score_list = []
        valid_list = []
        for offset_y, offset_x in offsets:
            shifted_key = self._shift(key, offset_y, offset_x)
            score_list.append(
                (query * shifted_key).sum(dim=2) * self.scale
            )
            shifted_valid = self._shift(valid, offset_y, offset_x)
            valid_list.append(shifted_valid[:, 0, 0] > 0)
        scores = torch.stack(score_list, dim=2)
        valid_mask = torch.stack(valid_list, dim=1).unsqueeze(1)
        scores = scores.masked_fill(~valid_mask, -float("inf"))
        attention = scores.softmax(dim=2)
        output = torch.zeros_like(query)
        for neighbor_index, (offset_y, offset_x) in enumerate(offsets):
            shifted_value = self._shift(value, offset_y, offset_x)
            output = output + (
                attention[:, :, neighbor_index].unsqueeze(2) * shifted_value
            )
        output = output.reshape(batch, self.channels, height, width)
        return self.to_out(output)

    def forward(self, fused, ego):
        enhanced = fused + self._local_cross_attention(fused, ego)
        weights = self.split_attention(enhanced + ego).view(
            fused.shape[0], 2, fused.shape[1], 1, 1
        ).softmax(dim=1)
        return weights[:, 0] * enhanced + weights[:, 1] * ego


class ERMVPFusion(CommunicationStatsMixin, nn.Module):
    """Official-first, paper-complete ERMVP fusion pipeline.

    The public input/output channel contract is intentionally preserved.
    Features are projected to the official 256-D internal space, processed by
    release-style FMS and AFF, completed with paper FSC/AEI, and projected
    back to the original input dimension.
    """

    requires_confidence_maps = False
    requires_fusion_context = False

    def __init__(self, args):
        super(ERMVPFusion, self).__init__()
        args = args or {}
        input_dim = int(args.get("input_dim", 64))
        model_dim = int(args.get("model_dim", 256))
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.input_proj = (
            nn.Identity() if input_dim == model_dim
            else nn.Conv2d(input_dim, model_dim, 1, bias=False)
        )
        self.spatial_downsample_stages = int(
            args.get("spatial_downsample_stages", 1)
        )
        self.spatial_adapter = SpatialDownsampleAdapter(
            input_dim, self.spatial_downsample_stages
        )
        self.sampler = ERMVPFeatureMergeSampler(
            model_dim,
            args.get("topk_ratio", 0.2),
            args.get("cluster_sample_ratio", 0.2),
            args.get("cluster_neighbors", 10),
        )
        self.calibrator = ERMVPFeatureSpatialCalibration(
            args.get("max_matches", 128), args.get("ransac_iterations", 16),
            args.get("tolerance_pixels", 2.5),
            args.get("fsc_position_cost_weight", 0.1),
        )
        self.enable_fsc = bool(args.get("enable_fsc", True))
        self.enable_aei = bool(args.get("enable_aei", True))

        # Official release hyperparameters are the default.  Dedicated
        # official_* overrides make any departure explicit instead of silently
        # inheriting the previous adapted implementation's values.
        self.window_size = int(args.get("official_window_size", 4))
        self.agent_size = int(
            args.get("official_agent_size", args.get("agent_size", 2))
        )
        aff_dim_head = int(args.get("official_dim_head", 32))
        aff_args = {
            "input_dim": model_dim,
            "mlp_dim": int(args.get("official_mlp_dim", model_dim)),
            "agent_size": self.agent_size,
            "window_size": self.window_size,
            "dim_head": aff_dim_head,
            "drop_out": float(args.get("official_drop_out", 0.1)),
            "depth": int(args.get("official_depth", 3)),
            "global_query_downsample_stages": int(
                args.get(
                    "official_global_query_downsample_stages",
                    round(math.log(max(self.window_size, 1), 2)),
                )
            ),
        }
        self.aff = AFFFusionEncoder(aff_args)
        self.aei = ERMVPAccuracyEnhancedInteraction(
            model_dim,
            dim_head=aff_dim_head,
            kernel_size=int(args.get("aei_kernel_size", 3)),
        )
        self.output_proj = (
            nn.Identity() if model_dim == input_dim
            else nn.Conv2d(model_dim, input_dim, 1, bias=False)
        )
        self._init_communication_stats(args, method="ERMVP")

    @staticmethod
    def _length(value):
        return int(value.detach().cpu().item()) if torch.is_tensor(value) else int(value)

    def forward(
        self, x, record_len, affine_matrix, confidence_maps=None,
        fusion_context=None, **kwargs,
    ):
        del confidence_maps, fusion_context, kwargs
        output_size = x.shape[-2:]
        message = self.spatial_adapter.encode(x)
        _, _, raw_h, raw_w = message.shape
        projected_message = self.input_proj(message)
        split_feature = regroup(projected_message, record_len)
        batch_size, max_cav = affine_matrix.shape[:2]
        if max_cav != self.agent_size:
            raise ValueError(
                "ERMVP official AFF agent size does not match affine_matrix: "
                f"{self.agent_size} vs {max_cav}."
            )
        batches = []
        ego_features = []
        feature_elements = index_elements = dense_elements = message_count = 0
        per_sample = []
        calibration_errors = []
        calibrated_message_count = 0

        for batch_idx in range(batch_size):
            num_agents = self._length(record_len[batch_idx])
            if num_agents < 1 or num_agents > max_cav:
                raise ValueError(
                    f"Invalid ERMVP record_len {num_agents} for max_cav {max_cav}."
                )
            transmitted = [split_feature[batch_idx][0]]
            sample_elements = 0
            for source_idx in range(1, num_agents):
                payload = self.sampler(
                    split_feature[batch_idx][source_idx]
                )
                receiver_map = self.sampler.reconstruct(
                    payload,
                    (self.model_dim, raw_h, raw_w),
                )
                transmitted.append(receiver_map)
                cluster_count = int(payload["cluster_count"])
                retained_count = int(payload["retained_count"])
                sample_elements += cluster_count * self.model_dim
                # Two coordinates and one cluster assignment for each retained
                # position constitute the paper's index correspondence.
                index_elements += retained_count * 3
                message_count += 1
            transmitted = torch.stack(transmitted)
            transforms = affine_matrix[batch_idx, 0, :num_agents]
            aligned = warp_affine_simple(transmitted, transforms, (raw_h, raw_w))
            calibrated = [aligned[0]]
            for source_idx in range(1, num_agents):
                if self.enable_fsc:
                    source, error, used_calibration = self.calibrator(
                        aligned[0], aligned[source_idx]
                    )
                    calibration_errors.append(error.detach())
                    calibrated_message_count += int(used_calibration)
                else:
                    source = aligned[source_idx]
                calibrated.append(source)
            calibrated = torch.stack(calibrated)
            if num_agents < max_cav:
                calibrated = torch.cat([
                    calibrated,
                    torch.zeros(
                        max_cav - num_agents, self.model_dim, raw_h, raw_w,
                        device=x.device, dtype=x.dtype,
                    ),
                ], dim=0)
            batches.append(calibrated)
            ego_features.append(aligned[0])
            feature_elements += sample_elements
            dense_elements += (
                max(num_agents - 1, 0)
                * self.model_dim * raw_h * raw_w
            )
            per_sample.append(sample_elements)

        features = torch.stack(batches)
        ego_features = torch.stack(ego_features)
        pad_h = (self.window_size - raw_h % self.window_size) % self.window_size
        pad_w = (self.window_size - raw_w % self.window_size) % self.window_size
        if pad_h or pad_w:
            features = F.pad(features, (0, pad_w, 0, pad_h))
        height, width = features.shape[-2:]
        valid_agents = torch.zeros(batch_size, max_cav, device=x.device, dtype=x.dtype)
        for batch_idx, length in enumerate(record_len):
            valid_agents[batch_idx, :self._length(length)] = 1.0
        com_mask = valid_agents[:, None, None, None, :].repeat(1, height, width, 1, 1)
        fused = self.aff(features, com_mask)[..., :raw_h, :raw_w]
        if self.enable_aei:
            fused = self.aei(fused, ego_features)
        fused = self.output_proj(fused)
        fused = self.spatial_adapter.decode(fused, output_size)
        mean_error = 0.0
        if calibration_errors:
            mean_error = float(torch.stack(calibration_errors).mean().cpu().item())
        self._record_communication(
            feature_elements=feature_elements,
            index_elements=index_elements,
            dense_feature_elements=dense_elements,
            message_count=message_count,
            batch_size=batch_size,
            mode="fms_clustered_sparse_bev",
            per_sample_feature_elements=per_sample,
            notes=(
                "Official SortSampler/global-top-k and DPC-KNN; paper Eq. (3) "
                "confidence merge; receiver reconstructs every retained position "
                "from two coordinates plus one cluster assignment. "
                f"FSC enabled={self.enable_fsc}, calibrated "
                f"{calibrated_message_count}/{message_count} messages, mean residual "
                f"{mean_error:.4f} pixels; AEI enabled={self.enable_aei}."
            ),
        )
        return fused
