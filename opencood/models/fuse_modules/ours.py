# -*- coding: utf-8 -*-
"""Density-guided Complementary Local Correction fusion.

This module contains DCG and CLC only. The ego feature remains the prediction
anchor, and aligned partner features can enter only through local residual
cross-attention.
"""

import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.fuse_modules.common import regroup
from opencood.models.fuse_modules.communication import CommunicationStatsMixin


class DirectionalSwinCrossAttention(nn.Module):
    """Ego-query/partner-key Swin attention with a zero-value null token."""

    def __init__(
        self,
        channels,
        window_size=4,
        num_heads=4,
        dropout=0.0,
        output_init_gain=0.01,
    ):
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.channels % self.num_heads != 0:
            raise ValueError(
                f"channels ({self.channels}) must be divisible by num_heads "
                f"({self.num_heads})."
            )
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.output_init_gain = float(output_init_gain)
        if self.output_init_gain <= 0.0:
            raise ValueError("output_init_gain must be positive.")
        self.q_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        # Bias-free output preserves the null-token guarantee: if every
        # complementarity value is zero, the local residual is exactly zero.
        self.out_proj = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=False
        )
        self.attn_drop = nn.Dropout(float(dropout))

        table_size = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, self.num_heads)
        )
        self.register_buffer(
            "relative_position_index",
            self._build_relative_position_index(),
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        # A zero output projection makes the residual initially safe but blocks
        # all first-step gradients to Q/K/V and relative-position parameters.
        # Small-gain Xavier keeps the correction near zero without disconnecting
        # the upstream attention branch.
        nn.init.xavier_uniform_(
            self.out_proj.weight, gain=self.output_init_gain
        )

    def _build_relative_position_index(self):
        ws = self.window_size
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        )
        coords_flat = coords.flatten(1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += ws - 1
        relative[:, :, 1] += ws - 1
        relative[:, :, 0] *= 2 * ws - 1
        return relative.sum(-1).long()

    def _relative_position_bias(self, num_partners, dtype, device):
        tokens = self.window_size * self.window_size
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        bias = bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        return bias.to(device=device, dtype=dtype).repeat(1, 1, num_partners)

    def _pad(self, x):
        height, width = x.shape[-2:]
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        return F.pad(x, (0, pad_w, 0, pad_h)), height, width

    def _partition(self, x):
        batch, channels, height, width = x.shape
        ws = self.window_size
        x = x.view(batch, channels, height // ws, ws, width // ws, ws)
        x = x.permute(2, 4, 0, 3, 5, 1).contiguous()
        return x.view((height // ws) * (width // ws), batch * ws * ws, channels)

    def _merge_query_windows(self, windows, height, width):
        ws = self.window_size
        channels = windows.shape[-1]
        x = windows.view(height // ws, width // ws, ws, ws, channels)
        x = x.permute(4, 0, 2, 1, 3).contiguous()
        return x.view(1, channels, height, width)

    def _window_mask(
        self,
        num_partners,
        raw_h,
        raw_w,
        padded_h,
        padded_w,
        shift_size,
        device,
        dtype,
    ):
        valid = torch.zeros((1, 1, padded_h, padded_w), device=device)
        valid[:, :, :raw_h, :raw_w] = 1.0
        if shift_size > 0:
            valid = torch.roll(valid, shifts=(-shift_size, -shift_size), dims=(-2, -1))
        valid_keys = self._partition(valid).squeeze(-1).bool()
        valid_keys = valid_keys.repeat(1, num_partners)
        mask = torch.zeros(
            valid_keys.shape[0],
            self.window_size * self.window_size,
            valid_keys.shape[1],
            device=device,
            dtype=dtype,
        )
        mask = mask.masked_fill(~valid_keys.unsqueeze(1), torch.finfo(dtype).min)

        if shift_size > 0 and padded_h > self.window_size and padded_w > self.window_size:
            region = torch.zeros((1, 1, padded_h, padded_w), device=device)
            ws = self.window_size
            h_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
            count = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    region[:, :, h_slice, w_slice] = count
                    count += 1
            region_tokens = self._partition(region).squeeze(-1)
            key_regions = region_tokens.repeat(1, num_partners)
            different_region = region_tokens.unsqueeze(2) != key_regions.unsqueeze(1)
            mask = mask.masked_fill(different_region, torch.finfo(dtype).min)
        return mask

    def forward(
        self,
        ego_feature,
        partner_features,
        complementarity,
        key_mask=None,
        use_complementarity_guidance=True,
        shift_size=0,
    ):
        num_partners = partner_features.shape[0]
        if num_partners == 0:
            return torch.zeros_like(ego_feature)

        ego_pad, raw_h, raw_w = self._pad(ego_feature)
        partner_pad, _, _ = self._pad(partner_features)
        comp_pad, _, _ = self._pad(complementarity)
        if key_mask is None:
            key_mask = torch.ones_like(complementarity)
        key_mask_pad, _, _ = self._pad(key_mask)
        padded_h, padded_w = ego_pad.shape[-2:]
        shift_size = int(shift_size)
        shift_active = (
            shift_size > 0
            and padded_h > self.window_size
            and padded_w > self.window_size
        )
        if shift_active:
            shifts = (-shift_size, -shift_size)
            ego_pad = torch.roll(ego_pad, shifts=shifts, dims=(-2, -1))
            partner_pad = torch.roll(partner_pad, shifts=shifts, dims=(-2, -1))
            comp_pad = torch.roll(comp_pad, shifts=shifts, dims=(-2, -1))
            key_mask_pad = torch.roll(
                key_mask_pad, shifts=shifts, dims=(-2, -1)
            )

        query = self._partition(self.q_proj(ego_pad))
        key = self._partition(self.k_proj(partner_pad))
        value = self._partition(self.v_proj(partner_pad))
        comp = self._partition(comp_pad).squeeze(-1)
        valid_keys = self._partition(key_mask_pad).squeeze(-1) > 0.0

        num_windows, query_tokens, _ = query.shape
        key_tokens = key.shape[1]
        query = query.view(
            num_windows, query_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            num_windows, key_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            num_windows, key_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        score = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        score = score + self._relative_position_bias(
            num_partners, score.dtype, score.device
        ).unsqueeze(0)
        if use_complementarity_guidance:
            # C acts as an additive attention bias and a key-validity gate.
            score = score + torch.log(comp.clamp_min(1.0e-6))[:, None, None, :]
            score = score.masked_fill(
                comp[:, None, None, :] <= 0.0, torch.finfo(score.dtype).min
            )
        else:
            # Ablate only complementarity guidance. Keep M as a binary key
            # mask so sparse BEV cells remain sparse instead of becoming
            # zero-valued attention tokens.
            score = score.masked_fill(
                ~valid_keys[:, None, None, :], torch.finfo(score.dtype).min
            )
        window_mask = self._window_mask(
            num_partners,
            raw_h,
            raw_w,
            padded_h,
            padded_w,
            shift_size if shift_active else 0,
            score.device,
            score.dtype,
        )
        score = score + window_mask.unsqueeze(1)

        null_score = torch.zeros(
            num_windows,
            self.num_heads,
            query_tokens,
            1,
            device=score.device,
            dtype=score.dtype,
        )
        null_value = torch.zeros(
            num_windows,
            self.num_heads,
            1,
            self.head_dim,
            device=value.device,
            dtype=value.dtype,
        )
        attention = self.attn_drop(
            torch.softmax(torch.cat([null_score, score], dim=-1), dim=-1)
        )
        context = torch.matmul(attention, torch.cat([null_value, value], dim=2))
        context = context.transpose(1, 2).contiguous().view(
            num_windows, query_tokens, self.channels
        )
        context = self._merge_query_windows(context, padded_h, padded_w)
        if shift_active:
            context = torch.roll(
                context, shifts=(shift_size, shift_size), dims=(-2, -1)
            )
        return self.out_proj(context[..., :raw_h, :raw_w])


class ComplementarityGuidedCLCFusion(CommunicationStatsMixin, nn.Module):
    """Density-guided Complementary Local Correction.

    The ego feature is always the prediction anchor. Aligned partner features
    can enter only through regular-window and shifted-window directional
    residual cross-attention.
    """

    def __init__(self, feature_dims, args=None):
        super().__init__()
        args = dict(args or {})
        if isinstance(feature_dims, (list, tuple)):
            feature_dims = feature_dims[0]
        self.channels = int(args.get("feat_dim", feature_dims))
        self.window_size = int(args.get("swin_window_size", 4))
        self.shift_size = max(
            0,
            min(
                int(args.get("swin_shift_size", self.window_size // 2)),
                self.window_size - 1,
            ),
        )
        num_heads = int(args.get("swin_num_heads", 4))
        dropout = float(args.get("swin_drop", 0.0))
        residual_output_init_gain = float(
            args.get("residual_output_init_gain", 0.01)
        )
        if residual_output_init_gain <= 0.0:
            raise ValueError("residual_output_init_gain must be positive.")
        self.use_density_quality = bool(args.get("use_density_quality", True))
        self.use_complementarity_guidance = bool(
            args.get("use_complementarity_guidance", self.use_density_quality)
        )
        self.require_support_cue = bool(
            args.get("require_support_cue", self.use_density_quality)
        )
        self.density_sparse_communication = bool(
            args.get("density_sparse_communication", False)
        )
        if self.use_density_quality and not self.require_support_cue:
            raise ValueError(
                "use_density_quality=True requires require_support_cue=True."
            )
        if self.use_complementarity_guidance and not self.use_density_quality:
            raise ValueError(
                "use_complementarity_guidance=True requires "
                "use_density_quality=True."
            )
        if not self.use_density_quality and self.require_support_cue:
            raise ValueError(
                "CLC without DCG must set require_support_cue=False."
            )
        if not self.use_density_quality and self.density_sparse_communication:
            raise ValueError(
                "CLC without DCG must use full dense BEV communication; set "
                "density_sparse_communication=False."
            )
        self.communication_density_threshold = float(
            args.get("communication_density_threshold", 0.0)
        )
        if self.communication_density_threshold < 0.0:
            raise ValueError("communication_density_threshold must be non-negative.")
        self.local_window = DirectionalSwinCrossAttention(
            self.channels,
            self.window_size,
            num_heads,
            dropout,
            output_init_gain=residual_output_init_gain,
        )
        self.local_shifted_window = DirectionalSwinCrossAttention(
            self.channels,
            self.window_size,
            num_heads,
            dropout,
            output_init_gain=residual_output_init_gain,
        )
        self.save_debug = False
        self.latest_debug = None
        self._init_communication_stats(args, method="DCG-CLC")

    @staticmethod
    def _scene_length(record_len, batch_idx):
        value = record_len[batch_idx]
        return int(value.detach().cpu().item()) if torch.is_tensor(value) else int(value)

    @staticmethod
    def _align_scene(
        scene_features,
        affine_matrix,
        batch_idx,
        num_agents,
        size,
        mode="bilinear",
    ):
        transforms = affine_matrix[batch_idx, 0, :num_agents]
        grid = F.affine_grid(
            transforms,
            [num_agents, scene_features.shape[1], size[0], size[1]],
            align_corners=False,
        ).to(scene_features)
        return F.grid_sample(
            scene_features,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )

    def _build_scene_messages(self, scene_features, scene_support):
        """Build the sender payload and receiver-side sparse reconstruction.

        The ego feature/support are local and therefore never counted or
        sparsified.  A partner transmits a feature vector, its density value,
        and a (y, x) coordinate only where its native density map is above the
        configured threshold.
        """

        num_agents, channels, height, width = scene_features.shape
        collaborators = max(num_agents - 1, 0)
        dense_feature_elements = collaborators * channels * height * width
        if collaborators == 0:
            return scene_features, scene_support, 0, 0, 0, 0, dense_feature_elements

        if not self.density_sparse_communication:
            density_elements = (
                collaborators * height * width if scene_support is not None else 0
            )
            return (
                scene_features,
                scene_support,
                dense_feature_elements,
                0,
                density_elements,
                collaborators,
                dense_feature_elements,
            )

        if scene_support is None:
            raise ValueError(
                "density_sparse_communication requires a native density/support map."
            )

        transmitted_features = [scene_features[0]]
        transmitted_support = [scene_support[0]]
        feature_elements = 0
        index_elements = 0
        density_elements = 0
        message_count = 0
        for source_idx in range(1, num_agents):
            selected = (
                scene_support[source_idx, 0]
                > self.communication_density_threshold
            )
            coordinates = selected.nonzero(as_tuple=False)
            receiver_feature = torch.zeros_like(scene_features[source_idx])
            receiver_support = torch.zeros_like(scene_support[source_idx])
            selected_count = int(coordinates.shape[0])
            if selected_count > 0:
                payload = scene_features[
                    source_idx, :, coordinates[:, 0], coordinates[:, 1]
                ].transpose(0, 1).contiguous()
                density_payload = scene_support[
                    source_idx, 0, coordinates[:, 0], coordinates[:, 1]
                ]
                receiver_feature[
                    :, coordinates[:, 0], coordinates[:, 1]
                ] = payload.transpose(0, 1)
                receiver_support[
                    0, coordinates[:, 0], coordinates[:, 1]
                ] = density_payload
                feature_elements += selected_count * channels
                index_elements += selected_count * 2
                density_elements += selected_count
                message_count += 1
            transmitted_features.append(receiver_feature)
            transmitted_support.append(receiver_support)

        return (
            torch.stack(transmitted_features),
            torch.stack(transmitted_support),
            feature_elements,
            index_elements,
            density_elements,
            message_count,
            dense_feature_elements,
        )

    def _fuse_scene(
        self,
        aligned_features,
        aligned_support,
        batch_idx,
        aligned_hard_mask=None,
    ):
        if not self.use_density_quality:
            aligned_support = aligned_features.new_zeros(
                (aligned_features.shape[0], 1, *aligned_features.shape[-2:])
            )
            aligned_hard_mask = aligned_features.new_ones(
                (aligned_features.shape[0], 1, *aligned_features.shape[-2:])
            )
        elif aligned_hard_mask is None:
            aligned_hard_mask = (aligned_support > 0).to(
                dtype=aligned_features.dtype
            )
        else:
            aligned_hard_mask = (aligned_hard_mask > 0).to(
                device=aligned_features.device,
                dtype=aligned_features.dtype,
            )
        # With Q enabled, aligned_features already contains the once-only
        # pre-cooperative hard-filtered representation. The no-Q ablation
        # instead arrives as the complete dense BEV representation.
        ego_feature = aligned_features[:1]
        if aligned_features.shape[0] <= 1:
            return ego_feature.squeeze(0)

        partner_hard_mask = aligned_hard_mask[1:]
        partner_features = aligned_features[1:]
        ego_support = aligned_support[:1].clamp(0.0, 1.0)
        partner_support = aligned_support[1:].clamp(0.0, 1.0)
        # Q alone defines complementarity. A cell is complementary when the
        # ego has weak LiDAR support and the aligned partner has strong support.
        complementarity = (
            (1.0 - ego_support).expand_as(partner_support)
            * partner_support
            * partner_hard_mask
            if self.use_density_quality
            else partner_support.new_ones(partner_support.shape)
        ).clamp(0.0, 1.0)

        delta_local_window = self.local_window(
            ego_feature,
            partner_features,
            complementarity,
            key_mask=partner_hard_mask,
            use_complementarity_guidance=self.use_complementarity_guidance,
            shift_size=0,
        )
        local_feature = ego_feature + delta_local_window
        delta_local_shifted = self.local_shifted_window(
            local_feature,
            partner_features,
            complementarity,
            key_mask=partner_hard_mask,
            use_complementarity_guidance=self.use_complementarity_guidance,
            shift_size=self.shift_size,
        )
        local_feature = local_feature + delta_local_shifted
        delta_local = local_feature - ego_feature
        capture_debug = self.save_debug and batch_idx == 0

        if capture_debug:
            self.latest_debug = {
                "support_Q_ego": ego_support[0],
                "support_Q_partner": partner_support[0],
                "support_M_ego": aligned_hard_mask[0],
                "support_M_partner": partner_hard_mask[0],
                "complementarity_C_partner": complementarity[0],
                "delta_local_window": delta_local_window[0],
                "delta_local_shifted": delta_local_shifted[0],
                "delta_local": delta_local[0],
            }
        return local_feature.squeeze(0)

    def forward(
        self,
        bev_features,
        record_len,
        ego_alignment_matrix,
        support_mask=None,
        pairwise_pose_matrix=None,
        **legacy_kwargs,
    ):
        del pairwise_pose_matrix, legacy_kwargs
        if not self.use_density_quality:
            # A no-Q ablation also removes M: ignore any accidentally supplied
            # support tensor and transmit/use the complete dense BEV feature.
            support_mask = None
        if support_mask is None and self.require_support_cue:
            raise ValueError(
                "ComplementarityGuidedCLCFusion requires the LiDAR density cue. "
                "Enable model.args.lidar_support_mask in the YAML."
            )
        if support_mask is not None and support_mask.ndim == 3:
            support_mask = support_mask.unsqueeze(1)

        _, _, height, width = bev_features.shape
        split_features = regroup(bev_features, record_len)
        split_support = (
            regroup(support_mask, record_len) if support_mask is not None else None
        )
        fused_scenes = []
        feature_elements = 0
        index_elements = 0
        density_elements = 0
        dense_feature_elements = 0
        message_count = 0
        per_sample_feature_elements = []
        self.latest_debug = None
        for batch_idx, scene_features in enumerate(split_features):
            num_agents = self._scene_length(record_len, batch_idx)
            scene_features = scene_features[:num_agents]
            scene_support = None
            if split_support is not None:
                scene_support = split_support[batch_idx][:num_agents].to(
                    device=scene_features.device, dtype=scene_features.dtype
                )
            (
                transmitted_features,
                transmitted_support,
                sample_feature_elements,
                sample_index_elements,
                sample_density_elements,
                sample_message_count,
                sample_dense_feature_elements,
            ) = self._build_scene_messages(scene_features, scene_support)
            feature_elements += sample_feature_elements
            index_elements += sample_index_elements
            density_elements += sample_density_elements
            message_count += sample_message_count
            dense_feature_elements += sample_dense_feature_elements
            per_sample_feature_elements.append(sample_feature_elements)
            aligned_features = self._align_scene(
                transmitted_features,
                ego_alignment_matrix,
                batch_idx,
                num_agents,
                (height, width),
            )
            if transmitted_support is None:
                aligned_support = aligned_features.new_zeros(
                    (num_agents, 1, height, width)
                )
                aligned_hard_mask = aligned_features.new_ones(
                    (num_agents, 1, height, width)
                )
            else:
                transmitted_hard_mask = (transmitted_support > 0).to(
                    dtype=transmitted_support.dtype
                )
                aligned_hard_mask = self._align_scene(
                    transmitted_hard_mask,
                    ego_alignment_matrix,
                    batch_idx,
                    num_agents,
                    (height, width),
                    mode="nearest",
                )
                aligned_hard_mask = (aligned_hard_mask > 0).to(
                    dtype=aligned_features.dtype
                )
                aligned_support = self._align_scene(
                    transmitted_support,
                    ego_alignment_matrix,
                    batch_idx,
                    num_agents,
                    (height, width),
                ).clamp(0.0, 1.0)
                # M is used after alignment only to keep Q and C outside the
                # received support exactly zero. The BEV feature itself was
                # hard-filtered once before cooperative fusion and is not
                # multiplied by M again here.
                aligned_support = aligned_support * aligned_hard_mask
            fused_scenes.append(
                self._fuse_scene(
                    aligned_features,
                    aligned_support,
                    batch_idx,
                    aligned_hard_mask=aligned_hard_mask,
                )
            )
        self._record_communication(
            feature_elements=feature_elements,
            index_elements=index_elements,
            metadata_elements=density_elements,
            dense_feature_elements=dense_feature_elements,
            message_count=message_count,
            batch_size=len(fused_scenes),
            mode=(
                (
                    "density_gated_spatially_sparse_bev"
                    if self.use_complementarity_guidance
                    else "density_gated_spatially_sparse_bev_no_complementarity"
                )
                if self.density_sparse_communication
                else (
                    "density_augmented_dense_bev"
                    if self.use_density_quality
                    else "full_dense_bev_no_q_no_m"
                )
            ),
            per_sample_feature_elements=per_sample_feature_elements,
            notes=(
                (
                    "Sender keeps only native BEV cells whose LiDAR density cue "
                    f"is > {self.communication_density_threshold:g}. Each selected "
                    "cell transmits one C-dimensional feature vector, one FP density "
                    "value, and two integer coordinates; ego-local data is excluded. "
                    + (
                        "CLC uses complementarity-guided attention."
                        if self.use_complementarity_guidance
                        else "CLC uses only the binary support mask; complementarity "
                        "attention bias/gating is disabled."
                    )
                )
                if self.density_sparse_communication
                else (
                    "Dense partner BEV feature and density cue; ego-local data is excluded."
                    if self.use_density_quality
                    else "Full dense partner BEV feature; Q and M are disabled."
                )
            ),
        )
        return torch.stack(fused_scenes, dim=0)


__all__ = [
    "ComplementarityGuidedCLCFusion",
]
