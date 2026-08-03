# -*- coding: utf-8 -*-
"""High-resolution BEVFusion intermediate fusion for IsaacSim.

This model exchanges the raw BEVFusion encoder message, typically
``[sum_cav, 64, 224, 224]``, before any local BEV decoder is applied. The ego
side fuses those high-resolution messages and then runs one shared BEV decoder
and prediction head.
"""

from collections import Counter, OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from opencood.models.fuse_modules.fusion_in_one import (
    CoBEVT,
    ComplementarityGuidedCLCFusion,
    ERMVPFusion,
    V2XViTFusion,
    Where2commFusion,
    warp_feature,
    regroup,
)
from opencood.models.heter_encoders_isaac import _find_encoder_class_isaac
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.center_head import CenterHead
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.utils.model_utils import check_trainable_module
from opencood.utils.transformation_utils import normalize_pairwise_tfm_range


class PoseResidualHead(nn.Module):
    """Predict an SE(2) correction residual from a noisy aligned BEV pair."""

    def __init__(self, channels, args=None):
        super().__init__()
        args = args or {}
        hidden = int(args.get("hidden_dim", 64))
        self.max_translate = float(args.get("max_translate", 0.5))
        self.max_rotation_rad = math.radians(float(args.get("max_rotation_deg", 5.0)))
        self.use_uncertainty = bool(args.get("use_uncertainty", True))
        in_channels = channels * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 6),
        )

    def forward(self, ego_feature, cav_feature):
        pair = torch.cat(
            [
                ego_feature,
                cav_feature,
                cav_feature - ego_feature,
                (cav_feature - ego_feature).abs(),
            ],
            dim=1,
        )
        raw = self.net(pair)
        dx = torch.tanh(raw[:, 0]) * self.max_translate
        dy = torch.tanh(raw[:, 1]) * self.max_translate
        yaw = torch.tanh(raw[:, 2]) * self.max_rotation_rad
        pred = {
            "dx": dx,
            "dy": dy,
            "cos": torch.cos(yaw),
            "sin": torch.sin(yaw),
        }
        if self.use_uncertainty:
            pred["log_var"] = F.softplus(raw[:, 3:6]).clamp(max=5.0)
        return pred


class HeterModelBevfusionHighresIsaac(nn.Module):
    """Exchange raw high-resolution BEVFusion maps, then decode after fusion."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.modality_name_list = [
            key for key in args.keys() if key.startswith("m") and key[1:].isdigit()
        ]
        if not self.modality_name_list:
            raise ValueError("At least one modality block such as m1 is required.")

        self.ego_modality = args["ego_modality"]
        self.cav_range = args["lidar_range"]
        self.sensor_type_dict = OrderedDict()
        self.head_type = args.get("head_type", "anchor_based")
        self.min_size = args.get("min_size", 0.05)
        self.max_size = args.get("max_size", 6.0)

        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting["sensor_type"]
            self.sensor_type_dict[modality_name] = sensor_name
            model_setting["encoder_args"].setdefault(
                "lidar_support_mask", args.get("lidar_support_mask", {})
            )
            encoder_class = _find_encoder_class_isaac(model_setting["core_method"])
            setattr(
                self,
                f"encoder_{modality_name}",
                encoder_class(model_setting["encoder_args"]),
            )
            setattr(
                self,
                f"depth_supervision_{modality_name}",
                bool(model_setting["encoder_args"].get("depth_supervision", False)),
            )
            if sensor_name == "camera":
                camera_mask_args = model_setting["camera_mask_args"]
                setattr(
                    self,
                    f"crop_ratio_W_{modality_name}",
                    self.cav_range[3] / camera_mask_args["grid_conf"]["xbound"][1],
                )
                setattr(
                    self,
                    f"crop_ratio_H_{modality_name}",
                    self.cav_range[4] / camera_mask_args["grid_conf"]["ybound"][1],
                )

        mask_args = args.get("lidar_support_mask", {}) or {}
        if isinstance(mask_args, bool):
            mask_args = {"enabled": mask_args}
        self.lidar_support_mask_enabled = bool(mask_args.get("enabled", False))
        self.lidar_support_apply_to_feature = bool(
            mask_args.get("apply_to_feature", True)
        )
        self.lidar_support_apply_to_ego_feature = bool(
            mask_args.get("apply_to_ego_feature", True)
        )
        self.lidar_support_apply_stage = str(
            mask_args.get("apply_stage", "encoder_output")
        ).lower()
        if (
            self.lidar_support_apply_to_feature
            and not self.lidar_support_apply_to_ego_feature
            and self.lidar_support_apply_stage != "pre_cooperative_fusion"
        ):
            raise ValueError(
                "lidar_support_mask.apply_to_ego_feature=False requires "
                "apply_stage='pre_cooperative_fusion'."
            )
        self.fusion_net = self._build_fusion_net(args)
        message_backbone_args, decoder_args, shrink_args = self._resolve_decoder_args(args)
        self.message_backbone = (
            ResNetBEVBackbone(message_backbone_args)
            if message_backbone_args is not None else None
        )
        self.pre_fusion_message_backbone = bool(
            args.get("pre_fusion_message_backbone", False)
        )
        pose_error_args = args.get("pose_error_training", {}) or {}
        self.pose_error_training_enabled = bool(pose_error_args.get("enabled", False))
        self.clean_teacher_enabled = bool(
            pose_error_args.get("clean_teacher", self.pose_error_training_enabled)
        )
        pose_head_args = pose_error_args.get("pose_head", {}) or {}
        self.pose_head_enabled = bool(
            pose_head_args.get("enabled", self.pose_error_training_enabled)
        )
        if self.pose_head_enabled:
            pose_feat_dim = int(pose_head_args.get("feat_dim", 64))
            self.pose_residual_head = PoseResidualHead(pose_feat_dim, pose_head_args)
        ego_dropout_args = args.get("ego_feature_dropout", {}) or {}
        if isinstance(ego_dropout_args, (int, float)):
            ego_dropout_args = {"enabled": True, "prob": float(ego_dropout_args)}
        self.ego_feature_dropout_enabled = bool(
            ego_dropout_args.get("enabled", False)
        )
        self.ego_feature_dropout_prob = float(
            ego_dropout_args.get("prob", 0.0)
        )
        self.ego_feature_dropout_min_cav = int(
            ego_dropout_args.get("min_cav", 2)
        )
        self.ego_feature_dropout_scale_kept = bool(
            ego_dropout_args.get("scale_kept", False)
        )
        if not 0.0 <= self.ego_feature_dropout_prob < 1.0:
            raise ValueError("ego_feature_dropout.prob must be in [0, 1).")
        if self.pre_fusion_message_backbone and self.message_backbone is None:
            raise ValueError(
                "pre_fusion_message_backbone requires message_backbone_args "
                f"or {self.ego_modality}.backbone_args."
            )
        self.decoder = ResNetBEVBackbone(decoder_args)
        self.shrink_flag = shrink_args is not None
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(shrink_args)

        self.supervise_single = bool(args.get("supervise_single", False))
        self.single_head_shared = bool(args.get("single_head_shared", False))
        if self.supervise_single and not self.single_head_shared:
            self._build_single_heads(args)

        if self.head_type == "center_head":
            self.center_head = CenterHead(args["in_head"], args.get("center_head", {}))
        elif self.head_type == "anchor_based":
            self.cls_head = nn.Conv2d(args["in_head"], args["anchor_number"], kernel_size=1)
            self.reg_head = nn.Conv2d(args["in_head"], 7 * args["anchor_number"], kernel_size=1)
            self.dir_head = nn.Conv2d(
                args["in_head"],
                args["dir_args"]["num_bins"] * args["anchor_number"],
                kernel_size=1,
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

        check_trainable_module(self)

    def _attach_depth_items_for_modality(self, output_dict, modality_name):
        if not getattr(self, f"depth_supervision_{modality_name}", False):
            return
        encoder = getattr(self, f"encoder_{modality_name}")
        if hasattr(encoder, "depth_items"):
            output_dict[f"depth_items_{modality_name}"] = encoder.depth_items

    def _build_fusion_net(self, args):
        method = args["fusion_method"]
        if method == "ours":
            ours_args = dict(args.get("ours", {}))
            feat_dim = ours_args.get("feat_dim", 64)
            if isinstance(feat_dim, (list, tuple)):
                feat_dim = feat_dim[0]
            ours_args.setdefault("cav_lidar_range", self.cav_range)
            return ComplementarityGuidedCLCFusion(feat_dim, ours_args)
        if method == "v2xvit":
            return V2XViTFusion(args["v2xvit"])
        if method == "cobevt":
            return CoBEVT(args["cobevt"])
        if method == "where2comm":
            return Where2commFusion(args["where2comm"])
        if method == "ermvp":
            return ERMVPFusion(args["ermvp"])
        supported = ("v2xvit", "cobevt", "where2comm", "ermvp", "ours")
        raise ValueError(
            f"Unsupported fusion_method: {method}. Supported methods: {supported}"
        )

    def _resolve_decoder_args(self, args):
        source_setting = args[self.ego_modality]
        if "message_backbone_args" in args:
            message_backbone_args = args["message_backbone_args"]
        else:
            message_backbone_args = source_setting.get("backbone_args")
        if isinstance(message_backbone_args, str) and message_backbone_args.lower() == "none":
            message_backbone_args = None

        if "decoder_args" in args:
            decoder_args = args["decoder_args"]
        else:
            decoder_args = source_setting.get("layers_args")
        if decoder_args is None:
            raise KeyError(
                "High-res BEVFusion model needs decoder_args or "
                f"{self.ego_modality}.layers_args."
            )

        shrink_args = args.get("decoder_shrink_header", None)
        if shrink_args is None:
            shrink_args = args.get("shrink_header", None)
        if shrink_args is None:
            shrink_args = source_setting.get("shrink_header", None)
        return message_backbone_args, decoder_args, shrink_args

    def _build_single_heads(self, args):
        in_head_single = args.get("in_head_single", args["in_head"])
        if self.head_type == "center_head":
            self.center_head_single = CenterHead(
                in_head_single,
                args.get("center_head", {}),
            )
        else:
            self.cls_head_single = nn.Conv2d(
                in_head_single,
                args["anchor_number"],
                kernel_size=1,
            )
            self.reg_head_single = nn.Conv2d(
                in_head_single,
                args["anchor_number"] * 7,
                kernel_size=1,
            )
            self.dir_head_single = nn.Conv2d(
                in_head_single,
                args["anchor_number"] * args["dir_args"]["num_bins"],
                kernel_size=1,
            )

    def _crop_camera_feature_if_needed(self, feature, modality_name):
        if self.sensor_type_dict[modality_name] != "camera":
            return feature
        _, _, height, width = feature.shape
        target_h = int(height * getattr(self, f"crop_ratio_H_{modality_name}"))
        target_w = int(width * getattr(self, f"crop_ratio_W_{modality_name}"))
        return torchvision.transforms.CenterCrop((target_h, target_w))(feature)

    def _apply_message_backbone(self, message_feature):
        if self.message_backbone is None:
            return message_feature
        return self.message_backbone({"spatial_features": message_feature})[
            "spatial_features_2d"
        ]

    def _apply_ego_feature_dropout(self, message_feature, record_len):
        if (
            not self.training
            or not self.ego_feature_dropout_enabled
            or self.ego_feature_dropout_prob <= 0.0
        ):
            return message_feature

        if isinstance(record_len, torch.Tensor):
            lengths = [int(v) for v in record_len.detach().cpu().tolist()]
        else:
            lengths = [int(v) for v in record_len]

        ego_indices = []
        offset = 0
        for length in lengths:
            if length >= self.ego_feature_dropout_min_cav:
                ego_indices.append(offset)
            offset += length
        if not ego_indices:
            return message_feature

        drop_mask = (
            torch.rand(len(ego_indices), device=message_feature.device)
            < self.ego_feature_dropout_prob
        )
        if not bool(drop_mask.any().item()):
            return message_feature

        output_feature = message_feature.clone()
        ego_index_tensor = torch.as_tensor(
            ego_indices, dtype=torch.long, device=message_feature.device
        )
        dropped_indices = ego_index_tensor[drop_mask]
        output_feature[dropped_indices] = 0.0

        if self.ego_feature_dropout_scale_kept:
            kept_indices = ego_index_tensor[~drop_mask]
            if kept_indices.numel() > 0:
                output_feature[kept_indices] = output_feature[kept_indices] / (
                    1.0 - self.ego_feature_dropout_prob
                )
        return output_feature

    def _decode_message(self, message_feature, already_message_backbone=False):
        if not already_message_backbone:
            message_feature = self._apply_message_backbone(message_feature)
        decoded_feature = self.decoder({"spatial_features": message_feature})[
            "spatial_features_2d"
        ]
        if self.shrink_flag:
            decoded_feature = self.shrink_conv(decoded_feature)
        return decoded_feature

    def _attach_head_outputs(self, output_dict, fused_feature):
        if self.head_type == "center_head":
            pred_dict = self.center_head(fused_feature)
            output_dict.update({
                "cls_preds": pred_dict["hm"],
                "center_head_preds": pred_dict,
                "center_head_decode_args": {
                    "lidar_range": self.cav_range,
                    "min_size": self.min_size,
                    "max_size": self.max_size,
                },
                "center_preds": pred_dict["hm"],
                "offset_preds": pred_dict["center"],
                "z_preds": pred_dict["center_z"],
                "size_preds": pred_dict["dim"],
                "yaw_preds": pred_dict["rot"],
            })
            return

        output_dict.update({
            "cls_preds": self.cls_head(fused_feature),
            "reg_preds": self.reg_head(fused_feature),
            "dir_preds": self.dir_head(fused_feature),
        })

    def _attach_single_outputs(self, output_dict, single_feature):
        if not self.supervise_single:
            return
        if self.head_type == "center_head":
            center_head = (
                self.center_head
                if self.single_head_shared
                else self.center_head_single
            )
            pred_dict = center_head(single_feature)
            output_dict.update({
                "center_head_preds_single": pred_dict,
                "cls_preds_single": pred_dict["hm"],
            })
            return

        cls_head = self.cls_head if self.single_head_shared else self.cls_head_single
        reg_head = self.reg_head if self.single_head_shared else self.reg_head_single
        dir_head = self.dir_head if self.single_head_shared else self.dir_head_single
        output_dict.update({
            "cls_preds_single": cls_head(single_feature),
            "reg_preds_single": reg_head(single_feature),
            "dir_preds_single": dir_head(single_feature),
        })


    def _fusion_accepts_pairwise(self):
        return isinstance(self.fusion_net, ComplementarityGuidedCLCFusion)

    def _run_fusion(self, feature, record_len, affine_matrix, support_mask, pairwise_t_matrix):
        if self._fusion_accepts_pairwise():
            return self.fusion_net(
                feature,
                record_len,
                affine_matrix,
                support_mask,
                pairwise_t_matrix,
            )
        return self.fusion_net(feature, record_len, affine_matrix)

    def _attach_clean_teacher_outputs(
        self,
        output_dict,
        raw_feature_2d,
        record_len,
        support_mask_2d,
        clean_pairwise_t_matrix,
    ):
        if not self.clean_teacher_enabled or clean_pairwise_t_matrix is None:
            return
        clean_affine = normalize_pairwise_tfm_range(
            clean_pairwise_t_matrix,
            self.cav_range,
        )
        teacher_support = support_mask_2d.detach() if support_mask_2d is not None else None
        with torch.no_grad():
            clean_fused_raw = self._run_fusion(
                raw_feature_2d.detach(),
                record_len,
                clean_affine,
                teacher_support,
                clean_pairwise_t_matrix,
            )
            clean_decoded = self._decode_message(
                clean_fused_raw,
                already_message_backbone=self.pre_fusion_message_backbone,
            )
            if self.head_type == "center_head":
                clean_pred = self.center_head(clean_decoded)
                output_dict["clean_teacher"] = {
                    "center_head_preds": clean_pred,
                    "hm": clean_pred["hm"],
                }
            else:
                output_dict["clean_teacher"] = {
                    "cls_preds": self.cls_head(clean_decoded),
                    "reg_preds": self.reg_head(clean_decoded),
                    "dir_preds": self.dir_head(clean_decoded),
                }
        output_dict["fused_feature_clean"] = clean_fused_raw.detach()

    def _build_pose_correction(
        self,
        feature,
        record_len,
        affine_matrix,
        noisy_pairwise_t_matrix,
        clean_pairwise_t_matrix,
    ):
        if (
            not self.pose_head_enabled
            or clean_pairwise_t_matrix is None
            or noisy_pairwise_t_matrix is None
        ):
            return None
        _, _, height, width = feature.shape
        split_feature = regroup(feature, record_len)
        noisy_pairwise_t_matrix = noisy_pairwise_t_matrix.to(
            device=feature.device, dtype=feature.dtype
        )
        clean_pairwise_t_matrix = clean_pairwise_t_matrix.to(
            device=feature.device, dtype=feature.dtype
        )
        pred_items = {"dx": [], "dy": [], "cos": [], "sin": [], "log_var": []}
        target_items = {"dx": [], "dy": [], "cos": [], "sin": []}

        for batch_idx in range(affine_matrix.shape[0]):
            if torch.is_tensor(record_len[batch_idx]):
                num_agents = int(record_len[batch_idx].detach().cpu().item())
            else:
                num_agents = int(record_len[batch_idx])
            if num_agents <= 1:
                continue
            t_matrix = affine_matrix[batch_idx][:num_agents, :num_agents, :, :]
            x_warp = warp_affine_simple(
                split_feature[batch_idx],
                t_matrix[0, :, :, :],
                (height, width),
            )
            ego_feature = x_warp[0:1]
            for agent_idx in range(1, num_agents):
                pred = self.pose_residual_head(
                    ego_feature,
                    x_warp[agent_idx:agent_idx + 1],
                )
                target = self._pose_delta_target(
                    noisy_pairwise_t_matrix[batch_idx, agent_idx, 0],
                    clean_pairwise_t_matrix[batch_idx, agent_idx, 0],
                )
                for key in ("dx", "dy", "cos", "sin"):
                    pred_items[key].append(pred[key].reshape(-1))
                    target_items[key].append(target[key].reshape(-1))
                if "log_var" in pred:
                    pred_items["log_var"].append(pred["log_var"])

        if not pred_items["dx"]:
            return None
        refine = {
            key: torch.cat(value, dim=0)
            for key, value in pred_items.items()
            if value
        }
        refine_target = {
            key: torch.cat(value, dim=0)
            for key, value in target_items.items()
            if value
        }
        return {"refine": refine, "refine_target": refine_target}

    @staticmethod
    def _pose_delta_target(noisy_t_matrix, clean_t_matrix):
        delta = torch.matmul(clean_t_matrix, torch.linalg.inv(noisy_t_matrix))
        yaw = torch.atan2(delta[1, 0], delta[0, 0])
        return {
            "dx": delta[0, 3].view(1),
            "dy": delta[1, 3].view(1),
            "cos": torch.cos(yaw).view(1),
            "sin": torch.sin(yaw).view(1),
        }

    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict["agent_modality_list"]
        affine_matrix = normalize_pairwise_tfm_range(
            data_dict["pairwise_t_matrix"],
            self.cav_range,
        )
        record_len = data_dict["record_len"]
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        modality_support_mask_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            encoder = getattr(self, f"encoder_{modality_name}")
            feature = encoder(data_dict, modality_name)
            feature = self._crop_camera_feature_if_needed(feature, modality_name)
            modality_feature_dict[modality_name] = feature
            modality_support_mask_dict[modality_name] = getattr(
                encoder, "lidar_support_mask", None
            )
            if self.sensor_type_dict[modality_name] in ("camera", "multimodal"):
                self._attach_depth_items_for_modality(output_dict, modality_name)

        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        support_mask_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            support_mask = modality_support_mask_dict.get(modality_name)
            if support_mask is not None:
                support_mask_list.append(support_mask[feat_idx])
            counting_dict[modality_name] += 1
        heter_feature_2d = torch.stack(heter_feature_2d_list)
        support_mask_2d = None
        if self.lidar_support_mask_enabled and len(support_mask_list) == len(heter_feature_2d_list):
            support_mask_2d = torch.stack(support_mask_list)

        raw_feature_2d = heter_feature_2d
        if self.pre_fusion_message_backbone:
            raw_feature_2d = self._apply_message_backbone(raw_feature_2d)
        if support_mask_2d is not None and support_mask_2d.shape[-2:] != raw_feature_2d.shape[-2:]:
            support_mask_2d = F.interpolate(
                support_mask_2d, size=raw_feature_2d.shape[-2:], mode="nearest"
            )
        if (
            support_mask_2d is not None
            and self.lidar_support_apply_to_feature
            and self.lidar_support_apply_stage == "pre_cooperative_fusion"
        ):
            # Apply M exactly once, after the message backbone and immediately
            # before cooperative fusion.
            hard_support_2d = (support_mask_2d > 0).to(
                dtype=raw_feature_2d.dtype
            )
            if not self.lidar_support_apply_to_ego_feature:
                # The first feature in each regrouped scene is the local ego.
                # Preserve its complete BEV representation while leaving every
                # partner feature hard-filtered by its own LiDAR support mask.
                hard_support_2d = hard_support_2d.clone()
                ego_indices = torch.cumsum(record_len, dim=0) - record_len
                ego_indices = ego_indices.to(
                    device=hard_support_2d.device, dtype=torch.long
                )
                hard_support_2d[ego_indices] = 1
            raw_feature_2d = raw_feature_2d * hard_support_2d
        debug_features = None
        if getattr(self, "save_feature_debug", False):
            debug_features = {
                "ego_bev_feature_raw": heter_feature_2d[0].detach().float().cpu(),
                "ego_bev_feature": raw_feature_2d[0].detach().float().cpu(),
            }
            first_record_len = int(record_len[0].detach().cpu().item())
            if first_record_len > 1:
                debug_features["co_bev_feature_raw"] = (
                    heter_feature_2d[1].detach().float().cpu()
                )
                aligned_raw_feature_2d = warp_feature(
                    raw_feature_2d, record_len, affine_matrix
                )
                debug_features["co_bev_feature"] = (
                    aligned_raw_feature_2d[1].detach().float().cpu()
                )

        if self.supervise_single:
            single_decoded_feature_2d = self._decode_message(
                raw_feature_2d,
                already_message_backbone=self.pre_fusion_message_backbone,
            )
            self._attach_single_outputs(output_dict, single_decoded_feature_2d)

        clean_pairwise_t_matrix = data_dict.get("pairwise_t_matrix_clean", None)
        if clean_pairwise_t_matrix is None and "label_dict" in data_dict:
            clean_pairwise_t_matrix = data_dict["label_dict"].get(
                "pairwise_t_matrix_clean", None
            )

        fusion_input_2d = self._apply_ego_feature_dropout(
            raw_feature_2d, record_len
        )
        if hasattr(self.fusion_net, "save_debug"):
            self.fusion_net.save_debug = debug_features is not None
        fused_raw_feature = self._run_fusion(
            fusion_input_2d,
            record_len,
            affine_matrix,
            support_mask_2d,
            data_dict.get("pairwise_t_matrix", None),
        )
        communication_getter = getattr(
            self.fusion_net, "get_communication_stats", None
        )
        if callable(communication_getter):
            communication_stats = communication_getter()
            if isinstance(communication_stats, dict):
                output_dict["communication"] = communication_stats
        opg_loss = getattr(self.fusion_net, "latest_opg_loss", None)
        if self.training and isinstance(opg_loss, torch.Tensor):
            output_dict["opg_loss"] = opg_loss
        if self.training and self.pose_error_training_enabled:
            output_dict["fused_feature_noisy"] = fused_raw_feature
            self._attach_clean_teacher_outputs(
                output_dict,
                raw_feature_2d,
                record_len,
                support_mask_2d,
                clean_pairwise_t_matrix,
            )
            pose_correction = self._build_pose_correction(
                raw_feature_2d,
                record_len,
                affine_matrix,
                data_dict.get("pairwise_t_matrix", None),
                clean_pairwise_t_matrix,
            )
            if pose_correction is not None:
                output_dict["pose_correction"] = pose_correction
        decoded_feature_2d = self._decode_message(
            fused_raw_feature,
            already_message_backbone=self.pre_fusion_message_backbone,
        )
        if debug_features is not None:
            fusion_debug = getattr(self.fusion_net, "latest_debug", None)
            if isinstance(fusion_debug, dict):
                for feature_name, feature_value in fusion_debug.items():
                    if not isinstance(feature_value, torch.Tensor):
                        continue
                    if feature_value.ndim >= 4:
                        feature_value = feature_value[0]
                    debug_features[feature_name] = feature_value.detach().float().cpu()
            debug_features["fusion_bev_feature"] = (
                fused_raw_feature[0].detach().float().cpu()
            )
            debug_features["decoder_bev_feature"] = (
                decoded_feature_2d[0].detach().float().cpu()
            )
            output_dict["debug_features"] = debug_features
        self._attach_head_outputs(output_dict, decoded_feature_2d)
        return output_dict
