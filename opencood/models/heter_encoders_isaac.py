import copy
import math

import torch
import torch.nn.functional as F

from opencood.models.heter_encoders import (
    BEVFusion as BaseBEVFusion,
    ConvFuser,
    LiftSplatShoot as BaseLiftSplatShoot,
)
from opencood.models.heter_encoders import _find_encoder_class as _find_base_encoder_class
from opencood.utils.camera_utils import gen_dx_bx
from opencood.models.sub_modules.lss_submodule_isaac import (
    CamEncodeDINOv3ViTBLSSFPNIsaac,
    CamEncodeEfficientNetIsaac,
    CamEncodeResnet101Isaac,
    CamEncode_TimmFeatureMapMultiScaleIsaac,
)


class LiftSplatShootIsaac(BaseLiftSplatShoot):
    """Isaac-specific LSS encoder with multi-scale timm camera features."""

    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.grid_conf = args["grid_conf"]
        self.data_aug_conf = args["data_aug_conf"]
        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))
        self.depth_supervision = args["depth_supervision"]
        self.downsample = args["img_downsample"]
        self.camC = args["img_features"]
        self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(
            torch.device("cuda")
        )
        self.use_quickcumsum = True
        self.D, _, _, _ = self.frustum.shape
        self.camera_encoder_type = args["camera_encoder"]
        if self.camera_encoder_type == "EfficientNet":
            self.camencode = CamEncodeEfficientNetIsaac(
                self.D, self.camC, self.downsample,
                self.grid_conf["ddiscr"], self.grid_conf["mode"],
                args["use_depth_gt"], args["depth_supervision"],
            )
        elif self.camera_encoder_type == "Resnet101":
            self.camencode = CamEncodeResnet101Isaac(
                self.D, self.camC, self.downsample,
                self.grid_conf["ddiscr"], self.grid_conf["mode"],
                args["use_depth_gt"], args["depth_supervision"],
            )
        elif self.camera_encoder_type in (
            "DINOv3ViTBLSSFPNIsaac",
            "DINOv3VitBLSSFPNIsaac",
            "DINOv3ViTSPlusLSSFPNIsaac",
            "DINOv3ViTB16LSSFPNIsaac",
        ):
            self.camencode = CamEncodeDINOv3ViTBLSSFPNIsaac(
                self.D, self.camC, self.downsample,
                self.grid_conf["ddiscr"], self.grid_conf["mode"],
                args["use_depth_gt"], args["depth_supervision"],
                args.get("dino_args", {}),
            )
        elif self.camera_encoder_type == "TimmFeatureMapMultiScaleIsaac":
            self.camencode = CamEncode_TimmFeatureMapMultiScaleIsaac(
                self.D, self.camC, self.downsample,
                self.grid_conf["ddiscr"], self.grid_conf["mode"],
                args["use_depth_gt"], args["depth_supervision"],
                args.get("timm_args", {}),
            )
        else:
            raise ValueError(
                "LiftSplatShootIsaac supports EfficientNet, Resnet101, "
                "DINOv3ViTBLSSFPNIsaac/DINOv3ViTSPlusLSSFPNIsaac, or TimmFeatureMapMultiScaleIsaac "
                "camera encoders."
            )
    def forward(self, data_dict, modality_name):
        feature = super().forward(data_dict, modality_name)
        return feature



def _find_encoder_class_isaac(core_method):
    target_model_name = core_method.replace("_", "").lower()
    for name, cls in globals().items():
        if name.lower() == target_model_name:
            return cls
    return _find_base_encoder_class(core_method)


class BEVFusionIsaac(BaseBEVFusion):
    """Isaac BEVFusion encoder with LiDAR, camera, and optional depth supervision."""

    def __init__(self, args):
        torch.nn.Module.__init__(self)
        lidar_cfg = copy.deepcopy(args["lidar_encoder"])
        camera_cfg = copy.deepcopy(args["camera_encoder"])
        fuser_cfg = copy.deepcopy(args.get("fuser", {}))

        self.lidar_input_key = lidar_cfg.get("input_key", "lidar")
        self.camera_input_key = camera_cfg.get("input_key", "camera")

        lidar_cls = _find_encoder_class_isaac(lidar_cfg["core_method"])
        camera_cls = _find_encoder_class_isaac(camera_cfg["core_method"])
        self.lidar_encoder = lidar_cls(lidar_cfg["encoder_args"])
        self.camera_encoder = camera_cls(camera_cfg["encoder_args"])

        self.depth_supervision = bool(
            camera_cfg["encoder_args"].get("depth_supervision", False)
        )
        self.depth_items = None
        mask_args = args.get("lidar_support_mask", {}) or {}
        if isinstance(mask_args, bool):
            mask_args = {"enabled": mask_args}
        self.lidar_support_mask_enabled = bool(mask_args.get("enabled", False))
        self.lidar_support_mask_mode = str(
            mask_args.get("mode", "binary")
        ).lower()
        if self.lidar_support_mask_mode not in ("binary", "log_density"):
            raise ValueError(
                "lidar_support_mask.mode must be 'binary' or 'log_density'."
            )
        self.lidar_support_log_q95 = float(
            mask_args.get("log_density_q95", math.log1p(32.0))
        )
        if self.lidar_support_log_q95 <= 0.0:
            raise ValueError("lidar_support_mask.log_density_q95 must be positive.")
        self.lidar_support_apply_to_feature = bool(
            mask_args.get("apply_to_feature", True)
        )
        self.lidar_support_apply_stage = str(
            mask_args.get("apply_stage", "encoder_output")
        ).lower()
        if self.lidar_support_apply_stage not in (
            "encoder_output",
            "pre_cooperative_fusion",
        ):
            raise ValueError(
                "lidar_support_mask.apply_stage must be 'encoder_output' or "
                "'pre_cooperative_fusion'."
            )
        self.lidar_support_mask_dilation = int(mask_args.get("dilation_radius", 2))
        if self.lidar_support_mask_dilation < 0:
            raise ValueError("lidar_support_mask.dilation_radius must be non-negative.")
        lidar_encoder_args = lidar_cfg.get("encoder_args", {})
        voxel_size = lidar_encoder_args.get("voxel_size")
        lidar_range = lidar_encoder_args.get("lidar_range")
        if voxel_size is not None and lidar_range is not None:
            self.lidar_support_src_w = int(round((lidar_range[3] - lidar_range[0]) / voxel_size[0]))
            self.lidar_support_src_h = int(round((lidar_range[4] - lidar_range[1]) / voxel_size[1]))
        else:
            self.lidar_support_src_w = None
            self.lidar_support_src_h = None
        self.lidar_support_mask = None
        self.lidar_hard_support_mask = None
        in_channels = fuser_cfg.get(
            "in_channels",
            [lidar_cfg.get("out_channels"), camera_cfg.get("out_channels")],
        )
        if any(channel is None for channel in in_channels):
            raise ValueError(
                "BEVFusion fuser needs in_channels, or branch out_channels."
            )
        out_channels = fuser_cfg.get("out_channels", in_channels[0])
        self.fuser = ConvFuser(in_channels, out_channels)

        if lidar_cfg.get("freeze", False):
            self._freeze_module(self.lidar_encoder)
        if camera_cfg.get("freeze", False):
            self._freeze_module(self.camera_encoder)
        if fuser_cfg.get("freeze", False):
            self._freeze_module(self.fuser)

    def _make_lidar_support_mask(self, lidar_data, modality_name, feature):
        if not self.lidar_support_mask_enabled:
            return None
        inputs = lidar_data.get(f"inputs_{modality_name}", {})
        if not isinstance(inputs, dict) or "voxel_coords" not in inputs:
            return None
        coords = inputs["voxel_coords"]
        if coords.numel() == 0:
            return feature.new_zeros((feature.shape[0], 1, feature.shape[-2], feature.shape[-1]))

        coords = coords.long().to(feature.device)
        num_cav, _, height, width = feature.shape
        support = feature.new_zeros((num_cav, 1, height, width))
        batch_idx = coords[:, 0]
        valid = (batch_idx >= 0) & (batch_idx < num_cav)
        if not bool(valid.any().item()):
            return support

        point_counts = inputs.get("voxel_num_points")
        if point_counts is not None:
            point_counts = point_counts.to(device=feature.device, dtype=feature.dtype)
            if point_counts.numel() != coords.shape[0]:
                point_counts = None
        coords = coords[valid]
        if point_counts is not None:
            point_counts = point_counts[valid]
        batch_idx = coords[:, 0]
        y_idx = coords[:, 2]
        x_idx = coords[:, 3]
        src_h = self.lidar_support_src_h
        src_w = self.lidar_support_src_w
        if src_h is None or src_w is None:
            src_h = int(torch.clamp(y_idx.max() + 1, min=1).item())
            src_w = int(torch.clamp(x_idx.max() + 1, min=1).item())
        y_feat = torch.div(y_idx * height, src_h, rounding_mode="floor").clamp_(0, height - 1)
        x_feat = torch.div(x_idx * width, src_w, rounding_mode="floor").clamp_(0, width - 1)
        flat_index = batch_idx * (height * width) + y_feat * width + x_feat
        flat_support = support.view(-1)
        if self.lidar_support_mask_mode == "log_density":
            if point_counts is None:
                point_counts = feature.new_ones((coords.shape[0],))
            flat_support.scatter_add_(0, flat_index, point_counts.clamp_min(0.0))
            support = (
                torch.log1p(support) / self.lidar_support_log_q95
            ).clamp_(0.0, 1.0)
        else:
            flat_support[flat_index] = 1.0

        radius = self.lidar_support_mask_dilation
        if radius > 0:
            kernel_size = 2 * radius + 1
            support = F.max_pool2d(
                support, kernel_size=kernel_size, stride=1, padding=radius
            )
        return support

    @staticmethod
    def _hard_mask_from_support(support):
        """Convert continuous density support Q into binary occupancy M."""
        if support is None:
            return None
        return (support > 0).to(dtype=support.dtype)

    def forward(self, data_dict, modality_name):
        lidar_data = self._branch_data_dict(
            data_dict, modality_name, self.lidar_input_key
        )
        camera_data = self._branch_data_dict(
            data_dict, modality_name, self.camera_input_key
        )

        lidar_feature = self.lidar_encoder(lidar_data, modality_name)
        camera_feature = self.camera_encoder(camera_data, modality_name)

        if self.depth_supervision and hasattr(self.camera_encoder, "depth_items"):
            self.depth_items = self.camera_encoder.depth_items

        camera_feature = self._match_hw(camera_feature, lidar_feature.shape[-2:])
        self.lidar_support_mask = self._make_lidar_support_mask(
            lidar_data, modality_name, camera_feature
        )
        self.lidar_hard_support_mask = self._hard_mask_from_support(
            self.lidar_support_mask
        )
        fused_feature = self.fuser([lidar_feature, camera_feature])
        if (
            self.lidar_hard_support_mask is not None
            and self.lidar_support_apply_to_feature
            and self.lidar_support_apply_stage == "encoder_output"
        ):
            # M_k = 1[Q_k > 0] hard-filters the complete BEVFusion feature.
            # Keep lidar_support_mask unchanged so downstream RSP still receives
            # the continuous density cue Q_k rather than a binary replacement.
            fused_feature = fused_feature * self.lidar_hard_support_mask.to(
                dtype=fused_feature.dtype
            )
        return fused_feature
