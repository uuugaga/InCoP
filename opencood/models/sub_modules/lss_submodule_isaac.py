import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.lss_submodule import (
    CamEncode,
    CamEncode_Resnet101,
    bin_depths,
)


class GeneralizedLSSFPNIsaac(nn.Module):
    # Small BEVFusion-style LSS FPN neck for Isaac camera encoders.

    def __init__(self, in_channels, out_channels=256, num_outs=None):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        self.in_channels = [int(channel) for channel in in_channels]
        self.out_channels = int(out_channels)
        self.num_outs = int(num_outs or len(self.in_channels))

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )
            for channel in self.in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.out_channels, self.out_channels,
                    kernel_size=3, padding=1, bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.out_channels * self.num_outs,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs, target_hw):
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f"GeneralizedLSSFPNIsaac expected {len(self.in_channels)} "
                f"features, got {len(inputs)}."
            )

        laterals = [
            lateral(feature)
            for lateral, feature in zip(self.lateral_convs, inputs)
        ]
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]
        outputs = outputs[:self.num_outs]
        outputs = [
            F.interpolate(
                output,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            if output.shape[-2:] != target_hw else output
            for output in outputs
        ]
        return self.fuse(torch.cat(outputs, dim=1))


class CamEncodeDINOv3ViTBLSSFPNIsaac(nn.Module):
    # DINOv3 ViT patch16 image encoder with a GeneralizedLSSFPN LSS neck.

    def __init__(self, D, C, downsample, ddiscr, mode,
                 use_gt_depth=False, depth_supervision=True, dino_args=None):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "CamEncodeDINOv3ViTBLSSFPNIsaac requires timm."
            ) from exc

        dino_args = dino_args or {}
        self.D = D
        self.C = C
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode
        self.use_gt_depth = use_gt_depth
        self.depth_supervision = depth_supervision
        self.freeze_backbone = bool(dino_args.get("freeze_backbone", True))
        self.model_name = dino_args.get("model_name", "vit_small_plus_patch16_dinov3")
        self.intermediate_indices = dino_args.get("intermediate_indices", [3, 7, 11])
        self.norm_intermediates = bool(dino_args.get("norm_intermediates", True))

        model_kwargs = dict(dino_args.get("model_kwargs", {}))
        cache_dir = dino_args.get("cache_dir", None)
        if cache_dir is None:
            cache_dir = model_kwargs.pop("cache_dir", None)
        else:
            model_kwargs.pop("cache_dir", None)
        self.trunk = timm.create_model(
            self.model_name,
            pretrained=bool(dino_args.get("pretrained", True)),
            checkpoint_path=dino_args.get("checkpoint_path", None) or None,
            cache_dir=cache_dir,
            num_classes=0,
            **model_kwargs,
        )
        self.embed_dim = int(getattr(self.trunk, "embed_dim"))
        patch_size = getattr(self.trunk.patch_embed, "patch_size", (16, 16))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = tuple(int(value) for value in patch_size)

        if isinstance(self.intermediate_indices, int):
            feature_count = int(self.intermediate_indices)
        else:
            feature_count = len(self.intermediate_indices)
        if feature_count <= 0:
            raise ValueError("intermediate_indices must select at least one layer.")

        neck_out_channels = int(dino_args.get("neck_out_channels", 256))
        neck_num_outs = int(dino_args.get("neck_num_outs", feature_count))
        self.neck = GeneralizedLSSFPNIsaac(
            [self.embed_dim] * feature_count,
            out_channels=neck_out_channels,
            num_outs=neck_num_outs,
        )
        if not use_gt_depth:
            self.depth_head = nn.Conv2d(neck_out_channels, self.D, kernel_size=1)
        self.image_head = nn.Conv2d(neck_out_channels, self.C, kernel_size=1)

        if self.freeze_backbone:
            self.trunk.eval()
            for param in self.trunk.parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.trunk.eval()
        return self

    def get_depth_dist(self, x):
        return F.softmax(x, dim=1)

    def get_gt_depth_dist(self, x):
        target = self.training
        torch.clamp_max_(x, self.d_max)
        depth_indices, mask = bin_depths(
            x, self.mode, self.d_min, self.d_max, self.num_bins, target=target
        )
        depth_indices = depth_indices[
            :, self.downsample // 2::self.downsample,
            self.downsample // 2::self.downsample,
        ]
        onehot_dist = F.one_hot(depth_indices.long()).permute(0, 3, 1, 2)
        if not target:
            mask = mask[
                :, self.downsample // 2::self.downsample,
                self.downsample // 2::self.downsample,
            ].unsqueeze(1)
            onehot_dist *= mask
        return onehot_dist, depth_indices

    def get_dino_lss_features(self, x):
        _, _, height, width = x.shape
        if height % self.patch_size[0] != 0 or width % self.patch_size[1] != 0:
            raise ValueError(
                f"DINOv3 input size {(height, width)} must be divisible by "
                f"patch size {self.patch_size}."
            )

        def forward_intermediates():
            return self.trunk.forward_intermediates(
                x,
                indices=self.intermediate_indices,
                norm=self.norm_intermediates,
                output_fmt="NCHW",
                intermediates_only=True,
            )

        if self.freeze_backbone:
            with torch.no_grad():
                features = forward_intermediates()
        else:
            features = forward_intermediates()

        target_hw = (height // self.downsample, width // self.downsample)
        return self.neck(features, target_hw)

    def forward(self, x):
        x_img_ = x[:, :3, :, :]
        features = self.get_dino_lss_features(x_img_)
        x_img = self.image_head(features)

        if self.depth_supervision or self.use_gt_depth:
            x_depth = x[:, 3, :, :]
            depth_gt, depth_gt_indices = self.get_gt_depth_dist(x_depth)

        if self.use_gt_depth:
            new_x = depth_gt.unsqueeze(1) * x_img.unsqueeze(2)
            return None, new_x

        depth_logit = self.depth_head(features)
        depth = self.get_depth_dist(depth_logit)
        new_x = depth.unsqueeze(1) * x_img.unsqueeze(2)
        if self.depth_supervision:
            return (depth_logit, depth_gt_indices), new_x
        return None, new_x


class CamEncodeEfficientNetIsaac(CamEncode):
    def __init__(self, D, C, downsample, ddiscr, mode,
                 use_gt_depth=False, depth_supervision=True):
        CamEncode.__init__(
            self, D, C, downsample, ddiscr, mode,
            use_gt_depth=use_gt_depth,
            depth_supervision=depth_supervision,
        )

    def forward(self, x):
        x_img_ = x[:, :3, :, :]
        features = self.get_eff_features(x_img_)
        x_img = self.image_head(features)

        if self.depth_supervision or self.use_gt_depth:
            x_depth = x[:, 3, :, :]
            depth_gt, depth_gt_indices = self.get_gt_depth_dist(x_depth)

        if self.use_gt_depth:
            new_x = depth_gt.unsqueeze(1) * x_img.unsqueeze(2)
            return None, new_x

        depth_logit = self.depth_head(features)
        depth = self.get_depth_dist(depth_logit)
        new_x = depth.unsqueeze(1) * x_img.unsqueeze(2)
        if self.depth_supervision:
            return (depth_logit, depth_gt_indices), new_x
        return None, new_x


class CamEncodeResnet101Isaac(CamEncode_Resnet101):
    def __init__(self, D, C, downsample, ddiscr, mode,
                 use_gt_depth=False, depth_supervision=True):
        CamEncode_Resnet101.__init__(
            self, D, C, downsample, ddiscr, mode,
            use_gt_depth=use_gt_depth,
            depth_supervision=depth_supervision,
        )

    def forward(self, x):
        x_img = x[:, :3, :, :].clone()
        features = self.get_resnet_features(x_img)
        x_img_feature = self.image_head(features)

        if self.depth_supervision or self.use_gt_depth:
            x_depth = x[:, 3, :, :]
            depth_gt, depth_gt_indices = self.get_gt_depth_dist(x_depth)

        if self.use_gt_depth:
            new_x = depth_gt.unsqueeze(1) * x_img_feature.unsqueeze(2)
            return None, new_x

        depth_logit = self.depth_head(features)
        depth = self.get_depth_dist(depth_logit)
        new_x = depth.unsqueeze(1) * x_img_feature.unsqueeze(2)
        if self.depth_supervision:
            return (depth_logit, depth_gt_indices), new_x
        return None, new_x


class CamEncode_TimmFeatureMapMultiScaleIsaac(nn.Module):
    """Isaac-specific multi-scale timm feature-map encoder for LSS.

    It keeps dense CNN-style DINOv3/ConvNeXt feature maps, upsamples selected
    stages to the LSS frustum stride, and fuses them before the depth and image
    heads. This mirrors the EfficientNet skip-fusion path without changing the
    shared OpenCOOD LSS modules.
    """

    def __init__(self, D, C, downsample, ddiscr, mode,
                 use_gt_depth=False, depth_supervision=True, timm_args=None):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "CamEncode_TimmFeatureMapMultiScaleIsaac requires timm."
            ) from exc

        timm_args = timm_args or {}
        self.D = D
        self.C = C
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode
        self.use_gt_depth = use_gt_depth
        self.depth_supervision = depth_supervision
        self.freeze_backbone = bool(timm_args.get("freeze_backbone", True))
        self.model_name = timm_args.get(
            "model_name", "convnext_tiny.dinov3_lvd1689m"
        )

        out_indices = timm_args.get("out_indices", [0, 1, 2])
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        self.out_indices = tuple(int(index) for index in out_indices)
        if not self.out_indices:
            raise ValueError("out_indices must contain at least one stage.")

        model_kwargs = timm_args.get("model_kwargs", {})
        self.trunk = timm.create_model(
            self.model_name,
            pretrained=bool(timm_args.get("pretrained", True)),
            checkpoint_path=timm_args.get("checkpoint_path", None) or None,
            features_only=True,
            out_indices=self.out_indices,
            **model_kwargs,
        )
        self.reductions = [int(v) for v in self.trunk.feature_info.reduction()]
        self.resize_to_lss_stride = bool(
            timm_args.get("resize_to_lss_stride", True)
        )
        if not self.resize_to_lss_stride and self.downsample not in self.reductions:
            raise ValueError(
                f"{self.model_name} out_indices={self.out_indices} reductions "
                f"{self.reductions} do not include LSS img_downsample "
                f"{self.downsample}."
            )

        channels = [int(v) for v in self.trunk.feature_info.channels()]
        branch_channels = int(timm_args.get("branch_channels", 128))
        adapter_channels = int(timm_args.get("adapter_channels", 256))
        self.stage_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(
                branch_channels * len(channels), adapter_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(adapter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_channels, adapter_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(adapter_channels),
            nn.ReLU(inplace=True),
        )

        if not use_gt_depth:
            self.depth_head = nn.Conv2d(adapter_channels, self.D, kernel_size=1)
        self.image_head = nn.Conv2d(adapter_channels, self.C, kernel_size=1)

        if self.freeze_backbone:
            self.trunk.eval()
            for param in self.trunk.parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.trunk.eval()
        return self

    def get_depth_dist(self, x):
        return F.softmax(x, dim=1)

    def get_gt_depth_dist(self, x):
        target = self.training
        torch.clamp_max_(x, self.d_max)
        depth_indices, mask = bin_depths(
            x, self.mode, self.d_min, self.d_max, self.num_bins, target=target
        )
        depth_indices = depth_indices[
            :, self.downsample // 2::self.downsample,
            self.downsample // 2::self.downsample,
        ]
        onehot_dist = F.one_hot(depth_indices.long()).permute(0, 3, 1, 2)
        if not target:
            mask = mask[
                :, self.downsample // 2::self.downsample,
                self.downsample // 2::self.downsample,
            ].unsqueeze(1)
            onehot_dist *= mask
        return onehot_dist, depth_indices

    def get_timm_features(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.trunk(x)
        else:
            features = self.trunk(x)

        target_h = x.shape[-2] // self.downsample
        target_w = x.shape[-1] // self.downsample
        fused = []
        for feature, adapter in zip(features, self.stage_adapters):
            feature = adapter(feature)
            if feature.shape[-2:] != (target_h, target_w):
                feature = F.interpolate(
                    feature, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )
            fused.append(feature)
        return self.fuse(torch.cat(fused, dim=1))

    def forward(self, x):
        x_img_ = x[:, :3, :, :]
        features = self.get_timm_features(x_img_)
        x_img = self.image_head(features)

        if self.depth_supervision or self.use_gt_depth:
            x_depth = x[:, 3, :, :]
            depth_gt, depth_gt_indices = self.get_gt_depth_dist(x_depth)

        if self.use_gt_depth:
            new_x = depth_gt.unsqueeze(1) * x_img.unsqueeze(2)
            return None, new_x

        depth_logit = self.depth_head(features)
        depth = self.get_depth_dist(depth_logit)
        new_x = depth.unsqueeze(1) * x_img.unsqueeze(2)
        if self.depth_supervision:
            return (depth_logit, depth_gt_indices), new_x
        return None, new_x
