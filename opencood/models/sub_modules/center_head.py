# -*- coding: utf-8 -*-
"""CenterPoint-style CenterHead modules for OpenCOOD BEV features."""

import copy

import torch.nn as nn
from torch.nn.init import kaiming_normal_


class SeparateHead(nn.Module):
    """Separate prediction heads used by CenterPoint/OpenPCDet CenterHead."""

    def __init__(
        self,
        input_channels,
        head_dict,
        init_bias=-2.19,
        use_bias=False,
        norm_func=None,
    ):
        super().__init__()
        self.head_dict = head_dict
        norm_func = nn.BatchNorm2d if norm_func is None else norm_func

        for head_name, head_cfg in self.head_dict.items():
            output_channels = head_cfg["out_channels"]
            num_conv = head_cfg.get("num_conv", 2)
            hidden_channels = head_cfg.get("head_conv_channels", input_channels)

            layers = []
            in_channels = input_channels
            for _ in range(num_conv - 1):
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels,
                            hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=use_bias,
                        ),
                        norm_func(hidden_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = hidden_channels

            layers.append(
                nn.Conv2d(
                    in_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
            )
            head = nn.Sequential(*layers)

            if head_name == "hm":
                head[-1].bias.data.fill_(init_bias)
            else:
                for module in head.modules():
                    if isinstance(module, nn.Conv2d):
                        kaiming_normal_(module.weight.data)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)

            self.__setattr__(head_name, head)

    def forward(self, x):
        return {head_name: getattr(self, head_name)(x) for head_name in self.head_dict}


class FusedSeparateHead(nn.Module):
    """Single-conv version for lightweight CenterHead configs.

    This keeps the public CenterHead outputs unchanged while replacing several
    one-layer prediction heads with one fused convolution and a cheap split.
    It is only valid when every head has ``num_conv == 1``.
    """

    def __init__(self, input_channels, head_dict, init_bias=-2.19):
        super().__init__()
        self.head_dict = head_dict
        self.head_names = list(head_dict.keys())
        self.out_channels = [
            head_dict[head_name]["out_channels"] for head_name in self.head_names
        ]

        self.pred = nn.Conv2d(
            input_channels,
            sum(self.out_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        kaiming_normal_(self.pred.weight.data)
        nn.init.constant_(self.pred.bias, 0)

        if "hm" in self.head_names:
            start = sum(
                self.out_channels[: self.head_names.index("hm")]
            )
            end = start + self.out_channels[self.head_names.index("hm")]
            self.pred.bias.data[start:end].fill_(init_bias)

    def forward(self, x):
        outputs = self.pred(x).split(self.out_channels, dim=1)
        return {
            head_name: head_output
            for head_name, head_output in zip(self.head_names, outputs)
        }


class CenterHead(nn.Module):
    """A compact CenterPoint CenterHead adapted to OpenCOOD BEV features.

    The head follows the standard CenterPoint/OpenPCDet split:
    heatmap, center offset, center z, log dimensions, and rot cos/sin.
    This module is intentionally model-only; target assignment and losses live
    in ``center_head_loss.py`` so it fits OpenCOOD's train loop.
    """

    DEFAULT_HEAD_DICT = {
        "center": {"out_channels": 2, "num_conv": 2},
        "center_z": {"out_channels": 1, "num_conv": 2},
        "dim": {"out_channels": 3, "num_conv": 2},
        "rot": {"out_channels": 2, "num_conv": 2},
    }

    def __init__(self, input_channels, args=None):
        super().__init__()
        args = {} if args is None else args
        shared_channels = args.get("shared_conv_channels", input_channels)
        num_classes = args.get("num_classes", 1)
        num_hm_conv = args.get("num_hm_conv", 2)
        fuse_final_conv = args.get(
            "fuse_final_conv", args.get("fused_final_conv", False)
        )
        init_bias = args.get("init_bias", -2.19)
        use_bias = args.get("use_bias_before_norm", False)
        bn_eps = args.get("bn_eps", 1e-5)
        bn_mom = args.get("bn_momentum", 0.1)

        def norm_func(channels):
            return nn.BatchNorm2d(channels, eps=bn_eps, momentum=bn_mom)

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                shared_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            ),
            norm_func(shared_channels),
            nn.ReLU(inplace=True),
        )

        separate_head_cfg = copy.deepcopy(args.get("separate_head", {}))
        head_dict = copy.deepcopy(
            separate_head_cfg.get("head_dict", self.DEFAULT_HEAD_DICT)
        )
        head_dict["hm"] = {
            "out_channels": num_classes,
            "num_conv": num_hm_conv,
            "head_conv_channels": separate_head_cfg.get(
                "head_conv_channels", shared_channels
            ),
        }

        for head_name, head_cfg in head_dict.items():
            head_cfg.setdefault("head_conv_channels", shared_channels)
            head_cfg.setdefault("num_conv", separate_head_cfg.get("num_conv", 2))

        if fuse_final_conv:
            invalid_heads = [
                head_name
                for head_name, head_cfg in head_dict.items()
                if head_cfg.get("num_conv", 2) != 1
            ]
            if invalid_heads:
                raise ValueError(
                    "CenterHead fuse_final_conv requires num_conv=1 for all "
                    f"heads, got {invalid_heads}."
                )
            self.head = FusedSeparateHead(
                input_channels=shared_channels,
                head_dict=head_dict,
                init_bias=init_bias,
            )
        else:
            self.head = SeparateHead(
                input_channels=shared_channels,
                head_dict=head_dict,
                init_bias=init_bias,
                use_bias=use_bias,
                norm_func=norm_func,
            )

    def forward(self, x):
        return self.head(self.shared_conv(x))


# Backward-compatible aliases for older logs/config snapshots.
SeparateHeadIsaac = SeparateHead
CenterHeadIsaac = CenterHead
